"""
SiftOps — Evaluation Pipeline
==============================
Loads evaluation_dataset.csv, queries the live backend, computes metrics,
and writes results to evaluation/results.csv, metrics.json, failures.csv,
and (with --compare) comparison.csv.

Usage:
    # Standard dense-only evaluation
    python evaluation/run_evaluation.py

    # BM25 vs Dense comparison
    python evaluation/run_evaluation.py --compare

    # Point at a non-default backend
    python evaluation/run_evaluation.py --base-url http://localhost:8000

    # Point at a different dataset
    python evaluation/run_evaluation.py --dataset ./my_eval.csv
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tqdm import tqdm

# ── Make sure the project root is on sys.path so we can import backend modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("siftops.eval")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EVAL_DIR / "results.csv"
METRICS_JSON = EVAL_DIR / "metrics.json"
FAILURES_CSV = EVAL_DIR / "failures.csv"
COMPARE_CSV = EVAL_DIR / "comparison.csv"
BM25_INDEX = PROJECT_ROOT / "bm25_index.pkl"

# ── Error categories ──────────────────────────────────────────────────────────
ERR_RETRIEVAL_FAILURE = "Retrieval Failure"
ERR_SEMANTIC_MISMATCH = "Semantic Mismatch"
ERR_AMBIGUOUS_QUERY = "Ambiguous Query"
ERR_REFUSAL_FAILURE = "Refusal Failure"
ERR_MISSING_CITATION = "Missing Citation"
ERR_CORRECT = "Correct"


# ─────────────────────────────────────────────────────────────────────────────
# Backend client
# ─────────────────────────────────────────────────────────────────────────────
class BackendClient:
    """Thin synchronous wrapper around the SiftOps FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def health(self) -> dict[str, Any]:
        r = self._client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def search(self, query: str, top_k: int = 5) -> Any:
        r = self._client.get(
            f"{self.base_url}/search",
            params={"q": query, "top_k": top_k},
        )
        r.raise_for_status()
        return r.json()

    def chat(self, question: str, top_k: int = 5) -> dict[str, Any]:
        r = self._client.post(
            f"{self.base_url}/chat",
            json={"question": question, "top_k": top_k},
        )
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────
def row_value(row: pd.Series, *keys: str, default: Any = "") -> Any:
    """Return the first existing key from a row, supporting multiple casings."""
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return default


def normalize(name: str) -> str:
    """Lowercase + strip + replace spaces for loose filename matching."""
    return str(name).lower().strip().replace(" ", "_")


def normalize_stem(name: str) -> str:
    return Path(str(name)).stem.lower().strip().replace(" ", "_")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "out_of_scope", "out-of-scope"}


def is_out_of_scope_row(row: pd.Series) -> bool:
    """Detect out-of-scope / refusal rows from either explicit or inferred columns."""
    for key in ("is_out_of_scope", "out_of_scope", "Out_of_Scope", "OUT_OF_SCOPE"):
        if key in row:
            return coerce_bool(row[key])

    category = normalize(str(row_value(row, "category", "Category", default="")))
    expected = str(row_value(row, "expected_document", "Expected_Document", default="")).strip()

    if expected.upper() == "REFUSE":
        return True

    return category in {"refusal", "out_of_scope", "out-of-scope", "out of scope"}


def extract_search_hits(payload: Any) -> list[dict[str, Any]]:
    """Handle either {'results': [...]} or raw list payloads."""
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("results", [])
    else:
        items = []

    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def extract_source(hit: Any) -> str:
    """Robustly extract a document/source name from a search hit of any shape."""
    if hit is None:
        return ""

    if isinstance(hit, dict):
        for key in ("source", "filename", "doc", "document", "name"):
            val = hit.get(key)
            if val:
                return str(val)
        return ""

    if isinstance(hit, (list, tuple)):
        if not hit:
            return ""
        for idx in (0, 1):
            if idx < len(hit) and hit[idx]:
                return str(hit[idx])
        return str(hit[0])

    for attr in ("source", "filename", "doc", "document", "name"):
        if hasattr(hit, attr):
            val = getattr(hit, attr)
            if val:
                return str(val)

    return str(hit)


def retrieve_list_from_search_response(search_payload: Any) -> list[str]:
    hits = extract_search_hits(search_payload)
    return [extract_source(h) for h in hits if extract_source(h)]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def hit_at_k(retrieved_sources: list[str], expected: str, k: int) -> bool:
    """True if expected document appears in the top-k retrieved sources."""
    top_k = retrieved_sources[:k]
    exp_n = normalize(expected)
    exp_stem = normalize_stem(expected)
    return any((exp_n in normalize(s)) or (exp_stem in normalize_stem(s)) for s in top_k)


def source_match(answer: str, expected: str, chat_sources: list[str] | None = None) -> bool:
    """True if the answer or citations mention the expected document name."""
    if not expected or str(expected).strip().upper() == "N/A":
        return False

    expected_norm = normalize_stem(expected)
    candidates = [answer or ""]
    if chat_sources:
        candidates.extend(chat_sources)

    return any(expected_norm in normalize_stem(c) or expected_norm in normalize(c) for c in candidates)


def classify_error(
    is_out_of_scope: bool,
    refused: bool,
    h1: bool,
    h5: bool,
    src_match: bool,
    question: str,
) -> str:
    """Assign a single error category to a failed question."""
    if is_out_of_scope:
        return ERR_CORRECT if refused else ERR_REFUSAL_FAILURE

    if not h1 and not h5:
        # No overlap at all — if the query is very short it is often ambiguous.
        if len(str(question).split()) < 4:
            return ERR_AMBIGUOUS_QUERY
        return ERR_RETRIEVAL_FAILURE

    if h5 and not h1:
        # Right doc found but not top-ranked.
        return ERR_SEMANTIC_MISMATCH

    if h1 and not src_match:
        return ERR_MISSING_CITATION

    return ERR_CORRECT


# ─────────────────────────────────────────────────────────────────────────────
# Dense evaluation (calls live backend)
# ─────────────────────────────────────────────────────────────────────────────
def run_dense_eval(
    df: pd.DataFrame,
    client: BackendClient,
    top_k: int = 5,
    delay: float = 0.2,
) -> list[dict[str, Any]]:
    """
    For each row in df, call /search and /chat, record results.
    Returns a list of result dicts (one per question).
    """
    rows: list[dict[str, Any]] = []

    for _, q in tqdm(df.iterrows(), total=len(df), desc="Dense eval"):
        qid = row_value(q, "question_id", "ID", default="")
        question = row_value(q, "question", "Question", default="")
        expected = row_value(q, "expected_document", "Expected_Document", default="N/A")
        category = row_value(q, "category", "Category", default="unknown")
        out_scope = is_out_of_scope_row(q)

        row: dict[str, Any] = {
            "question_id": qid,
            "question": question,
            "expected_document": expected,
            "category": category,
            "is_out_of_scope": out_scope,
        }

        # ── /search ──────────────────────────────────────────────────────────
        try:
            search_payload = client.search(question, top_k=top_k)
            retrieved = retrieve_list_from_search_response(search_payload)
            top1_source = retrieved[0] if retrieved else ""
            search_ok = True
        except Exception as exc:
            log.warning("[%s] /search failed: %s", qid, exc)
            retrieved = []
            top1_source = ""
            search_ok = False

        h1 = hit_at_k(retrieved, expected, 1) if not out_scope else False
        h5 = hit_at_k(retrieved, expected, 5) if not out_scope else False

        row.update(
            {
                "search_success": search_ok,
                "retrieved_top1": top1_source,
                "retrieved_sources": "|".join(retrieved[:top_k]),
                "hit_at_1": h1,
                "hit_at_5": h5,
            }
        )

        # ── /chat ─────────────────────────────────────────────────────────────
        try:
            chat_resp = client.chat(question, top_k=top_k)
            answer = str(chat_resp.get("answer", "") or "")
            refused = bool(chat_resp.get("refused", False))
            chat_ok = True
            chat_sources = retrieve_chat_sources(chat_resp)
        except Exception as exc:
            log.warning("[%s] /chat failed: %s", qid, exc)
            answer = ""
            refused = False
            chat_ok = False
            chat_sources = []

        sm = source_match(answer, expected, chat_sources=chat_sources) if not out_scope else False
        error_cat = classify_error(out_scope, refused, h1, h5, sm, question)

        row.update(
            {
                "chat_success": chat_ok,
                "answer_preview": answer[:200],
                "refused": refused,
                "source_match": sm,
                "error_category": error_cat,
            }
        )

        rows.append(row)
        time.sleep(delay)  # be polite to the backend

    return rows


def retrieve_chat_sources(chat_resp: dict[str, Any]) -> list[str]:
    sources = chat_resp.get("sources", [])
    out: list[str] = []
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, dict):
                src = item.get("source") or item.get("filename") or item.get("doc") or item.get("document")
                if src:
                    out.append(str(src))
            elif item:
                out.append(str(item))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BM25 evaluation (local, no backend required)
# ─────────────────────────────────────────────────────────────────────────────
def _call_bm25_query(retriever: Any, question: str, top_k: int):
    """Call retriever in a version-agnostic way."""
    for method_name in ("query", "search", "retrieve", "rank"):
        fn = getattr(retriever, method_name, None)
        if fn is None:
            continue

        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            if "top_k" in params:
                return fn(question, top_k=top_k)
            if "k" in params:
                return fn(question, k=top_k)
            if len(params) >= 2:
                return fn(question, top_k)
            return fn(question)
        except TypeError:
            continue

    raise AttributeError("No compatible BM25 query method found (query/search/retrieve/rank).")


def run_bm25_eval(
    df: pd.DataFrame,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Run BM25 retrieval over the local index.
    Returns list of result dicts matching dense output shape.
    """
    from backend.bm25_index import BM25Retriever  # noqa: PLC0415

    retriever = BM25Retriever(index_path=BM25_INDEX)
    rows: list[dict[str, Any]] = []

    for _, q in tqdm(df.iterrows(), total=len(df), desc="BM25 eval"):
        qid = row_value(q, "question_id", "ID", default="")
        question = row_value(q, "question", "Question", default="")
        expected = row_value(q, "expected_document", "Expected_Document", default="N/A")
        category = row_value(q, "category", "Category", default="unknown")
        out_scope = is_out_of_scope_row(q)

        try:
            hits = _call_bm25_query(retriever, question, top_k)
            retrieved = [extract_source(h) for h in (hits or []) if extract_source(h)]
            top1 = retrieved[0] if retrieved else ""
            ok = True
        except Exception as exc:
            log.warning("[%s] BM25 query failed: %s", qid, exc)
            retrieved = []
            top1 = ""
            ok = False

        h1 = hit_at_k(retrieved, expected, 1) if not out_scope else False
        h5 = hit_at_k(retrieved, expected, 5) if not out_scope else False

        rows.append(
            {
                "question_id": qid,
                "question": question,
                "expected_document": expected,
                "category": category,
                "is_out_of_scope": out_scope,
                "search_success": ok,
                "retrieved_top1": top1,
                "retrieved_sources": "|".join(retrieved[:top_k]),
                "hit_at_1": h1,
                "hit_at_5": h5,
            }
        )

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute overall + per-category metrics from a list of result dicts.
    Returns a nested dict ready for JSON serialisation.
    """
    df = pd.DataFrame(results)

    if df.empty:
        return {
            "overall": {
                "total_questions": 0,
                "in_scope_questions": 0,
                "out_of_scope_questions": 0,
                "search_success_rate": 0.0,
                "chat_success_rate": 0.0,
                "hit_at_1": 0.0,
                "hit_at_5": 0.0,
                "refusal_accuracy": None,
                "source_match_rate": None,
            },
            "per_category": {},
            "error_distribution": {},
        }

    # Separate in-scope and out-of-scope questions
    in_scope = df[~df["is_out_of_scope"]]
    oos = df[df["is_out_of_scope"]]

    def safe_mean(series: pd.Series) -> float:
        return round(float(series.mean()), 4) if len(series) > 0 else 0.0

    overall = {
        "total_questions": int(len(df)),
        "in_scope_questions": int(len(in_scope)),
        "out_of_scope_questions": int(len(oos)),
        "search_success_rate": safe_mean(df["search_success"]),
        "chat_success_rate": safe_mean(df["chat_success"]) if "chat_success" in df.columns else None,
        "hit_at_1": safe_mean(in_scope["hit_at_1"]),
        "hit_at_5": safe_mean(in_scope["hit_at_5"]),
        "refusal_accuracy": safe_mean(oos["refused"]) if len(oos) > 0 and "refused" in df.columns else None,
        "source_match_rate": safe_mean(in_scope["source_match"]) if "source_match" in df.columns else None,
    }

    # Per-category (in-scope only)
    per_category: dict[str, Any] = {}
    for cat, grp in in_scope.groupby("category"):
        per_category[str(cat)] = {
            "count": int(len(grp)),
            "hit_at_1": safe_mean(grp["hit_at_1"]),
            "hit_at_5": safe_mean(grp["hit_at_5"]),
            "source_match_rate": safe_mean(grp["source_match"]) if "source_match" in grp.columns else None,
        }

    # Error distribution
    if "error_category" in df.columns:
        error_dist = df["error_category"].value_counts().to_dict()
    else:
        error_dist = {}

    return {
        "overall": overall,
        "per_category": per_category,
        "error_distribution": error_dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Failure extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_failures(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the rows where error_category != Correct."""
    failures: list[dict[str, Any]] = []
    for r in results:
        cat = r.get("error_category", "")
        if cat != ERR_CORRECT:
            failures.append(
                {
                    "question_id": r.get("question_id", ""),
                    "question": r.get("question", ""),
                    "expected_document": r.get("expected_document", "N/A"),
                    "retrieved_document": r.get("retrieved_top1", ""),
                    "category": r.get("category", ""),
                    "error_category": cat,
                    "refused": r.get("refused", False),
                    "answer_preview": r.get("answer_preview", ""),
                }
            )
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# BM25 vs Dense comparison
# ─────────────────────────────────────────────────────────────────────────────
def build_comparison(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Merge BM25 and dense results side-by-side, compute per-question winner,
    and return (comparison_rows, comparison_metrics).
    """
    bm25_map = {r["question_id"]: r for r in bm25_results}
    dense_map = {r["question_id"]: r for r in dense_results}

    all_ids = sorted(set(bm25_map) | set(dense_map))
    rows: list[dict[str, Any]] = []

    for qid in all_ids:
        b = bm25_map.get(qid, {})
        d = dense_map.get(qid, {})

        bm25_h1 = b.get("hit_at_1", False)
        dense_h1 = d.get("hit_at_1", False)

        if dense_h1 and not bm25_h1:
            winner = "Dense"
        elif bm25_h1 and not dense_h1:
            winner = "BM25"
        elif bm25_h1 and dense_h1:
            winner = "Both"
        else:
            winner = "Neither"

        rows.append(
            {
                "question_id": qid,
                "question": b.get("question", d.get("question", "")),
                "expected_document": b.get("expected_document", "N/A"),
                "bm25_top1": b.get("retrieved_top1", ""),
                "dense_top1": d.get("retrieved_top1", ""),
                "bm25_hit_at_1": bm25_h1,
                "dense_hit_at_1": dense_h1,
                "bm25_hit_at_5": b.get("hit_at_5", False),
                "dense_hit_at_5": d.get("hit_at_5", False),
                "winner": winner,
            }
        )

    in_scope = [
        r for r in rows
        if not bm25_map.get(r["question_id"], {}).get("is_out_of_scope", False)
    ]

    def mean(vals: list[bool]) -> float:
        return round(sum(bool(v) for v in vals) / len(vals), 4) if vals else 0.0

    metrics = {
        "bm25_hit_at_1": mean([r["bm25_hit_at_1"] for r in in_scope]),
        "dense_hit_at_1": mean([r["dense_hit_at_1"] for r in in_scope]),
        "bm25_hit_at_5": mean([r["bm25_hit_at_5"] for r in in_scope]),
        "dense_hit_at_5": mean([r["dense_hit_at_5"] for r in in_scope]),
        "dense_wins": sum(1 for r in in_scope if r["winner"] == "Dense"),
        "bm25_wins": sum(1 for r in in_scope if r["winner"] == "BM25"),
        "both_win": sum(1 for r in in_scope if r["winner"] == "Both"),
        "neither_wins": sum(1 for r in in_scope if r["winner"] == "Neither"),
    }

    return rows, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="SiftOps Evaluation Pipeline")
    parser.add_argument("--base-url", default="http://localhost:8000", help="FastAPI backend URL")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "evaluation_dataset.csv"), help="Path to evaluation CSV")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between requests (be kind to local backend)")
    parser.add_argument("--compare", action="store_true", help="Run BM25 vs Dense comparison")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    df = pd.read_csv(dataset_path)

    # Normalise boolean column if present; otherwise derive from category/expected doc.
    if "is_out_of_scope" in df.columns:
        df["is_out_of_scope"] = df["is_out_of_scope"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    else:
        df["is_out_of_scope"] = df.apply(is_out_of_scope_row, axis=1)

    log.info("Loaded %d questions from %s", len(df), dataset_path)

    # ── Health check ─────────────────────────────────────────────────────────
    client = BackendClient(base_url=args.base_url)
    try:
        h = client.health()
        log.info("Backend healthy: %s | points=%s", h.get("status"), h.get("points"))
    except Exception as exc:
        log.error("Backend not reachable at %s — %s", args.base_url, exc)
        log.error("Start the backend with: uvicorn backend.main:app --reload")
        sys.exit(1)

    # ── Dense evaluation ─────────────────────────────────────────────────────
    log.info("Running dense (FastEmbed + Qdrant) evaluation ...")
    dense_results = run_dense_eval(df, client, top_k=args.top_k, delay=args.delay)
    client.close()

    # Save per-question results
    results_df = pd.DataFrame(dense_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    log.info("Results saved → %s", RESULTS_CSV)

    # Compute + save metrics
    metrics = compute_metrics(dense_results)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info("Metrics saved → %s", METRICS_JSON)

    # Save failures
    failures = extract_failures(dense_results)
    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_CSV, index=False)
        log.info("Failures saved → %s  (%d rows)", FAILURES_CSV, len(failures))
    else:
        log.info("No failures detected!")

    # ── BM25 comparison ───────────────────────────────────────────────────────
    if args.compare:
        if not BM25_INDEX.exists():
            log.error(
                "BM25 index not found at %s.\n"
                "Build it first with: python backend/bm25_index.py",
                BM25_INDEX,
            )
            sys.exit(1)

        log.info("Running BM25 baseline evaluation ...")
        bm25_results = run_bm25_eval(df, top_k=args.top_k)
        comparison_rows, comp_metrics = build_comparison(bm25_results, dense_results)
        pd.DataFrame(comparison_rows).to_csv(COMPARE_CSV, index=False)
        log.info("Comparison saved → %s", COMPARE_CSV)

        # Append comparison metrics to metrics.json
        metrics["retrieval_comparison"] = comp_metrics
        with open(METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        log.info("Comparison metrics merged into %s", METRICS_JSON)

    # ── Summary ───────────────────────────────────────────────────────────────
    ov = metrics["overall"]
    print("\n" + "=" * 60)
    print("  SiftOps Evaluation Summary")
    print("=" * 60)
    print(f"  Questions evaluated : {ov['total_questions']}")
    print(f"  In-scope            : {ov['in_scope_questions']}")
    print(f"  Out-of-scope        : {ov['out_of_scope_questions']}")
    print(f"  Search success rate : {ov['search_success_rate']:.1%}")
    print(f"  Chat success rate   : {ov.get('chat_success_rate') if ov.get('chat_success_rate') is not None else 'N/A'}")
    print(f"  Hit@1               : {ov['hit_at_1']:.1%}")
    print(f"  Hit@5               : {ov['hit_at_5']:.1%}")
    print(f"  Refusal accuracy    : {ov.get('refusal_accuracy') if ov.get('refusal_accuracy') is not None else 'N/A'}")
    print(f"  Source match rate   : {ov.get('source_match_rate') if ov.get('source_match_rate') is not None else 'N/A'}")
    if args.compare and "retrieval_comparison" in metrics:
        cm = metrics["retrieval_comparison"]
        print(f"\n  BM25  Hit@1 / Hit@5 : {cm['bm25_hit_at_1']:.1%} / {cm['bm25_hit_at_5']:.1%}")
        print(f"  Dense Hit@1 / Hit@5 : {cm['dense_hit_at_1']:.1%} / {cm['dense_hit_at_5']:.1%}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()


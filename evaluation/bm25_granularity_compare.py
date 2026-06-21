"""
SiftOps — BM25 granularity-matched comparison (fairness analysis)
=================================================================
The default BM25 baseline (backend/bm25_index.py) indexes WHOLE documents,
while the dense pipeline indexes 512-word chunks. That makes the headline
BM25-vs-Dense comparison partly a granularity artifact rather than a pure
retrieval-quality difference.

This script rebuilds BM25 over the SAME 512-word / 64-word-overlap chunks the
dense pipeline uses, scores it with the identical scorer (run_evaluation.hit_at_k),
and reports a granularity-matched comparison against the dense numbers already
saved in evaluation/results.csv.

It is read-only with respect to the canonical artifacts: it prints a table and
writes only evaluation/comparison_chunked.csv. The whole-document baseline in
comparison.csv / metrics.json is left untouched.

Usage:
    python evaluation/bm25_granularity_compare.py
    python evaluation/bm25_granularity_compare.py --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.bm25_index import preprocess  # noqa: E402
from backend.ingest import chunk_text, clean_text, extract_pages_from_pdf  # noqa: E402
from evaluation.run_evaluation import hit_at_k  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_CSV = EVAL_DIR / "results.csv"
OUT_CSV = EVAL_DIR / "comparison_chunked.csv"
DATA_DIR = PROJECT_ROOT / "data"


def build_chunk_corpus(chunk_size: int, overlap: int) -> list[dict]:
    """One entry per 512-word chunk, mirroring the dense ingest pipeline."""
    corpus: list[dict] = []
    for cat_dir in sorted(DATA_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        for pdf in cat_dir.glob("**/*.pdf"):
            pages = extract_pages_from_pdf(pdf)
            text = " ".join(clean_text(t) for _, t in pages)
            for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                corpus.append({"source": pdf.name, "chunk_index": i, "tokens": preprocess(chunk)})
    return corpus


def chunk_bm25_retrieve(bm25: BM25Okapi, corpus: list[dict], question: str, top_k: int) -> list[str]:
    """Return the source filenames of the top_k chunks, in rank order.

    Mirrors the dense pipeline, which scores chunks and returns one source per
    retrieved chunk (no dedup) before hit_at_k slices the top-k.
    """
    scores = bm25.get_scores(preprocess(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [corpus[i]["source"] for i in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 granularity-matched comparison")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        sys.exit(f"results.csv not found at {RESULTS_CSV}. Run run_evaluation.py first.")

    dense = pd.read_csv(RESULTS_CSV)
    for col in ("hit_at_1", "hit_at_5", "is_out_of_scope"):
        dense[col] = dense[col].astype(str).str.lower().isin(["true", "1", "yes"])

    print(f"Building chunk-level BM25 ({args.chunk_size}w / {args.overlap}w overlap) ...")
    corpus = build_chunk_corpus(args.chunk_size, args.overlap)
    bm25 = BM25Okapi([d["tokens"] for d in corpus])
    print(f"Indexed {len(corpus)} chunks over the same corpus the dense pipeline uses.\n")

    rows: list[dict] = []
    for _, q in dense.iterrows():
        if q["is_out_of_scope"]:
            continue
        expected = q["expected_document"]
        retrieved = chunk_bm25_retrieve(bm25, corpus, str(q["question"]), args.top_k)
        rows.append({
            "question_id": q["question_id"],
            "expected_document": expected,
            "bm25_chunk_top1": retrieved[0] if retrieved else "",
            "bm25_chunk_hit_at_1": hit_at_k(retrieved, expected, 1),
            "bm25_chunk_hit_at_5": hit_at_k(retrieved, expected, args.top_k),
            "dense_hit_at_1": bool(q["hit_at_1"]),
            "dense_hit_at_5": bool(q["hit_at_5"]),
        })

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_CSV, index=False)

    n = len(comp)
    def rate(col: str) -> float:
        return round(comp[col].mean(), 4)

    print("=" * 64)
    print(f"  Granularity-matched BM25 vs Dense  (in-scope n={n}, top_k={args.top_k})")
    print("=" * 64)
    print(f"  {'System':<34}{'Hit@1':>10}{'Hit@5':>10}")
    print(f"  {'BM25 (512-word chunks)':<34}{rate('bm25_chunk_hit_at_1'):>10.1%}{rate('bm25_chunk_hit_at_5'):>10.1%}")
    print(f"  {'BM25 (whole document) [baseline]':<34}{'73.3%':>10}{'93.3%':>10}")
    print(f"  {'Dense (FastEmbed chunks)':<34}{rate('dense_hit_at_1'):>10.1%}{rate('dense_hit_at_5'):>10.1%}")
    print("=" * 64)
    print(f"  Per-question detail written to {OUT_CSV.name}")


if __name__ == "__main__":
    main()

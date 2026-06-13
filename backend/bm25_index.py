"""
SiftOps — BM25 Index Builder
Builds and persists a BM25 index over the same corpus used by the dense pipeline.
Used by evaluation/run_evaluation.py for the System A baseline.

Usage:
    python backend/bm25_index.py           # builds index, saves to bm25_index.pkl
    python backend/bm25_index.py --rebuild # forces a rebuild
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import string
from pathlib import Path

import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

log = logging.getLogger("siftops.bm25")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Download NLTK data
import os as _os
_nltk_dir = _os.path.expanduser("~/nltk_data")
for _res in ("stopwords", "punkt", "punkt_tab"):
    nltk.download(_res, download_dir=_nltk_dir, quiet=True)
if _nltk_dir not in nltk.data.path:
    nltk.data.path.insert(0, _nltk_dir)

_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words("english"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "bm25_index.pkl"


# ── Text preprocessing ────────────────────────────────────────────────────────
def preprocess(text: str) -> list[str]:
    """Lowercase, remove punctuation, remove stopwords, stem."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    return [_stemmer.stem(t) for t in tokens if t not in _stopwords and len(t) > 1]


# ── Document loader ───────────────────────────────────────────────────────────
def load_corpus(data_dir: Path) -> list[dict]:
    """
    Returns a list of document dicts:
      { "source", "category", "text", "tokens" }
    One entry per PDF (whole doc, not chunked).
    For BM25 we keep docs whole to avoid chunk boundary issues,
    but you can switch to chunks by calling chunk_text from ingest.py.
    """
    corpus = []
    for cat_dir in sorted(data_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for pdf in cat_dir.glob("**/*.pdf"):
            try:
                doc = fitz.open(str(pdf))
                text = "\n".join(p.get_text("text") for p in doc)
                doc.close()
                text = re.sub(r"\s+", " ", text).strip()
            except Exception as exc:
                log.warning("Cannot read %s: %s", pdf, exc)
                continue
            corpus.append({
                "source": pdf.name,
                "source_path": str(pdf.relative_to(data_dir)),
                "category": category,
                "text": text,
                "tokens": preprocess(text),
            })
    return corpus


# ── Build & persist ───────────────────────────────────────────────────────────
def build_index(data_dir: Path, save_path: Path) -> dict:
    log.info("Loading corpus from %s ...", data_dir)
    corpus = load_corpus(data_dir)
    log.info("Building BM25 index over %d documents ...", len(corpus))

    tokenized = [d["tokens"] for d in corpus]
    bm25 = BM25Okapi(tokenized)

    payload = {
        "bm25": bm25,
        "corpus": corpus,   # metadata + raw text (no vectors needed)
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

    log.info("BM25 index saved to %s", save_path)
    return payload


def load_index(save_path: Path) -> dict:
    with open(save_path, "rb") as f:
        return pickle.load(f)


# ── Query API (used by evaluation pipeline) ───────────────────────────────────
class BM25Retriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}. "
                "Run `python backend/bm25_index.py --rebuild` first."
            )
        payload = load_index(index_path)
        self._bm25 = payload["bm25"]
        self._corpus = payload["corpus"]

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """
        Returns top_k results as list of dicts:
          { "source", "category", "text", "score", "rank" }
        """
        tokens = preprocess(text)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for rank, (idx, score) in enumerate(ranked, 1):
            doc = self._corpus[idx]
            results.append({
                "source":   doc["source"],
                "category": doc["category"],
                "text":     doc["text"][:500],
                "score":    round(float(score), 4),
                "rank":     rank,
            })
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build BM25 index")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--index-out", default=str(INDEX_PATH))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    out = Path(args.index_out)
    if out.exists() and not args.rebuild:
        log.info("Index already exists at %s (use --rebuild to force)", out)
        return

    build_index(Path(args.data_dir), out)


if __name__ == "__main__":
    main()
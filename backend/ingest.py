"""
SiftOps — Document Ingestion
Reads all PDFs from data/ subfolders, chunks them, embeds with FastEmbed,
and upserts into Qdrant.

Usage:
    python backend/ingest.py
    python backend/ingest.py --data-dir ./data --chunk-size 512 --overlap 64
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

load_dotenv()

log = logging.getLogger("siftops.ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST  = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT  = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION   = os.getenv("QDRANT_COLLECTION", "siftops_docs")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_DIM   = 384   # bge-small-en-v1.5 output size
BATCH_SIZE   = 64    # upsert batch size


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text_from_pdf(path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def extract_pages_from_pdf(path: Path) -> list[tuple[int, str]]:
    """Extract list of (page_number, text) from a PDF. Page numbers are 1-indexed."""
    doc = fitz.open(str(path))
    result = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    doc.close()
    return result


def clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, remove null bytes."""
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    chunk_size and overlap are measured in words.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── Document iterator ─────────────────────────────────────────────────────────
def iter_documents(data_dir: Path, chunk_size: int, overlap: int) -> Generator[dict, None, None]:
    """
    Walk data_dir, extract + chunk every PDF, yield payload dicts.
    The immediate subdirectory name becomes the 'category'.
    Each chunk stores the page number of the page it was primarily drawn from.
    """
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        pdf_files = list(category_dir.glob("**/*.pdf"))
        log.info("Category '%s': found %d PDFs", category, len(pdf_files))

        for pdf_path in pdf_files:
            try:
                page_texts = extract_pages_from_pdf(pdf_path)
            except Exception as exc:
                log.warning("Failed to read %s: %s", pdf_path, exc)
                continue

            # Build a flat list of (word, page_number) pairs to track page origins
            word_pages: list[tuple[str, int]] = []
            for page_num, raw_text in page_texts:
                cleaned = clean_text(raw_text)
                for word in cleaned.split():
                    word_pages.append((word, page_num))

            if not word_pages:
                continue

            words = [w for w, _ in word_pages]
            pages = [p for _, p in word_pages]
            total_words = len(words)

            # Build chunks with page tracking
            chunk_list: list[tuple[str, int]] = []  # (chunk_text, start_page)
            start = 0
            while start < total_words:
                end = min(start + chunk_size, total_words)
                chunk = " ".join(words[start:end])
                # Use the page of the first word in the chunk
                start_page = pages[start]
                chunk_list.append((chunk, start_page))
                if end == total_words:
                    break
                start += chunk_size - overlap

            for i, (chunk, page_num) in enumerate(chunk_list):
                doc_id = hashlib.md5(
                    f"{pdf_path.name}:{i}".encode()
                ).hexdigest()
                yield {
                    "id": str(uuid.UUID(doc_id)),
                    "text": chunk,
                    "source": pdf_path.name,
                    "source_path": str(pdf_path.relative_to(data_dir)),
                    "category": category,
                    "chunk_index": i,
                    "total_chunks": len(chunk_list),
                    "page": page_num,
                }


# ── Qdrant helpers ────────────────────────────────────────────────────────────
def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        log.info("Creating collection '%s' (dim=%d)", name, dim)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        log.info("Collection '%s' already exists — upserting into it.", name)


def upsert_batch(
    client: QdrantClient,
    collection: str,
    embedder: TextEmbedding,
    docs: list[dict],
) -> None:
    texts = [d["text"] for d in docs]
    vectors = list(embedder.embed(texts))
    points = [
        PointStruct(
            id=d["id"],
            vector=vec.tolist(),
            payload={k: v for k, v in d.items() if k != "id"},
        )
        for d, vec in zip(docs, vectors)
    ]
    client.upsert(collection_name=collection, points=points)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument("--data-dir", default="./data", help="Root data directory")
    parser.add_argument("--chunk-size", type=int, default=512, help="Words per chunk")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap words")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate collection before ingesting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), f"Data dir not found: {data_dir}"

    client   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    if args.recreate:
        log.info("Dropping collection '%s'...", COLLECTION)
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    ensure_collection(client, COLLECTION, VECTOR_DIM)

    batch: list[dict] = []
    total_chunks = 0

    for doc in tqdm(iter_documents(data_dir, args.chunk_size, args.overlap),
                    desc="Chunking & embedding"):
        batch.append(doc)
        if len(batch) >= BATCH_SIZE:
            upsert_batch(client, COLLECTION, embedder, batch)
            total_chunks += len(batch)
            batch.clear()

    if batch:
        upsert_batch(client, COLLECTION, embedder, batch)
        total_chunks += len(batch)

    log.info("Ingestion complete. Total chunks upserted: %d", total_chunks)
    info = client.get_collection(COLLECTION)
    log.info("Collection now has %d points.", info.points_count)


if __name__ == "__main__":
    main()
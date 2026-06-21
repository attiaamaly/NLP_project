"""
SiftOps — FastAPI Backend

Endpoints:
  GET  /health
  GET  /search?q=...&top_k=5
  POST /chat        { "question": "...", "top_k": 5 }
  POST /reindex
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams
from fastembed import TextEmbedding
from openai import OpenAI

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("siftops")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "siftops_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
VALID_CATEGORIES = ["HR", "Finance", "legal_compliance", "product_en_support", "security_it"]

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))

# ──────────────────────────────────────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────────────────────────────────────
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = TextEmbedding(model_name=EMBED_MODEL)
llm = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SiftOps", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceRef(BaseModel):
    id: str
    filename: str
    source: str
    page: Optional[int] = None
    score: float
    text: str
    category: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    refused: bool = False


class SearchResult(BaseModel):
    id: str
    filename: str
    source: str
    page: Optional[int] = None
    score: float
    snippet: str
    text: str
    category: str


class SearchResponse(BaseModel):
    status: str = "ok"
    results: list[SearchResult]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _payload(h: ScoredPoint) -> dict[str, Any]:
    return h.payload or {}


def _doc_name(payload: dict[str, Any]) -> str:
    return str(payload.get("source") or payload.get("filename") or payload.get("doc") or "unknown")


def _page_num(payload: dict[str, Any]) -> Optional[int]:
    page = payload.get("page")
    if page is None:
        return None
    try:
        return int(page)
    except Exception:
        return None


def embed(text: str) -> list[float]:
    return list(embedder.embed([text]))[0].tolist()


def qdrant_search(query: str, top_k: int) -> list[ScoredPoint]:
    vec = embed(query)
    return qdrant.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
    )


def build_context(hits: list[ScoredPoint]) -> str:
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        payload = _payload(h)
        src = _doc_name(payload)
        page = _page_num(payload)
        text = str(payload.get("text", "")).strip()
        page_str = f"Page {page}" if page is not None else "Page ?"
        parts.append(f"[{i}] Source: {src} | {page_str}\n{text}")
    return "\n\n".join(parts)


def unique_sources(hits: list[ScoredPoint]) -> list[SourceRef]:
    seen: set[tuple[str, Optional[int]]] = set()
    out: list[SourceRef] = []

    for h in hits:
        payload = _payload(h)
        src = _doc_name(payload)
        page = _page_num(payload)
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)

        text = str(payload.get("text", "")).strip()
        out.append(
            SourceRef(
                id=str(h.id),
                filename=src,
                source=src,
                page=page,
                score=round(float(h.score), 4),
                text=text[:300],
                category=str(payload.get("category", "unknown")),
            )
        )
    return out


def refusal_response() -> ChatResponse:
    return ChatResponse(
        answer="I don't have enough information to answer this question.",
        sources=[],
        refused=True,
    )


REFUSAL_PHRASES = [
    "i don't have enough information",
    "cannot answer",
    "not found in",
    "no relevant information",
]


def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in REFUSAL_PHRASES)


def count_pdfs() -> int:
    if not DATA_ROOT.exists():
        return 0
    return len(list(DATA_ROOT.rglob("*.pdf")))


def ensure_collection_exists() -> None:
    try:
        collections = qdrant.get_collections().collections
        if not any(c.name == COLLECTION for c in collections):
            qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            log.info("Created missing collection '%s'", COLLECTION)
    except Exception as exc:
        log.warning("Collection check failed: %s", exc)


def extractive_answer(hits: list[ScoredPoint]) -> str:
    if not hits:
        return "I don't have enough information to answer this question."

    best = hits[0]
    payload = _payload(best)
    src = _doc_name(payload)
    text = str(payload.get("text", "")).strip()
    snippet = text[:500] if text else ""
    answer = f"Based on {src}, {snippet}".strip()

    sources = unique_sources(hits)
    if sources:
        cite_str = " ".join(f"[Source: {s.filename}]" for s in sources[:3])
        answer = f"{answer}\n\n{cite_str}"

    return answer


SYSTEM_PROMPT = """You are SiftOps, an internal knowledge-base assistant.
Answer questions ONLY using the provided context snippets.
If the context does not contain enough information to answer the question,
respond with exactly: "I don't have enough information to answer this question."
Always cite the source document(s) at the end of your answer using [Source: filename].
Be concise and factual.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event() -> None:
    ensure_collection_exists()


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, Any]:
    try:
        info = qdrant.get_collection(COLLECTION)
        count = int(info.points_count or 0)
    except Exception:
        count = 0

    return {
        "status": "ok",
        "collection": COLLECTION,
        "points": count,
        "documents_count": count_pdfs(),
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "qdrant_connected": True,
        "timestamp": time.time(),
    }


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    limit: Optional[int] = Query(None, ge=1, le=50),
) -> SearchResponse:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    k = limit if limit is not None else top_k

    try:
        hits = qdrant_search(q, k)
    except Exception as exc:
        log.error("Qdrant search failed: %s", exc)
        raise HTTPException(status_code=503, detail="Search service unavailable.")

    results: list[SearchResult] = []
    for h in hits:
        payload = _payload(h)
        src = _doc_name(payload)
        page = _page_num(payload)
        text = str(payload.get("text", "")).strip()
        results.append(
            SearchResult(
                id=str(h.id),
                filename=src,
                source=src,
                page=page,
                score=round(float(h.score), 4),
                snippet=text[:300],
                text=text[:500],
                category=str(payload.get("category", "unknown")),
            )
        )

    return SearchResponse(status="ok", results=results)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        hits = qdrant_search(req.question, req.top_k)
    except Exception as exc:
        log.error("Qdrant search failed: %s", exc)
        raise HTTPException(status_code=503, detail="Search service unavailable.")

    if not hits:
        return refusal_response()

    top_score = float(hits[0].score)
    if top_score < CONFIDENCE_THRESHOLD:
        return refusal_response()

    sources = unique_sources(hits)
    context = build_context(hits)

    # If no OpenAI key is configured, use a deterministic extractive fallback.
    if llm is None:
        answer = extractive_answer(hits)
        return ChatResponse(
            answer=answer,
            sources=sources,
            refused=is_refusal(answer),
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {req.question}",
        },
    ]

    try:
        resp = llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            answer = extractive_answer(hits)
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        answer = extractive_answer(hits)

    return ChatResponse(
        answer=answer,
        sources=sources,
        refused=is_refusal(answer),
    )


@app.post("/reindex")
def reindex() -> dict[str, Any]:
    """Trigger re-ingestion. Run backend/ingest.py as a subprocess."""
    result = subprocess.run(
        [sys.executable, "backend/ingest.py", "--recreate"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr[:2000])

    return {
        "status": "reindex complete",
        "output": result.stdout[-2000:],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
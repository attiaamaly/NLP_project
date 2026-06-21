# SiftOps — Internal Knowledge Base

**Semantic search and grounded question answering over internal policy documents.**

NLP Group Project · Option 1: Application Development

---

## What SiftOps Does

SiftOps indexes a company's internal policy documents and answers employee questions with answers cited to the exact document and page. The proof of concept covers 36 PDFs across five departments. Two read paths run over one vector index: **semantic search** and **document-grounded chat**.

Questions are answered only from retrieved passages — the model cannot invent policy. If retrieved evidence is insufficient, the system refuses to answer.

---

## Folder Structure

```
NLP Project/
├── backend/
│   ├── main.py              # FastAPI application (search + chat endpoints)
│   ├── ingest.py            # PDF extraction, chunking, embedding, Qdrant upsert
│   └── bm25_index.py        # BM25 keyword baseline indexer
├── data/
│   ├── HR/                  # 8 policy PDFs
│   ├── finance/             # 7 policy PDFs
│   ├── legal_compliance/    # 6 policy PDFs
│   ├── security_it/         # 7 policy PDFs
│   └── product_en_support/  # 8 policy PDFs
├── evaluation/
│   ├── run_evaluation.py    # Full evaluation pipeline (dense + BM25 comparison)
│   ├── report_export.py     # Report-ready JSON + console tables
│   ├── results.csv          # Per-question results (generated)
│   ├── metrics.json         # Aggregated metrics (generated)
│   ├── failures.csv         # Failure analysis (generated)
│   ├── comparison.csv       # BM25 vs Dense comparison (generated)
│   └── figures/             # Auto-generated evaluation plots
├── evaluation_dataset.csv   # 35 benchmark questions (5 categories)
├── evaluation_analysis.ipynb
├── bm25_index.pkl           # Pre-built BM25 index
├── requirements.txt
└── README.md
```

---

## Installation Guide

### Prerequisites

- Python 3.10 or 3.11
- [Docker](https://docs.docker.com/get-docker/) (for Qdrant)
- An OpenAI API key (optional — the system falls back to extractive answers without one)

### Step 1 — Clone / unzip the project

```bash
unzip NLP_Project.zip
cd "NLP Project"
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Start Qdrant (vector database)

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`. To verify: open `http://localhost:6333/dashboard` in your browser.

### Step 5 — (Optional) Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4o-mini
CONFIDENCE_THRESHOLD=0.45
```

Without an API key, SiftOps uses an extractive fallback (returns the best matching passage directly). All retrieval metrics are unaffected.

### Step 6 — Ingest documents

```bash
python backend/ingest.py --recreate
```

This reads all 36 PDFs, splits them into overlapping 512-word chunks, embeds them with `BAAI/bge-small-en-v1.5` (384 dimensions), and stores them in Qdrant.

Expected output:
```
INFO | Category 'HR': found 8 PDFs
INFO | Category 'finance': found 7 PDFs
...
INFO | Ingestion complete. Total chunks upserted: 36
```

### Step 7 — Start the backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at `http://localhost:8000`.

---

## User Manual

### API Endpoints

#### `GET /health`
Check system status and index statistics.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "collection": "siftops_docs",
  "points": 36,
  "documents_count": 36,
  "embed_model": "BAAI/bge-small-en-v1.5",
  "chat_model": "gpt-4o-mini"
}
```

#### `GET /search?q=<query>&top_k=5`
Semantic search — returns ranked document chunks.

```bash
curl "http://localhost:8000/search?q=working+from+home&top_k=5"
```

Response includes `filename`, `page`, `score`, `snippet`, and `category` for each result.

#### `POST /chat`
Grounded question answering — returns an answer with cited sources.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the lunch reimbursement limit?", "top_k": 5}'
```

Response:
```json
{
  "answer": "The lunch reimbursement limit is €25. [Source: Finance_Expenses_Policy.pdf]",
  "sources": [...],
  "refused": false
}
```

If evidence is thin (top cosine score < 0.45), the system refuses:
```json
{
  "answer": "I don't have enough information to answer this question.",
  "sources": [],
  "refused": true
}
```

#### `POST /reindex`
Trigger a full re-ingestion of the `data/` folder.

```bash
curl -X POST http://localhost:8000/reindex
```

---

### Running the Evaluation

#### Standard dense evaluation (requires backend running):
```bash
python evaluation/run_evaluation.py
```

#### BM25 vs Dense comparison:
```bash
# Build BM25 index first (already pre-built as bm25_index.pkl)
python backend/bm25_index.py --rebuild

# Run comparison
python evaluation/run_evaluation.py --compare
```

#### Export results for report:
```bash
python evaluation/report_export.py
```

This prints formatted tables to the console and writes `evaluation/report_data.json`.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` (384-d) | Efficient, strong semantic understanding, runs locally via FastEmbed |
| Vector DB | Qdrant | Native cosine similarity, persistent local mode |
| Chunking | 512 words, 64-word overlap | Balances context window and granularity |
| Confidence threshold | 0.45 (`CONFIDENCE_THRESHOLD` in `backend/main.py`) | Fixed refusal cutoff: top cosine score below 0.45 → refuse. Not tuned against the evaluation set |
| LLM fallback | Extractive (no API key needed) | System functions without OpenAI access |
| Baseline | BM25 (Okapi) | Standard sparse IR baseline per proposal |

---

## Evaluation Results Summary

Produced by `python evaluation/run_evaluation.py --compare` against the dense
(FastEmbed + Qdrant) pipeline with a BM25 keyword baseline. The figures below
mirror `evaluation/metrics.json` exactly.

| Metric | Value |
|---|---|
| Questions evaluated | 35 (30 in-scope, 5 out-of-scope) |
| Search success rate | 100.0% |
| Chat success rate | 100.0% |
| Hit@1 | 86.7% |
| Hit@5 | 96.7% |
| Source match rate | 96.7% |
| Refusal accuracy | 40.0% |

Refusal accuracy is measured at the canonical `CONFIDENCE_THRESHOLD = 0.45`
(`backend/main.py`). At this threshold 2 of the 5 out-of-scope questions are
correctly refused; the other 3 retrieve a tangentially-related chunk whose top
cosine score still exceeds 0.45 and are answered instead. The threshold is a
fixed design parameter and is **not** tuned against the evaluation set.

**Per-category retrieval (in-scope, 30 questions):**

| Category | Count | Hit@1 | Hit@5 | Source match |
|---|---|---|---|---|
| Exact | 8 | 100% | 100% | 100% |
| Acronym | 5 | 100% | 100% | 100% |
| Reasoning | 5 | 100% | 100% | 100% |
| Ambiguous | 4 | 75% | 100% | 100% |
| Semantic | 8 | 62.5% | 87.5% | 87.5% |

**BM25 vs Dense retrieval:**

| System | Hit@1 | Hit@5 |
|---|---|---|
| BM25 (Okapi keyword baseline) | 73.3% | 93.3% |
| Dense (FastEmbed `bge-small-en-v1.5`) | 86.7% | 96.7% |

Per-question winners (in-scope): both correct 21, dense-only 5, BM25-only 1,
both wrong 3. Dense retrieval recovers paraphrase and synonym queries that the
keyword baseline misses.

---

## NLP Concepts Applied

| Concept | Where in SiftOps |
|---|---|
| Distributional semantics / dense embeddings | Chunks → 384-d vectors; cosine similarity is the ranking signal |
| Subword tokenization (BPE) | BGE model handles acronyms (MFA, GDPR, SLA) as subword units |
| Sparse vs dense retrieval | Dense in production; BM25 baseline in evaluation ablation |
| Retrieval-Augmented Generation | Passages retrieved first; LLM writes grounded, cited prose only |
| Polysemy / word-sense disambiguation | Contextual embeddings + department scoping resolve ambiguous terms |
| Regex data cleaning | Per-page PDF text cleaned before chunking |

---

## Use of AI Tools

LLMs were used throughout this project for: boilerplate code scaffolding, debugging, drafting docstrings and README prose, and exploring the literature. All substantive intellectual decisions — problem framing, evaluation design, label definitions, error taxonomy, conclusions — were made by the team. Every team member can explain and defend any design choice during Q&A.

---

## Requirements

See `requirements.txt`. Key packages:
- `fastapi`, `uvicorn` — API server
- `qdrant-client`, `fastembed` — vector search
- `PyMuPDF` — PDF extraction
- `rank-bm25`, `nltk` — keyword baseline
- `openai` — optional LLM generation
- `pandas`, `matplotlib`, `seaborn` — evaluation and plotting

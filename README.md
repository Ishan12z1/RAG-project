# RAG Project

A retrieval-augmented generation (RAG) assistant built with **FastAPI**, **FAISS**, **BM25**, and a **cross-encoder reranker**. The project combines dense retrieval, lexical retrieval, reranking, and a generation backend to produce grounded answers with citations, health reporting, runtime metrics, and evaluation utilities.

## What this project does

This repository implements an end-to-end RAG workflow:

1. **Chunk source documents** into retrieval-friendly passages.
2. **Embed chunks** with a sentence-transformer model.
3. **Build a FAISS index** for dense retrieval.
4. **Build a BM25 index** for lexical retrieval.
5. **Blend dense + BM25 results** into a hybrid candidate set.
6. **Rerank candidates** with a cross-encoder.
7. **Generate grounded answers** with a remote text-generation service.
8. **Serve the pipeline through FastAPI** with health, metrics, and chat endpoints.

## Core features

- **Hybrid retrieval** using FAISS + BM25.
- **Reranking** using either a local or API-backed cross-encoder.
- **Grounded responses with citations** returned by the `/chat` endpoint.
- **Structured health checks** for config, pipeline, dense index, BM25, chunk store, reranker, and generation provider.
- **Runtime metrics** including request counts, cache hit rates, and latency percentiles.
- **Evaluation scripts** for retrieval benchmarking and snapshot testing.
- **Config-driven setup** using YAML files in `configs/`.

## Architecture overview

At runtime, the API loads a singleton `RAGPipeline` that wires together the retrieval and generation stack:

```text
User query
   ↓
Conversation-aware query builder
   ↓
Dense retrieval (FAISS) + lexical retrieval (BM25)
   ↓
Hybrid score blending
   ↓
Cross-encoder reranking
   ↓
Prompt assembly with evidence block
   ↓
Generation provider (/generate)
   ↓
Structured parsing + citation extraction
   ↓
FastAPI response
```

### Main runtime components

- `app/main.py` — FastAPI app factory, middleware, and exception handling.
- `app/routes.py` — `/`, `/health`, `/metrics`, and `/chat` routes.
- `rag/chat.py` — the main `RAGPipeline` orchestration.
- `rag/retrieval/retrieve.py` — dense FAISS retriever with embedding cache.
- `rag/retrieval/hybrid_retriever.py` — hybrid retrieval and retrieval cache.
- `rag/rerank/cross_encoder_reranker_API.py` — local/API reranker wrapper.
- `rag/model/model_collab.py` — remote generation client.

## Repository layout

```text
app/                  FastAPI application code
artifacts/            Example retrieval artifacts and validation scripts
configs/              Runtime/configuration YAML files
docs/                 Design notes for architecture and metrics
evaluation/           Retrieval evaluation and golden-set tooling
rag/                  Core chunking, embedding, indexing, retrieval, reranking, prompting, and pipeline code
tests/                Smoke and snapshot tests
```

## Requirements

### Python

- Python **3.10+** recommended

### Python packages

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### External services

The runtime pipeline expects two HTTP services:

- a **generation endpoint** that accepts requests at `POST <api_url>/generate`
- a **reranker endpoint** that accepts requests at `POST <url>/rerank`

By default, those URLs are configured in:

- `configs/chat_config.yaml`
- `configs/reranker_config.yaml`

Replace the sample ngrok URLs with your own hosted services before running the full API.

## Configuration

The project is driven primarily by four YAML files:

### `configs/chat_config.yaml`
Controls:
- chunk parquet path
- embeddings directory
- FAISS index directory
- BM25 directory
- chunk store path
- reranker config path
- generation API URL
- CORS settings
- runtime version metadata
- generation timeout/retry settings

### `configs/embeddings.yaml`
Controls:
- embedding model name
- batch size
- normalization
- max input text length

### `configs/index.yaml`
Controls:
- index type (`flat_ip`, `flat_l2`, `hnsw_ip`, etc. depending on implementation)
- default retrieval settings

### `configs/reranker_config.yaml`
Controls:
- reranker model name
- `model_type` (`api` or `local`)
- batching
- truncation
- normalization
- retry and timeout settings

## Important local setup note

The checked-in `configs/chat_config.yaml` references paths under `data/run_2/...`, while this repository currently includes example artifacts under `artifacts/...`.

If you want to run the API locally with the files already present in the repo, update the config paths to match your local artifacts, for example:

```yaml
chunking_loc: "artifacts/processed_chunks.parquet"
embedding_loc: "artifacts/embeddings"
indexing_loc: "artifacts/index/flat_ip"
chunk_store_path: "artifacts/processed_chunks.parquet"
```

Also ensure `bm25_path` points to a valid BM25 directory containing `bm25.json`.

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Update runtime configuration

Edit the following files as needed:

- `configs/chat_config.yaml`
- `configs/reranker_config.yaml`

At minimum, verify:
- your chunk/index paths exist
- your generation service URL is reachable
- your reranker URL is reachable

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Default local server:

- `http://127.0.0.1:8000`

### 4. Check service health

```bash
curl http://127.0.0.1:8000/health
```

### 5. Send a chat request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is diabetes?",
    "top_k": 5,
    "debug": true,
    "history": []
  }'
```

## API reference

### `GET /health`
Returns service readiness plus component-level health checks.

Example response areas:
- config loaded
- pipeline initialized
- FAISS index ready
- chunk store loaded
- BM25 ready
- reranker configured
- generation provider configured
- version metadata loaded

### `GET /metrics`
Returns in-memory runtime counters such as:
- total requests
- total errors
- answer / abstain / parse-error counts
- schema-valid rate
- embedding cache hit rate
- retrieval cache hit rate
- p50 and p95 latency

### `POST /chat`
Accepts:

```json
{
  "query": "What is diabetes?",
  "session_id": null,
  "history": [],
  "debug": false,
  "top_k": 5
}
```

Returns:
- answer text
- citations
- abstain flag
- session id
- request id
- timing breakdown
- optional debug payload

#### Request fields

- `query` — user question
- `session_id` — optional UUID for chat continuity
- `history` — prior turns for lightweight conversation context
- `debug` — if `true`, includes retrieved chunks, cache hit info, and version metadata
- `top_k` — number of results to keep after retrieval/reranking

## Data preparation pipeline

The repository includes scripts for chunking, embedding, indexing, and evaluation.

### 1. Chunk documents

Example using the Canada-data chunking entry point:

```bash
python -m rag.chunking.canada_data \
  --corpus_root data/diabetes_canada_corpus \
  --manifest data/diabetes_canada_corpus/manifest.jsonl \
  --out_parquet data/run_2/processed_chunks.parquet \
  --stats_json data/run_2/chunk_stats.json
```

### 2. Build embeddings

```bash
python -m rag.embedding.embed \
  --chunks_path data/run_2/processed_chunks.parquet \
  --configure_path configs/embeddings.yaml \
  --output_dir data/run_2/embeddings
```

### 3. Build the FAISS index

```bash
python -m rag.indexing.index_faiss \
  --index_cfg configs/index.yaml \
  --embeddings_root data/run_2/embeddings \
  --out_dir data/run_2/index/flat_ip
```

### 4. Build the BM25 index

The BM25 builder currently defaults to `data/run_2/processed_chunks.parquet`, so either place your parquet there or update the script before running:

```bash
python -m rag.retrieval.bm25_builder
```

## Testing and checks

### Pytest

```bash
pytest
```

### Focused retrieval snapshot test

```bash
pytest tests/test_retrieval_snapshot.py
```

### Focused smoke test

```bash
pytest tests/test_retrieve_and_prompt_smoke.py
```

### Artifact sanity scripts

Useful utility scripts are available in `artifacts/checks/`, for example:

```bash
python artifacts/checks/check_faiss.py --index_path artifacts/index/flat_ip/faiss.index
python artifacts/checks/embeddings_check.py
python artifacts/checks/smoke_check_faiss_index.py
```

## Evaluation utilities

The `evaluation/` directory contains scripts and outputs for measuring retrieval quality, generating examples, and assembling labeled datasets.

Useful entry points include:

- `evaluation/scripts/evaluate_retrievers.py`
- `evaluation/scripts/run_retrieval_eval.py`
- `evaluation/scripts/run_retrieval_examples.py`
- `evaluation/scripts/generate_retrieval_snapshot.py`
- `evaluation/golden_set/compile_golden_set.py`

The repo also includes notes in:

- `docs/architecture.md`
- `docs/metrics.md`

## Operational notes

### Caching

The runtime uses:
- an **embedding cache** in the dense retriever
- a **retrieval cache** in the hybrid retriever

These improve repeated-query latency and are surfaced through `/metrics` and debug responses.

### Health behavior

`/health` reports `ok` only when all major runtime components are ready. If the generation or reranker service is unreachable or indexes are missing, the service will report a degraded state.

### Error handling

The API returns structured error responses for:
- invalid requests
- empty queries
- retrieval failures
- generation timeouts
- generation provider failures
- unexpected internal errors

### Logging

Requests and failures are logged as JSON for easier machine parsing.

## Known gotchas

- The repository includes **example artifacts**, but the runtime config may still point to a different local directory structure.
- The generation and reranking backends are **not fully self-contained in this repo**; you need reachable HTTP endpoints.
- Some tests and utility scripts assume default paths like `data/processed_chunks.parquet` or `artifacts/index`, so align your paths before running them.
- If you switch reranking to `model_type: local`, ensure the required transformer weights can be downloaded or are already present in your environment.

## Suggested development workflow

1. Prepare chunked data.
2. Generate embeddings.
3. Build FAISS and BM25 indexes.
4. Update `configs/chat_config.yaml`.
5. Verify `/health`.
6. Run smoke tests.
7. Run evaluation scripts.
8. Iterate on retrieval, reranking, prompts, and parsing.

## Future improvements

A few natural next steps for this project:

- add a reproducible `.env` or config override mechanism
- add Docker and compose support
- add a proper UI module for the root demo route
- document the expected schemas for `/generate` and `/rerank`
- add CI to validate retrieval assets and tests automatically
- version data artifacts more explicitly

## License

No license file is currently included in this repository. Add one before distributing or open-sourcing the project.

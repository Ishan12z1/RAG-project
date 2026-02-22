# tests/test_retrieve_and_prompt_smoke.py
#
# Smoke + determinism tests for:
#   Slice 3.1: rag/retrieve.py (Retriever.retrieve)
#   Slice 3.2: rag/prompt.py (build_evidence_block/build_prompt)
#
# Run:
#   pytest -q
#
# Configure paths via env vars (recommended) or edit defaults below:
#   RAG_INDEX_DIR=/path/to/index_dir
#   RAG_EMB_DIR=/path/to/embeddings_dir
#   RAG_CHUNKS_PATH=data/processed_chunks.parquet

import os
import re

import pytest

from rag.retrieval.retrieve import Retriever
from rag.prompt import build_evidence_block, build_prompt


INDEX_DIR = os.getenv("RAG_INDEX_DIR", "artifacts/index")          # <-- change if needed
EMB_DIR = os.getenv("RAG_EMB_DIR", "artifacts/embeddings")        # <-- change if needed
CHUNKS_PATH = os.getenv("RAG_CHUNKS_PATH", "data/processed_chunks.parquet")

# Use a stable query that should exist in your diabetes corpus.
SMOKE_QUERY = os.getenv("RAG_SMOKE_QUERY", "What is diabetes?")


@pytest.mark.smoke
def test_slice31_retrieve_returns_results():
    r = Retriever(index_dir=INDEX_DIR, embeddings_dir=EMB_DIR, chunks_path=CHUNKS_PATH)
    out = r.retrieve(SMOKE_QUERY, top_k=5)

    assert isinstance(out, list)
    assert len(out) > 0, "Retriever returned no results. Check paths, index, embeddings, and chunks parquet."

    # Basic schema checks
    first = out[0]
    assert isinstance(first.chunk_id, str) and first.chunk_id
    assert isinstance(first.text, str)
    assert isinstance(first.score, float)

    # Citation fields must exist (Slice 3.1 requirement)
    c = first.citation
    for attr in ("source", "title", "section", "chunk_id", "doc_id"):
        assert hasattr(c, attr), f"Missing citation field: {attr}"
        val = getattr(c, attr)
        assert isinstance(val, str) and val, f"Citation field {attr} is empty"


@pytest.mark.smoke
def test_slice31_filtering_doc_id_works():
    r = Retriever(index_dir=INDEX_DIR, embeddings_dir=EMB_DIR, chunks_path=CHUNKS_PATH)
    out = r.retrieve(SMOKE_QUERY, top_k=5)
    assert out, "Need at least one result to test filters."

    target_doc_id = out[0].citation.doc_id
    filtered = r.retrieve(SMOKE_QUERY, top_k=5, filters={"doc_id": target_doc_id}, oversample=10)

    assert filtered, "Filter returned no results; oversample may be too small or doc_id not in md_view."
    assert all(x.citation.doc_id == target_doc_id for x in filtered), "Filter by doc_id did not constrain results."


def test_slice31_determinism_same_topk_chunk_ids():
    """
    Determinism check: same query twice should return same top-k chunk_ids.
    This is weaker than the snapshot test (Slice 3.4) but catches obvious nondeterminism.
    """
    r = Retriever(index_dir=INDEX_DIR, embeddings_dir=EMB_DIR, chunks_path=CHUNKS_PATH)
    a = r.retrieve(SMOKE_QUERY, top_k=5)
    b = r.retrieve(SMOKE_QUERY, top_k=5)

    assert [x.chunk_id for x in a] == [x.chunk_id for x in b], "Top-k chunk_ids changed between runs."


@pytest.mark.smoke
def test_slice32_evidence_block_format_and_tags():
    r = Retriever(index_dir=INDEX_DIR, embeddings_dir=EMB_DIR, chunks_path=CHUNKS_PATH)
    chunks = r.retrieve(SMOKE_QUERY, top_k=5)

    evidence_block, items = build_evidence_block(chunks, max_chunks=5, max_chars_per_chunk=300)

    assert "EVIDENCE" in evidence_block
    assert len(items) == len(chunks[:5])

    # Tags must be [C1]..[Ck] in order
    expected_tags = [f"[C{i}]" for i in range(1, len(items) + 1)]
    got_tags = [it.citation_tag for it in items]
    assert got_tags == expected_tags

    # Evidence headers must include required fields
    for tag in expected_tags:
        assert tag in evidence_block

    # Quick regex to ensure header contains key/value fields
    header_re = re.compile(r"\[C\d+\]\s+doc_id=.*\|\s+title=.*\|\s+section=.*\|\s+chunk_id=.*\|\s+source=.*")
    assert header_re.search(evidence_block), "Evidence header format does not match expected contract."


def test_slice32_build_prompt_contains_question_and_evidence():
    r = Retriever(index_dir=INDEX_DIR, embeddings_dir=EMB_DIR, chunks_path=CHUNKS_PATH)
    chunks = r.retrieve(SMOKE_QUERY, top_k=3)

    prompt = build_prompt("Explain diabetes briefly.", chunks)

    assert "system" in prompt and "user" in prompt
    assert "Citation rules" in prompt["system"]
    assert "Question:" in prompt["user"]
    assert "EVIDENCE" in prompt["user"]
    assert "[C1]" in prompt["user"]
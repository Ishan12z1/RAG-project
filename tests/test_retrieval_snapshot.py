# tests/test_retrieval_snapshot.py
#
# Slice 3.4 (part 2): Enforce retrieval snapshot determinism.
#
# Run:
#   pytest -q
#
# The test fails if top-k chunk_ids change for any snapshot query.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from rag.retrieval.retrieve import Retriever


SNAPSHOT_PATH = Path(os.getenv("RAG_SNAPSHOT_PATH", "tests/retrieval_snapshot.json"))

INDEX_DIR = os.getenv("RAG_INDEX_DIR", "artifacts/index")
EMB_DIR = os.getenv("RAG_EMB_DIR", "artifacts/embeddings")
CHUNKS_PATH = os.getenv("RAG_CHUNKS_PATH", "data/processed_chunks.parquet")


def _load_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {path}. "
            f"Generate it with: python eval/generate_retrieval_snapshot.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.snapshot
def test_retrieval_matches_snapshot():
    snap = _load_snapshot(SNAPSHOT_PATH)

    top_k = int(snap.get("top_k", 5))
    oversample = int(snap.get("oversample", 10))
    items: List[Dict[str, Any]] = snap["items"]

    r = Retriever(index_dir=INDEX_DIR, embeddings_dir=EMB_DIR, chunks_path=CHUNKS_PATH)

    mismatches = []
    for it in items:
        q = it["query"]
        expected_ids = it["expected_chunk_ids"]

        got = r.retrieve(q, top_k=top_k, oversample=oversample)
        got_ids = [h.chunk_id for h in got]

        if got_ids != expected_ids:
            mismatches.append(
                {
                    "query": q,
                    "expected": expected_ids,
                    "got": got_ids,
                }
            )

    assert not mismatches, "Snapshot mismatches:\n" + json.dumps(mismatches, indent=2)
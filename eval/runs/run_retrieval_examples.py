# eval/run_retrieval_examples.py
#
# Slice 3.3: Generate eval/results/retrieval_examples.json
#
# What it does:
# - Runs your Retriever on a small set of queries
# - Saves top-k results with score + citation fields + short text preview
#
# Run:
#   python -m eval.run_retrieval_examples
# or:
#   python eval/run_retrieval_examples.py
#
# Optional env vars:
#   RAG_INDEX_DIR, RAG_EMB_DIR, RAG_CHUNKS_PATH
#   RAG_TOP_K, RAG_OVERSAMPLE
#   RAG_EXAMPLES_OUT (default: eval/results/retrieval_examples.json)

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from rag.retrieval.retrieve import Retriever


def _truncate(s: str, n: int = 240) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def main() -> None:
    index_dir = os.getenv("RAG_INDEX_DIR", "artifacts/index")
    emb_dir = os.getenv("RAG_EMB_DIR", "artifacts/embeddings")
    chunks_path = os.getenv("RAG_CHUNKS_PATH", "data/processed_chunks.parquet")

    top_k = int(os.getenv("RAG_TOP_K", "5"))
    oversample = int(os.getenv("RAG_OVERSAMPLE", "2"))

    out_path = Path(os.getenv("RAG_EXAMPLES_OUT", "eval/results/retrieval_examples.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep these queries stable; add domain-specific queries as you like.
    queries = [
        "What is diabetes?",
        "What is the difference between type 1 and type 2 diabetes?",
        "What is blood glucose (blood sugar)?",
        "What is A1C and why is it important?",
        "What is hypoglycemia?",
        "What are common symptoms of diabetes?",
        "How does insulin work in the body?",
        "What are complications of uncontrolled diabetes?",
        "What is prediabetes?",
        "What is diabetic ketoacidosis (DKA)?",
    ]

    r = Retriever(index_dir=index_dir, embeddings_dir=emb_dir, chunks_path=chunks_path)

    examples: List[Dict[str, Any]] = []
    for q in queries:
        hits = r.retrieve(q, top_k=top_k, oversample=oversample)
        examples.append(
            {
                "query": q,
                "top_k": top_k,
                "oversample": oversample,
                "results": [
                    {
                        "rank": i + 1,
                        "score": float(h.score),
                        "chunk_id": h.chunk_id,
                        "doc_id": h.citation.doc_id,
                        "title": h.citation.title,
                        "section": h.citation.section,
                        "source": h.citation.source,
                        # store both a preview and (optionally) full text
                        "text_preview": _truncate(h.text, 280),
                    }
                    for i, h in enumerate(hits)
                ],
            }
        )

    payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "index_dir": str(index_dir),
        "embeddings_dir": str(emb_dir),
        "chunks_path": str(chunks_path),
        "notes": "Slice 3.3 retrieval traces. Scores are standardized to higher-is-better in Retriever.",
        "examples": examples,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} with {len(queries)} queries.")


if __name__ == "__main__":
    main()
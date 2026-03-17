# eval/generate_retrieval_snapshot.py


from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from rag.retrieval.retrieve import Retriever


def main() -> None:
    index_dir = os.getenv("RAG_INDEX_DIR", "artifacts/index")
    emb_dir = os.getenv("RAG_EMB_DIR", "artifacts/embeddings")
    chunks_path = os.getenv("RAG_CHUNKS_PATH", "data/processed_chunks.parquet")

    top_k = int(os.getenv("RAG_SNAPSHOT_TOP_K", "5"))
    oversample = int(os.getenv("RAG_SNAPSHOT_OVERSAMPLE", "10"))

    out_path = Path(os.getenv("RAG_SNAPSHOT_OUT", "tests/retrieval_snapshot.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep these stable. These should be the same ones you care about long-term.
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

    snapshot_items: List[Dict[str, Any]] = []
    for q in queries:
        hits = r.retrieve(q, top_k=top_k, oversample=oversample)
        snapshot_items.append(
            {
                "query": q,
                "top_k": top_k,
                "expected_chunk_ids": [h.chunk_id for h in hits],
                # optional: extra guardrail (helps detect bad remaps)
                "expected_doc_ids": [h.citation.doc_id for h in hits],
            }
        )

    payload: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "index_dir": str(index_dir),
        "embeddings_dir": str(emb_dir),
        "chunks_path": str(chunks_path),
        "top_k": top_k,
        "oversample": oversample,
        "items": snapshot_items,
        "notes": "Golden snapshot for retrieval determinism. Regenerate only when you intentionally change chunking/embeddings/index params.",
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} with {len(queries)} queries.")


if __name__ == "__main__":
    main()
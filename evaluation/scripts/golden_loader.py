from __future__ import annotations

import json
from typing import Any, Dict, List, Set
from rag.utils.contracts import RetrievalExample

def load_golden_set(path: str) -> List[RetrievalExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples: List[RetrievalExample] = []
    for item in data:
        qid = str(item["qid"])
        query = str(item["query"])

        gold: Set[str] = set()
        for x in item.get("chunk_ids", []):
            cid = x.get("chunk_id")
            if cid:
                gold.add(str(cid))

        best = (item.get("best_chunk") or {}).get("chunk_id")
        if best:
            gold.add(str(best))

        if not gold:
            continue

        examples.append(
            RetrievalExample(
                qid=qid,
                query=query,
                gold_chunk_ids=gold,
                meta={"bucket": item.get("bucket"), "best_chunk_id": str(best) if best else None},
            )
        )

    return examples
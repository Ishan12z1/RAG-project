from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
from rag.utils.contracts import RetrievedChunk

class Reranker(Protocol):
    """
    Reranker protocol. Any implementation must be swappable behind this interface.

    Contract:
      - Input is a query + list of candidate chunks (text included).
      - Output is a list of (chunk_id, score) for the same candidates.
      - Higher score = more relevant.
      - Must not add/drop chunk_ids.
    """
    def rerank(self, query:str, candidates:Sequence[RetrievedChunk])->list[tuple[RetrievedChunk,float]]:
        ...
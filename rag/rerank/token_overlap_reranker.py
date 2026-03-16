# rag/rerank/token_overlap_reranker.py
from __future__ import annotations

import re
from typing import List, Sequence, Set

from rag.rerank.base import RetrievedChunk, RetrievedChunk, Reranker

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tok(s: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s)}


class TokenOverlapReranker(Reranker):
    """
    Lightweight reranker (no ML dependency):
      score = |tokens(query) ∩ tokens(chunk)| / |tokens(query)|

    Not meant to be final, but:
      - validates the rerank stage plumbing
      - often improves ordering on entity/keyword-heavy queries
    """

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        q = _tok(query)
        denom = max(1, len(q))
        outs: List[RetrievedChunk] = []
        for c in candidates:
            ct = _tok(c.text)
            score = len(q & ct) / float(denom)
            outs.append(RetrievedChunk(chunk_id=c.chunk_id, score=score))
        return outs
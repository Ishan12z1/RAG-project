from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class HybridConfig:
    enable:bool=False
    alpha:float=0.6
    dense_candidates:int=25
    bm25_candidates:int=25

@dataclass(frozen=True)
class RerankConfig:
    enable:bool=False
    candidate_k:int=50
    max_text_chars:int=2000
    batch_size:int=16

@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    hybrid: HybridConfig = HybridConfig()
    rerank: RerankConfig = RerankConfig()

@dataclass(frozen=True)
class RunConfig:
    """
    run_tag is what gets written into ladder.csv, e.g.
    baseline, +hybrid, +hybrid+rerank
    """
    run_tag: str = "baseline"
    retrieval: RetrievalConfig = RetrievalConfig()
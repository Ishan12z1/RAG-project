from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Dict,List,Optional

@dataclass(frozen=True)
class ChunkRecord:
    chunk_id:str
    text:str
    metadata: Dict[str,Any]

@dataclass(frozen=True)
class EmbeddingMeta:
    model_name:str
    dim:int
    normalized:bool
    emb_hash:str

@dataclass(frozen=True)
class IndexMeta:
    index_type:str
    params:Dict[str,Any]
    index_hash:str
    emb_hash:str

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
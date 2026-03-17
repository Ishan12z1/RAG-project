from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Dict,List,Optional, Set
from pathlib import Path
##Chunking 
@dataclass(frozen=True)
class ChunkPolicy:
    policy_version: str = "v1"
    target_tokens: int = 550
    overlap_tokens: int = 80
    min_tokens: int = 100
    max_tokens: int = 750

@dataclass(frozen=True)
class RawDoc:
    path: Path
    source: str          # relative path string
    doc_id: str          # stable hash of file content


@dataclass(frozen=True)
class ExtractedDoc:
    raw: RawDoc
    title: str
    normalized_text: str
    url: Optional[str] = None
    page_texts: Optional[List[str]] = None  # only set for PDFs

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    section_path: str
    kind: str  # "heading", "para", "bullets", "table", "other"

@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    section_path: str
    chunk_index: int
    start_offset: int
    end_offset: int
    token_count: int
    checksum: str
    chunk_text: str
    url: Optional[str] 
    page_start: Optional[int] = None
    page_end: Optional[int] = None

## Retrive 
@dataclass(frozen=True)
class Citation:
    source:str
    url:str
    title: str
    section: str
    chunk_id : str
    doc_id:str

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id:str
    score:float
    text:str
    citation :Citation
    metadata:Dict[str,Any]
    rank:Optional[int]=None

## Prompt
@dataclass(frozen=True)
class EvidenceItem:
    citation_tag:str
    citation:Any
    text:str
    score:float
    metadata:Dict[str,Any]

## checking 
@dataclass(frozen=True)
class AbstainDecision:
    abstain:bool
    reasons:List[str]
    signals:Dict[str,Any]

@dataclass(frozen=True)
class AbstainConfig:
    # Score-based
    min_top1: float = 0.35
    min_gap: float = 0.05
    gap_k: int = 5

    # Lexical overlap
    min_overlap: float = 0.18
    max_context_chars: int = 20_000 


@dataclass(frozen=True)
class ParsedAnswer:
    '''
mode: str # "answer" or "abstain": tells whether the model produced a normal answer or an abstention. \n
bullets: List[str]: the answer content split into bullets (empty if abstain).\n
citations_by_bullet: List[List[str]]: for each bullet, the list of citation tags used (e.g., ["C1","C2"]).\n
resolved_chunk_ids_by_bullet: List[List[str]]: same structure as above, but citation tags mapped to real chunk_ids (used for eval/logging).\n
abstain_reason: Optional[str]: if abstained, the text after ABSTAIN:; otherwise None.\n
needs: List[str]: if abstained, the clarifying items after NEED:; otherwise empty.\n
raw_text: str: the exact raw model output (always stored for debugging).\n
parse_warnings: List[str]: parser flags like missing citations, invalid tags, wrong bullet count, etc.\n
    '''
    mode:str
    bullets:List[str]
    citation_by_bullet:List[List[str]]
    resolved_chunk_ids_by_bullet :List[List[str]]
    abstain_reason:Optional[str]
    needs:List[str]
    raw_text:str
    parse_warning:List[str]


### Eval Contracts 
@dataclass(frozen=True)
class RetrievalExample:
    qid:str
    query:str
    gold_chunk_ids:Set[str]
    golden_doc_ids:Optional[Set[str]]=None
    meta:Optional[Dict[str,Any]]=None

@dataclass
class RetrievalMetrics:
    recall_at_k: Dict[int, float]
    mrr_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    p50_ms: float
    p95_ms: float
    mean_ms: float
    n_queries: int

@dataclass
class PerQueryResult:
    qid: str
    query: str
    gold_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    first_hit_rank: Optional[int]
    latency_ms: float
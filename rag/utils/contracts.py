from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional, Set, Literal
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
class AnswerSegment:
    text: str
    citations: List[str] = field(default_factory=list)
    resolved_chunk_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedAnswer:
    """
    Internal structured representation of the model output.

    mode:
        "answer" -> validated grounded answer segments
        "abstain" -> validated abstention request
        "parse_error" -> invalid or unusable model output
    """

    mode: Literal["answer", "abstain", "parse_error"]
    segments: List[AnswerSegment] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)
    raw_text: str = ""
    parse_warnings: List[str] = field(default_factory=list)
    schema_valid: bool = False

    def __post_init__(self) -> None:
        if self.mode == "answer":
            if not self.segments:
                self.parse_warnings.append("answer_mode_without_segments")
            if self.needs:
                self.parse_warnings.append("answer_mode_with_needs")
            for idx, segment in enumerate(self.segments, start=1):
                if not segment.text.strip():
                    self.parse_warnings.append(f"segment_{idx}_text_missing")
                if not segment.citations:
                    self.parse_warnings.append(f"segment_{idx}_citations_missing")
        elif self.mode == "abstain":
            if self.segments:
                self.parse_warnings.append("abstain_mode_with_segments")
        elif self.mode == "parse_error":
            if self.segments and self.needs:
                self.parse_warnings.append("parse_error_with_mixed_answer_and_abstain_fields")


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

@dataclass(frozen=True)
class CacheHitInfo:
    embedding: Optional[bool] = None
    retrieval: Optional[bool] = None

@dataclass 
class PipelineTimings:
    embed: Optional[float] = None
    retrieve: Optional[float] = None
    rerank: Optional[float] = None
    generate: Optional[float] = None
    total: Optional[float] = None

@dataclass 
class PipelineResult:
    parsed_output:ParsedAnswer
    retrieved_chunks:list[RetrievedChunk]
    timings_ms: PipelineTimings = field(default_factory=PipelineTimings)
    cache_hits: CacheHitInfo = field(default_factory=CacheHitInfo)
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


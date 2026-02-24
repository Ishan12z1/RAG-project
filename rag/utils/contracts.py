from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Dict,List,Optional
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

## Prompt
@dataclass(frozen=True)
class EvidenceItem:
    citation_tag:str
    citation:Any
    text:str
    score:float
    metadata:Dict[str,Any]
from pydantic import BaseModel,Field
from typing import Optional, Any
from uuid import UUID

class ChatbotRequest(BaseModel):
    query:str=Field(...,min_length=1,description="User query")
    session_id:Optional[UUID]=Field(default=None,description="Optional chat session id")
    debug:bool=Field(default=False,description="Return extra debug info")   
    top_k:int=Field(default=5,ge=1,le=20)

class Citation(BaseModel):
    source:Optional[str]=None
    title:Optional[str]=None
    section:Optional[str]=None
    chunk_id:Optional[str]=None


class TimingInfo(BaseModel):
    embed: Optional[float] = None
    retrieve: Optional[float] = None
    rerank: Optional[float] = None
    generate: Optional[float] = None
    total: Optional[float] = None

class CacheInfo(BaseModel):
    embedding: Optional[bool] = None
    retrieval: Optional[bool] = None


class DebugInfo(BaseModel):
    cache_hits: Optional[CacheInfo] = None
    retrieved_chunks: Optional[list[dict[str, Any]]] = None
    versions: Optional[dict[str, str]] = None

class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    request_id: Optional[str] = None
    timings_ms: Optional[TimingInfo] = None
    debug: Optional[DebugInfo] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class MetricsResponse(BaseModel):
    requests_total: int
    cache_hit_rate: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None

class ErrorDetail(BaseModel):
    code:str
    message:str

class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: Optional[str] = None
from typing import Any, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Speaker role for a prior turn")
    content: str = Field(..., min_length=1, max_length=2000, description="Turn content")


class ChatbotRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    session_id: Optional[UUID] = Field(default=None, description="Optional chat session id")
    history: list[ChatHistoryTurn] = Field(
        default_factory=list,
        max_length=12,
        description="Optional prior turns for lightweight multi-turn continuity",
    )
    debug: bool = Field(default=False, description="Return extra debug info")
    top_k: int = Field(default=5, ge=1, le=20)

class Citation(BaseModel):
    source: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    url: Optional[str] = None


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
    conversation_context: Optional[str] = None
    effective_query: Optional[str] = None
    versions: Optional[dict[str, str]] = None

class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    session_id: Optional[UUID] = None
    request_id: Optional[str] = None
    timings_ms: Optional[TimingInfo] = None
    debug: Optional[DebugInfo] = None


class HealthCheck(BaseModel):
    ready: bool
    detail: Optional[str] = None


class HealthDetails(BaseModel):
    config: HealthCheck
    pipeline: HealthCheck
    dense_index: HealthCheck
    chunk_store: HealthCheck
    bm25: HealthCheck
    reranker: HealthCheck
    generation_provider: HealthCheck
    versions: HealthCheck


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    details: HealthDetails


class MetricsResponse(BaseModel):
    requests_total: int
    errors_total: int = 0
    answer_total: int = 0
    abstain_total: int = 0
    parse_error_total: int = 0
    schema_valid_rate: Optional[float] = None
    embedding_cache_hit_rate: Optional[float] = None
    retrieval_cache_hit_rate: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None

class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: Optional[str] = None

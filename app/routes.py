from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
import re
from app.schemas import (
    TimingInfo,
    ChatbotRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    Citation as SchemaCitation,
    CacheInfo,
    DebugInfo,
    ErrorResponse,
) 
from app.deps import get_pipeline
from rag.chat import RAGPipeline
from app.utils import get_used_citations,format_answer, log_json
from error_handler.errors import EmptyQueryError
from app.metrics import runtime_metrics
from app.health import build_health_response
from app.ui import DEMO_HTML


router=APIRouter()

def _serialize_history(payload: ChatbotRequest) -> list[dict[str, str]]:
    return [turn.model_dump() for turn in payload.history]


def _serialize_retrieved_chunks(result) -> list[dict]:
    serialized = []
    for chunk in result.retrieved_chunks:
        serialized.append(
            {
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "rank": chunk.rank,
                "text": chunk.text,
                "citation": {
                    "source": chunk.citation.source,
                    "title": chunk.citation.title,
                    "section": chunk.citation.section,
                    "chunk_id": chunk.citation.chunk_id,
                    "doc_id": chunk.citation.doc_id,
                    "url": chunk.citation.url,
                },
            }
        )
    return serialized


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo():
    return HTMLResponse(DEMO_HTML)

@router.get("/health",response_model=HealthResponse)
def health():
    return build_health_response()

@router.get("/metrics", response_model=MetricsResponse)
def metrics():
    snap = runtime_metrics.snapshot()
    return MetricsResponse(**snap)

@router.post("/chat", response_model=ChatResponse,
                 responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },)
def chat(
    payload: ChatbotRequest,
    request: Request,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    query = payload.query.strip()
    if not query:
        raise EmptyQueryError()
    session_id: UUID = payload.session_id or uuid4()
    history = _serialize_history(payload)

    result = pipeline.run(query=query, top_k=payload.top_k, history=history)

    answer_text = format_answer(result.parsed_output)
    citations = get_used_citations(result)
    abstained = result.parsed_output.mode == "abstain"

    schema_citations = [
        SchemaCitation(
            source=c.source,
            title=c.title,
            section=c.section,
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            url=c.url,
        )
        for c in citations
    ]
    debug_info = None
    if payload.debug:
        debug_info = DebugInfo(
            cache_hits=CacheInfo(
                embedding=result.cache_hits.embedding,
                retrieval=result.cache_hits.retrieval,
            ),
            retrieved_chunks=_serialize_retrieved_chunks(result),
            conversation_context=result.context.get("conversation_context"),
            effective_query=result.context.get("effective_query"),
            versions=result.versions
        )
    response = ChatResponse(
        answer=answer_text,
        citations=schema_citations,
        abstained=abstained,
        session_id=session_id,
        request_id=request.state.request_id,
        timings_ms=TimingInfo(
            embed=result.timings_ms.embed,
            retrieve=result.timings_ms.retrieve,
            rerank=result.timings_ms.rerank,
            generate=result.timings_ms.generate,
            total=result.timings_ms.total,
        ),
        debug=debug_info,
    )
    log_json(
        {
            "event": "chat_request",
            "request_id": request.state.request_id,
            "path": "/chat",
            "session_id": str(session_id),
            "query_length": len(payload.query),
            "top_k": payload.top_k,
            "history_turns": len(history),
            "effective_query": result.context.get("effective_query"),
            "abstained": abstained,
            "schema_valid": result.parsed_output.schema_valid,
            "outcome_bucket": result.parsed_output.mode,
            "abstain_precheck": result.context.get("abstain_precheck"),
            "parsed_mode": result.parsed_output.mode,
            "parse_warnings": result.parsed_output.parse_warnings,
            "citation_count": len(schema_citations),
            "retrieved_chunk_count": len(result.retrieved_chunks),
            "cache_hits": {
                "embedding": result.cache_hits.embedding,
                "retrieval": result.cache_hits.retrieval,
            },
            "cache_stats": result.cache_stats,
            "versions": result.versions,
            "conversation_context": result.context.get("conversation_context"),
            "timings_ms": {
                "embed": result.timings_ms.embed,
                "retrieve": result.timings_ms.retrieve,
                "rerank": result.timings_ms.rerank,
                "generate": result.timings_ms.generate,
                "total": result.timings_ms.total,
            },
            "transport": {
                "generation": getattr(pipeline.model, "last_call_meta", None),
                "reranker": getattr(pipeline.reranker, "last_call_meta", None),
            },
            "status": "ok",
        }
    )

    runtime_metrics.record_request(
        total_ms=result.timings_ms.total,
        mode=result.parsed_output.mode,
        schema_valid=result.parsed_output.schema_valid,
        embedding_cache_hit=getattr(result.cache_hits, "embedding", None),
        retrieval_cache_hit=getattr(result.cache_hits, "retrieval", None),
    )
    return response

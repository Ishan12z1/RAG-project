from fastapi import APIRouter, Depends, Request
from app.schemas import (
    TimingInfo,
    ChatbotRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    Citation as SchemaCitation,
    CacheInfo,
    DebugInfo,
    ErrorResponse
)
from app.deps import get_app_version,get_pipeline
from rag.chat import RAGPipeline
from app.utils import get_used_citations,format_answer, log_json
from error_handler.errors import EmptyQueryError

router=APIRouter()

@router.get("/health",response_model=HealthResponse)
def health():
    return  HealthResponse(
    status="ok",
    service="rag-assistant",
    version=get_app_version(),
)

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
    result = pipeline.run(query=payload.query, top_k=payload.top_k)

    answer_text = format_answer(result.parsed_output)
    citations = get_used_citations(result)
    abstained = result.parsed_output.mode != "answer"

    schema_citations = [
        SchemaCitation(
            source=c.source,
            title=c.title,
            section=c.section,
            chunk_id=c.chunk_id,
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
            versions=result.versions
        )
    response = ChatResponse(
        answer=answer_text,
        citations=schema_citations,
        abstained=abstained,
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
            "query_length": len(payload.query),
            "top_k": payload.top_k,
            "abstained": abstained,
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
            "timings_ms": {
                "embed": result.timings_ms.embed,
                "retrieve": result.timings_ms.retrieve,
                "rerank": result.timings_ms.rerank,
                "generate": result.timings_ms.generate,
                "total": result.timings_ms.total,
            },
            "status": "ok",
        }
    )


    return response


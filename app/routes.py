from fastapi import APIRouter, Depends, Request
from app.schemas import TimingInfo,ChatbotRequest,ChatResponse, HealthResponse,MetricsResponse, Citation as SchemaCitation
from app.deps import get_app_version,get_pipeline
from rag.chat import RAGPipeline
from app.utils import get_used_citations,format_answer, log_json
router=APIRouter()

@router.get("/health",response_model=HealthResponse)
def health():
    return  HealthResponse(
    status="ok",
    service="rag-assistant",
    version=get_app_version(),
)

@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatbotRequest,
    request: Request,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    try:
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
            debug=None,
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

    except Exception as e:
        log_json(
            {
                "event": "chat_request",
                "request_id": getattr(request.state, "request_id", None),
                "path": "/chat",
                "query_length": len(payload.query) if payload and payload.query else 0,
                "top_k": getattr(payload, "top_k", None),
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
        )
        raise
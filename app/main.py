from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4

from app.routes import router
from app.deps import get_app_version, get_runtime_config
from app.schemas import ErrorResponse, ErrorDetail
from app.utils import configure_logging, log_json
from error_handler import RAGAppError

def create_app()->FastAPI:
    configure_logging()
    cfg = get_runtime_config()
    app=FastAPI(
        title="RAG Assistant API",
        version=get_app_version(),
        description="FastAPI service for RAG chatbot"
    )
    
    cors_cfg = cfg.get("cors", {}) or {}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_cfg.get("allow_origins", []),
        allow_credentials=bool(cors_cfg.get("allow_credentials", True)),
        allow_methods=cors_cfg.get("allow_methods", ["*"]),
        allow_headers=cors_cfg.get("allow_headers", ["*"]),
    )
    app.include_router(router)

    @app.middleware("http")
    async def add_request_id(request:Request, call_next):
        request.state.request_id=str(uuid4())
        response=await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RAGAppError)
    async def handle_rag_app_error(request: Request, exc: RAGAppError):
        request_id = getattr(request.state, "request_id", None)

        log_json(
            {
                "event": "request_error",
                "request_id": request_id,
                "path": request.url.path,
                "status": "error",
                "error_type": exc.code,
                "error_message": exc.message,
                "status_code": exc.status_code,
            }
        )

        body = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=exc.message),
            request_id=request_id,
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)

        log_json(
            {
                "event": "request_error",
                "request_id": request_id,
                "path": request.url.path,
                "status": "error",
                "error_type": "INVALID_REQUEST",
                "error_message": "Request validation failed.",
                "status_code": 422,
                "details": exc.errors(),
            }
        )

        body = ErrorResponse(
            error=ErrorDetail(
                code="INVALID_REQUEST",
                message="Request validation failed.",
            ),
            request_id=request_id,
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)

        log_json(
            {
                "event": "request_error",
                "request_id": request_id,
                "path": request.url.path,
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "status_code": 500,
            }
        )

        body = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="Internal server error.",
            ),
            request_id=request_id,
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    return app


app=create_app()
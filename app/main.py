from fastapi import FastAPI, Request
from app.routes import router
from app.deps import get_app_version
from app.utils import configure_logging
from uuid import uuid4

def create_app()->FastAPI:
    configure_logging()

    app=FastAPI(
        title="RAG Assistant API",
        version=get_app_version(),
        description="FastAPI service for RAG chatbot"
    )
    
    app.include_router(router)

    @app.middleware("http")
    async def add_request_id(request:Request, call_next):
        request.state.request_id=str(uuid4())
        response=await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


    return app


app=create_app()
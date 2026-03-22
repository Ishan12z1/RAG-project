from __future__ import annotations

from typing import Tuple

from app.deps import get_runtime_config, get_pipeline, get_app_version
from app.schemas import HealthCheck, HealthDetails, HealthResponse


def _check_config() -> HealthCheck:
    try:
        cfg = get_runtime_config()
        if not isinstance(cfg, dict) or not cfg:
            return HealthCheck(ready=False, detail="Runtime config is empty.")
        return HealthCheck(ready=True, detail="Runtime config loaded.")
    except Exception as e:
        return HealthCheck(ready=False, detail=f"Failed to load config: {type(e).__name__}")


def _check_pipeline() -> tuple[HealthCheck, object | None]:
    try:
        pipeline = get_pipeline()
        return HealthCheck(ready=True, detail="Pipeline initialized."), pipeline
    except Exception as e:
        return HealthCheck(ready=False, detail=f"Pipeline init failed: {type(e).__name__}"), None


def _check_dense_index(pipeline) -> HealthCheck:
    try:
        dense = pipeline.hybrid_retriever.dense
        ntotal = int(getattr(dense.index, "ntotal", 0))
        dim = int(getattr(dense, "dim", 0))
        if ntotal <= 0:
            return HealthCheck(ready=False, detail="FAISS index loaded but empty.")
        return HealthCheck(ready=True, detail=f"FAISS index loaded: ntotal={ntotal}, dim={dim}.")
    except Exception as e:
        return HealthCheck(ready=False, detail=f"Dense index check failed: {type(e).__name__}")


def _check_chunk_store(pipeline) -> HealthCheck:
    try:
        chunk_count = len(getattr(pipeline.hybrid_retriever.chunk_store, "_chunks", {}))
        if chunk_count <= 0:
            return HealthCheck(ready=False, detail="Chunk store is empty.")
        return HealthCheck(ready=True, detail=f"Chunk store loaded: {chunk_count} chunks.")
    except Exception as e:
        return HealthCheck(ready=False, detail=f"Chunk store check failed: {type(e).__name__}")


def _check_bm25(pipeline) -> HealthCheck:
    try:
        bm25 = pipeline.hybrid_retriever.bm25
        n_docs = int(getattr(bm25, "N", 0))
        if n_docs <= 0:
            return HealthCheck(ready=False, detail="BM25 index loaded but empty.")
        return HealthCheck(ready=True, detail=f"BM25 loaded: N={n_docs}.")
    except Exception as e:
        return HealthCheck(ready=False, detail=f"BM25 check failed: {type(e).__name__}")


def _check_reranker(pipeline) -> HealthCheck:
    try:
        cfg = pipeline.reranker.cfg
        if cfg.model_type == "local":
            has_model = hasattr(pipeline.reranker, "model")
            if not has_model:
                return HealthCheck(ready=False, detail="Local reranker model not loaded.")
            return HealthCheck(ready=True, detail=f"Local reranker ready: {cfg.model_name}.")
        if cfg.model_type == "api":
            if not cfg.url:
                return HealthCheck(ready=False, detail="Reranker API URL missing.")
            return HealthCheck(ready=True, detail=f"Reranker API configured: {cfg.model_name}.")
        return HealthCheck(ready=False, detail=f"Unknown reranker mode: {cfg.model_type}.")
    except Exception as e:
        return HealthCheck(ready=False, detail=f"Reranker check failed: {type(e).__name__}")


def _check_generation_provider(pipeline) -> HealthCheck:
    try:
        url = getattr(pipeline.model, "url", None)
        if not url:
            return HealthCheck(ready=False, detail="Generation provider URL missing.")
        return HealthCheck(ready=True, detail="Generation provider configured.")
    except Exception as e:
        return HealthCheck(ready=False, detail=f"Generation provider check failed: {type(e).__name__}")


def _check_versions(pipeline) -> HealthCheck:
    try:
        versions = getattr(pipeline, "versions", None)
        if not versions:
            return HealthCheck(ready=False, detail="Version metadata missing or empty.")
        return HealthCheck(ready=True, detail=f"Version metadata loaded: {len(versions)} fields.")
    except Exception as e:
        return HealthCheck(ready=False, detail=f"Version check failed: {type(e).__name__}")


def build_health_response() -> HealthResponse:
    cfg_check = _check_config()
    pipeline_check, pipeline = _check_pipeline()

    if pipeline is None:
        details = HealthDetails(
            config=cfg_check,
            pipeline=pipeline_check,
            dense_index=HealthCheck(ready=False, detail="Pipeline unavailable."),
            chunk_store=HealthCheck(ready=False, detail="Pipeline unavailable."),
            bm25=HealthCheck(ready=False, detail="Pipeline unavailable."),
            reranker=HealthCheck(ready=False, detail="Pipeline unavailable."),
            generation_provider=HealthCheck(ready=False, detail="Pipeline unavailable."),
            versions=HealthCheck(ready=False, detail="Pipeline unavailable."),
        )
        overall_status = "degraded" if cfg_check.ready else "error"
        return HealthResponse(
            status=overall_status,
            service="rag-assistant",
            version=get_app_version(),
            details=details,
        )

    dense_check = _check_dense_index(pipeline)
    chunk_store_check = _check_chunk_store(pipeline)
    bm25_check = _check_bm25(pipeline)
    reranker_check = _check_reranker(pipeline)
    generation_check = _check_generation_provider(pipeline)
    versions_check = _check_versions(pipeline)

    details = HealthDetails(
        config=cfg_check,
        pipeline=pipeline_check,
        dense_index=dense_check,
        chunk_store=chunk_store_check,
        bm25=bm25_check,
        reranker=reranker_check,
        generation_provider=generation_check,
        versions=versions_check,
    )

    all_ready = all(
        [
            cfg_check.ready,
            pipeline_check.ready,
            dense_check.ready,
            chunk_store_check.ready,
            bm25_check.ready,
            reranker_check.ready,
            generation_check.ready,
            versions_check.ready,
        ]
    )

    return HealthResponse(
        status="ok" if all_ready else "degraded",
        service="rag-assistant",
        version=get_app_version(),
        details=details,
    )
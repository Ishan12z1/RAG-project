from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from rag.prompt import PROMPT_VERSION


def _stringify(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def resolve_index_version(index_meta: Dict[str, Any], index_path: str) -> str:
    explicit = (
        index_meta.get("index_version")
        or index_meta.get("index_id")
        or index_meta.get("build_id")
    )
    if explicit:
        return _stringify(explicit)

    return Path(index_path).name


def build_pipeline_versions(
    *,
    config: Dict[str, Any],
    dense_retriever: Any,
    reranker: Any,
) -> Dict[str, str]:
    config_versions = dict(config.get("versions", {}) or {})
    index_meta = getattr(dense_retriever, "index_meta", {}) or {}
    emb_meta = getattr(dense_retriever, "emb_meta", {}) or {}

    versions: Dict[str, str] = {
        "app_version": _stringify(config_versions.get("app_version"), "0.1.0"),
        "service_name": _stringify(config_versions.get("service_name"), "rag-assistant"),
        "index_version": resolve_index_version(index_meta, config["indexing_loc"]),
        "embedding_model_version": _stringify(emb_meta.get("model_name")),
        "prompt_version": _stringify(PROMPT_VERSION),
        "reranker_version": _stringify(getattr(reranker.cfg, "model_name", None)),
        "reranker_mode": _stringify(getattr(reranker.cfg, "model_type", None)),
    }

    generation_model_name = config_versions.get("generation_model_name")
    if generation_model_name and str(generation_model_name).strip().lower() != "unknown":
        versions["generation_model_version"] = _stringify(generation_model_name)
    else:
        versions["generation_endpoint"] = _stringify(config.get("api_url"))

    return versions
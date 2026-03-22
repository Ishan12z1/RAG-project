from functools import lru_cache
from rag.chat import RAGPipeline
import yaml

CONFIG_PATH="configs/chat_config.yaml"

@lru_cache
def get_runtime_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def get_app_version()->str:
    cfg=get_runtime_config()
    return str(cfg.get("versions",{}).get("app_version","1.0"))

# using LRU cache here so that we won't need to load retriever, reranke, models on every request.
@lru_cache 
def get_pipeline()->RAGPipeline:
    return RAGPipeline(config_path=CONFIG_PATH)

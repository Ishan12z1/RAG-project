from functools import lru_cache
from rag.chat import RAGPipeline

CONFIG_PATH="configs/chat_config.yaml"


def get_app_version()->str:
    return "0.1.0"

# using LRU cache here so that we won't need to load retriever, reranke, models on every request.
@lru_cache 
def get_pipeline()->RAGPipeline:
    return RAGPipeline(config_path=CONFIG_PATH)

from __future__ import annotations

from rag.pipeline import answer_question
from rag.retrieval.retrieve import Retriever
from rag.model.model_ollamma import OllamaModel
from rag.model.model_Provider import ModelSpec
from rag.abstain import should_abstain

query="What is diabetes ?" 
index_loc="artifacts//index//flat_ip"
embed_loc="artifacts//embeddings"
chunks_loc="artifacts//processed_chunks.parquet"
model_name="phi3.5:3.8b-mini-instruct-q4_K_M"


retriver=Retriever(
    index_dir=index_loc,
    embeddings_dir=embed_loc,
    chunks_path=chunks_loc
)

chunks=retriver.retrieve(query)

model_spec=ModelSpec(model_name)
model=OllamaModel(model_spec)
answer=answer_question(
    query,
    chunks,
    model
)
print(answer.raw_text)
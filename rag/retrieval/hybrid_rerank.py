from rag.retrieval import HybridRetriever
from rag.rerank.base import Reranker
from rag.utils.contracts import RetrievedChunk

class HybridRerankRetriever:
    def __init__(self,retriever:HybridRetriever,reranker:Reranker) -> None:
        self.retriever=retriever
        self.reranker=reranker
    
    def retrieve(self,query:str,top_k:int)->list[RetrievedChunk]:
        chunks=self.retriever.retrieve(query=query,top_k=top_k)
        reranked_chunks=self.reranker.rerank(query=query,candidates=chunks)
        return  [chunk for chunk, score in sorted(reranked_chunks, key=lambda x: x[1], reverse=True)]
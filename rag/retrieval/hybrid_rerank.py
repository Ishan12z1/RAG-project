from rag.retrieval import HybridRetriever
from rag.rerank.base import Reranker
from rag.utils.contracts import RetrievedChunk
import time
class HybridRerankRetriever:
    def __init__(self,retriever:HybridRetriever,reranker:Reranker,candidate_k:int=25) -> None:
        self.retriever=retriever
        self.reranker=reranker
        self.candidate_k=candidate_k
    
    def retrieve(self,query:str,top_k:int)->tuple[list[RetrievedChunk],float,float,float]:

        chunks,embed_time,retrieve_time=self.retriever.retrieve(query=query,top_k=max(self.candidate_k,top_k))
        reranked_start=time.perf_counter()
        reranked_chunks=self.reranker.rerank(query=query,candidates=chunks)
        reranked_chunks.sort(key=lambda x:x[1],reverse=True)
        out = []
        for i, (chunk, sc) in enumerate(reranked_chunks[:top_k], start=1):
            out.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                score=sc,
                text=chunk.text,
                citation=chunk.citation,
                metadata=chunk.metadata,
                rank=i
            ))
        reranked_ms=(time.perf_counter()-reranked_start)*1000
        return out,embed_time,retrieve_time,reranked_ms
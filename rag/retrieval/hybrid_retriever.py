from __future__ import annotations

from rag.retrieval.bm25_index import BM25Index
from rag.retrieval.hybrid import blend_scores
from rag.retrieval.retrieve import Retriever
from rag.utils.contracts import RetrievedChunk
import pandas as pd 

class ChunkStore:
    def __init__(self,path:str):
        chunks_df=pd.read_parquet(path)
        self._chunks:dict[str,RetrievedChunk]={}

        for row in chunks_df.itertuples(index=False):

            self._chunks[row.chunk_id]=RetrievedChunk(
                chunk_id=row.chunk_id,
                score=0.0,
                text=row.chunk_text,
                citation=row.url,
                metadata=getattr(row,"metadata",None),
            )

    def get(self,id:str)->RetrievedChunk:
        try:
            return self._chunks[id]
        except KeyError as e :
            raise KeyError(f"Chunk id not found : {id}") from e

class HybridRetriever:
    def __init__(
        self,
        *,
        dense: Retriever,
        bm25: BM25Index,
        chunk_store: ChunkStore,
        alpha: float = 0.6,
        dense_candidate_k: int = 100,
        bm25_candidate_k: int = 100,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.chunk_store = chunk_store
        self.alpha = alpha
        self.dense_candidate_k = dense_candidate_k
        self.bm25_candidate_k = bm25_candidate_k

    
    def retrieve(self,query:str, top_k:int)->list[RetrievedChunk]:
        
        dense_hits=self.dense.retrieve(query,top_k=top_k)
        dense_by_id:dict[str,RetrievedChunk]={ch.chunk_id:ch for ch in dense_hits }
        
        dense_scores:dict[str,float]={ch.chunk_id : float(ch.score) for ch in dense_hits}

        bm25_hits= self.bm25.search(query,top_k)
        bm_25_scores={cid:float(scores) for cid,scores in bm25_hits}

        blended=blend_scores(dense_scores,bm_25_scores,alpha=self.alpha)
        ranked_ids = sorted(blended.items(), key=lambda x:x[1],reverse=True)[:top_k]
        out=[]
        for rank_idx, (cid,score) in enumerate(ranked_ids,start=1):
            ch=dense_by_id.get(cid)
            print("ch",ch)
            if ch is None:
                ch= self.chunk_store.get(cid)
            
            out.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    score=float(score),
                    text=ch.text,
                    citation=ch.citation,
                    metadata=ch.metadata,
                    rank=rank_idx,
                )
            )

        return out
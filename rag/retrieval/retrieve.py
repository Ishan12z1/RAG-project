# query → embedding → FAISS search → map indices to chunk_ids → fetch text/metadata → return results.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict

import numpy as np
import pandas as pd 
import faiss
import json

from rag.embedding.embedding_provider import ProviderSpec
from rag.embedding.hf_embeddings import SentTransEmb
from rag.utils.helper_functions import _l2_normalize, read_meta

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id:str
    score:float
    text:str
    metadata:Dict[str,Any]


def _load_chunk_ids(ids_path:Path)->List[str]:
    return ids_path.read_text(encoding="utf-8").splitlines()

def _load_chunks_df(paraquet_path:str)->pd.DataFrame:
    df=pd.read_parquet(paraquet_path)
    # chunk_id must be comparable to ids list type
    df["chunk_id"]=df["chunk_id"].astype(str)
    if df["chunk_id"].duplicated().any():
        raise ValueError("processed_chunks.parquet has duplicate chunk_id values")
    return df 

class Retriever:
    def __init__(
        self,
        index_dir:str,
        embeddings_dir:str,
        chunks_path:str="data/processed_chunks.parquet",
        device:str="cpu"
        ):
        self.index_dir=Path(index_dir)
        self.emb_dir=Path(embeddings_dir)
        
        self.index_meta=read_meta(self.index_dir)
        self.emb_meta=read_meta(self.emb_dir)

        self.index=faiss.read_index((str(self.index_dir/"faiss.index")))
        self.chunk_ids = _load_chunk_ids(self.emb_dir / "chunk_ids.jsonl")
        self.chunks_df = _load_chunks_df(chunks_path).set_index("chunk_id",drop=False)

        self.provider = SentTransEmb(ProviderSpec(model_name=self.emb_meta["model_name"]), device=device)

        self.normalized=bool(self.emb_meta.get("normalized",True))
        self.dim = int(self.emb_meta["dim"])
    

    def retrieve(self,query:str,top_k:int=5)->List[RetrievedChunk]:

        q_text=query.strip()

        # BGE query prefix (safe even if docs weren’t prefixed; main mismatch is doc side) 
        if self.emb_meta["model_name"].startswith("BAAI/bge"):
            q_text=f"query: {q_text}"

        q_vec=self.provider.embed_texts([q_text])
        q=np.array(q_vec,dtype=np.float32)
        if q.shape[1] !=self.dim:
            raise ValueError(f"Query dim {q.shape[1]} != index dim {self.dim}")
        q=np.ascontiguousarray(q)

        if self.normalized:
            q=_l2_normalize(q)

        #Calls self.index.search(q, top_k):
            # D = scores/distances (shape (1, top_k))
            # I = row indices in the FAISS index (shape (1, top_k))
        D,I =self.index.search(q,top_k)
        
        results:List[RetrievedChunk]=[]

        for score,row_idx in zip(D[0].tolist(),I[0].tolist()):
            if row_idx< 0 :
                continue 

            cid=self.chunk_ids[row_idx]
            if str(cid) not in self.chunks_df.index:
                continue

            r0=self.chunks_df.loc[str(cid)].to_dict()
            text=str(r0.pop("chunk_text"))
            chunk_id=str(r0.pop('chunk_id'))
            metadata=r0

            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                score=float(score),
                metadata=metadata
            ))
        return results

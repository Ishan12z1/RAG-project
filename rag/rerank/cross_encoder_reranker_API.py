from __future__ import annotations
from rag.rerank.base import Reranker
from rag.rerank.utils import batch_iter, truncate_text,min_max
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from rag.utils.contracts import RetrievedChunk
from typing import Sequence, Optional
import requests
import torch

@dataclass
class CrossEncoderConfig:
    model_name:str="cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_type:str="api" # can be api, local
    batch_size:int=16
    max_text_chars:int=2000
    normalize_scores:bool=False
    device:str="cuda" if torch.cuda.is_available() else "cpu"
    url:Optional[str]=None
class APIModel():
    def __init__(self):
        pass

class CrossEncoderReranker(Reranker):
    def __init__(self,config:CrossEncoderConfig=CrossEncoderConfig()):
        self.cfg=config
        if self.cfg.model_type=="local":
            if self.cfg.device:
                self.model=CrossEncoder(self.cfg.model_name,device=self.cfg.device)
            else:
                self.model=CrossEncoder(self.cfg.model_name)

    def rerank(self,query:str, candidates:Sequence[RetrievedChunk])->list[tuple[RetrievedChunk,float]] | None:

        if not candidates:
            return []
        
        chunks:list[RetrievedChunk]=[]
        pairs:list[tuple[str,str]]=[]

        for c in candidates:
            chunks.append(c)
            text=truncate_text(c.text,self.cfg.max_text_chars)
            pairs.append((query,text))
        
        scores:list[float]=[]
        for batch in batch_iter(pairs,self.cfg.batch_size):
            batch_score=self.get_score(batch)
            for s in batch_score:
                scores.append(float(s))
        
        if len(scores)!= len(chunks):
            raise RuntimeError("CrossEncoder returned mismatched score count")
        
        if self.cfg.normalize_scores:
            scores=min_max(scores)
        
        return sorted(zip(chunks,scores),key=lambda x: x[1],reverse=True)

    
    def get_score(self,pairs:list[tuple[str,str]]):

        if self.cfg.model_type=="local":            
            return self.model.predict(pairs)

        elif self.cfg.model_type=="api": 
            if self.cfg.url==None:
                raise ValueError("URL missing in config, url is required for api cross encoder")

            response = requests.post(self.cfg.url+"/rerank", json=pairs, timeout=120)
            response.raise_for_status()
            scores = response.json()["scores"]
            return[float(x) for x in scores]
        
        else:
            raise ValueError(f"Model type is incoorect can only be 'local' or 'api' not {self.cfg.model_type}")
        

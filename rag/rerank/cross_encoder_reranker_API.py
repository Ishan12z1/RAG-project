from __future__ import annotations
from rag.rerank.base import Reranker
from rag.rerank.utils import batch_iter, truncate_text,min_max
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from rag.utils.contracts import RetrievedChunk
from typing import Sequence, Optional,Any
import requests
import torch
import yaml
from rag.utils.http_retry import HTTPRetryPolicy,post_json_with_retry
@dataclass
class CrossEncoderConfig:
    model_name:str="cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_type:str="api" # can be api, local
    batch_size:int=16
    max_text_chars:int=2000
    normalize_scores:bool=False
    device:str="cuda" if torch.cuda.is_available() else "cpu"
    url:Optional[str]=None
    timeout_s: float = 20.0
    max_retries: int = 1
    backoff_s: float = 0.5
    backoff_multiplier: float = 2.0

class APIModel():
    def __init__(self):
        pass

class CrossEncoderReranker(Reranker):
    def __init__(self,config_path:str):
        self.cfg=self._load_config(config_path) 
        self.last_call_meta: dict[str, Any] = {
            "stage": "rerank",
            "batches": 0,
            "attempts": 0,
            "retries_used": 0,
            "timeouts": 0,
            "timeout_s": self.cfg.timeout_s,
        }
        if self.cfg.model_type=="local":
            if self.cfg.device:
                self.model=CrossEncoder(self.cfg.model_name,device=self.cfg.device)
            else:
                self.model=CrossEncoder(self.cfg.model_name)

    def _load_config(self,path:str)->CrossEncoderConfig:
        with open(path,"r") as f:
            config=yaml.safe_load(f)

        return CrossEncoderConfig(
            model_name=config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            model_type=config.get("model_type", "api"),
            batch_size=config.get("batch_size", 16),
            max_text_chars=config.get("max_text_chars", 2000),
            normalize_scores=config.get("normalize_scores", False),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            url=config.get("url"),
            timeout_s=float(config.get("timeout_s", 20.0)),
            max_retries=int(config.get("max_retries", 1)),
            backoff_s=float(config.get("backoff_s", 0.5)),
            backoff_multiplier=float(config.get("backoff_multiplier", 2.0)),
        )

    def rerank(self,query:str, candidates:Sequence[RetrievedChunk])->list[tuple[RetrievedChunk,float]]:

        if not candidates:
            raise KeyError("Candidates not found")
        
        chunks:list[RetrievedChunk]=[]
        pairs:list[tuple[str,str]]=[]

        for c in candidates:
            chunks.append(c)
            text=truncate_text(c.text,self.cfg.max_text_chars)
            pairs.append((query,text))
        
        scores:list[float]=[]
        aggregate_meta = {
            "stage": "rerank",
            "batches": 0,
            "attempts": 0,
            "retries_used": 0,
            "timeouts": 0,
            "timeout_s": self.cfg.timeout_s,
        }
        for batch in batch_iter(pairs,self.cfg.batch_size):
            batch_score,batch_meta=self.get_score(batch)
            aggregate_meta["batches"] += 1
            aggregate_meta["attempts"] += int(batch_meta.get("attempts", 0))
            aggregate_meta["retries_used"] += int(batch_meta.get("retries_used", 0))
            aggregate_meta["timeouts"] += int(batch_meta.get("timeouts", 0))
            
            for s in batch_score:
                scores.append(float(s))
        
        self.last_call_meta=aggregate_meta

        if len(scores)!= len(chunks):
            raise RuntimeError("CrossEncoder returned mismatched score count")
        
        if self.cfg.normalize_scores:
            scores=min_max(scores)
        
        return sorted(zip(chunks,scores),key=lambda x: x[1],reverse=True)

    
    def get_score(self,pairs:list[tuple[str,str]]):

        if self.cfg.model_type=="local":            
            scores = self.model.predict(pairs)
            return scores, {
                "stage": "rerank",
                "attempts": 1,
                "retries_used": 0,
                "timeouts": 0,
                "timeout_s": None,
            }

        elif self.cfg.model_type=="api": 
            if self.cfg.url==None:
                raise ValueError("URL missing in config, url is required for api cross encoder")

            policy = HTTPRetryPolicy(
                timeout_s=self.cfg.timeout_s,
                max_retries=self.cfg.max_retries,
                backoff_s=self.cfg.backoff_s,
                backoff_multiplier=self.cfg.backoff_multiplier,
            )

            response, meta = post_json_with_retry(
                url=self.cfg.url.rstrip("/") + "/rerank",
                payload=pairs,
                stage="rerank",
                policy=policy,
            )
            scores = response.json()["scores"]
            return [float(x) for x in scores], meta
        
        else:
            raise ValueError(f"Model type is incoorect can only be 'local' or 'api' not {self.cfg.model_type}")
        

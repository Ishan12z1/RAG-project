from __future__ import annotations

from rag.retrieval.bm25_index import BM25Index
from rag.retrieval.hybrid import blend_scores
from rag.retrieval.retrieve import Retriever
from rag.utils.contracts import RetrievedChunk, Citation
import pandas as pd 
from rag.retrieval.utils import build_retrieved_chunk_from_row
import time
import numpy as np 
from collections import OrderedDict
from hashlib import sha256
from threading import Lock

class ChunkStore:
    def __init__(self, path: str):
        chunks_df = pd.read_parquet(path)
        chunks_df["chunk_id"] = chunks_df["chunk_id"].astype(str)

        if chunks_df["chunk_id"].duplicated().any():
            raise ValueError("ChunkStore parquet has duplicate chunk_id values")

        self._chunks: dict[str, RetrievedChunk] = {}

        for _, row in chunks_df.iterrows():
            chunk = build_retrieved_chunk_from_row(row.to_dict(), score=0.0)
            self._chunks[chunk.chunk_id] = chunk

    def get(self, id: str) -> RetrievedChunk:
        try:
            return self._chunks[str(id)]
        except KeyError as e:
            raise KeyError(f"Chunk id not found: {id}") from e

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
        retrieval_cache_size:int=512
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.chunk_store = chunk_store
        self.alpha = alpha
        self.dense_candidate_k = dense_candidate_k
        self.bm25_candidate_k = bm25_candidate_k

        self.index_version = self._resolve_index_version()
        self.bm25_fingerprint = str(getattr(self.bm25, "corpus_fingerprint", "unknown"))

        self.retrieval_cache_size = retrieval_cache_size
        self._retrieval_cache: OrderedDict[str, tuple[RetrievedChunk, ...]] = OrderedDict()
        self._retrieval_cache_lock = Lock()
        self._retrieval_cache_hits = 0
        self._retrieval_cache_misses = 0
    
    
    
    def _resolve_index_version(self) -> str:
        index_meta = getattr(self.dense, "index_meta", {}) or {}
        explicit = (
            index_meta.get("index_version")
            or index_meta.get("index_id")
            or index_meta.get("build_id")
        )
        if explicit:
            return str(explicit)
        return self.dense.index_dir.name

    def _normalize_lexical_query(self, query: str) -> str:
        return " ".join(query.strip().split())

    def _hash_query_vector(self, q: np.ndarray) -> str:
        q_bytes = np.ascontiguousarray(q, dtype=np.float32).tobytes()
        return sha256(q_bytes).hexdigest()

    def _hash_lexical_query(self, query: str) -> str:
        return sha256(self._normalize_lexical_query(query).encode("utf-8")).hexdigest()

    def _make_retrieval_cache_key(
        self,
        *,
        embedding_hash: str,
        lexical_hash: str,
        top_k: int,
    ) -> str:
        return "|".join(
            [
                f"embedding={embedding_hash}",
                f"lexical={lexical_hash}",
                f"index_version={self.index_version}",
                f"bm25={self.bm25_fingerprint}",
                f"alpha={self.alpha}",
                f"top_k={top_k}",
                f"dense_k={self.dense_candidate_k}",
                f"bm25_k={self.bm25_candidate_k}",
            ]
        )

    def _get_cached_retrieval(self, cache_key: str) -> tuple[RetrievedChunk, ...] | None:
        with self._retrieval_cache_lock:
            cached = self._retrieval_cache.get(cache_key)
            if cached is None:
                return None

            self._retrieval_cache.move_to_end(cache_key)
            self._retrieval_cache_hits += 1
            return cached

    def _store_retrieval(self, cache_key: str, chunks: list[RetrievedChunk]) -> None:
        with self._retrieval_cache_lock:
            self._retrieval_cache_misses += 1
            self._retrieval_cache[cache_key] = tuple(chunks)
            self._retrieval_cache.move_to_end(cache_key)

            if len(self._retrieval_cache) > self.retrieval_cache_size:
                self._retrieval_cache.popitem(last=False)

    def get_retrieval_cache_stats(self) -> dict[str, int | str]:
        with self._retrieval_cache_lock:
            return {
                "hits": self._retrieval_cache_hits,
                "misses": self._retrieval_cache_misses,
                "size": len(self._retrieval_cache),
                "capacity": self.retrieval_cache_size,
                "index_version": self.index_version,
                "bm25_fingerprint": self.bm25_fingerprint,
            }
    def retrieve(self,query:str, top_k:int)->tuple[list[RetrievedChunk],float,float,bool,bool]:
        hybrid_retrieve_start=time.perf_counter()

        q_vec, embed_time, embedding_cache_hit = self.dense._embed_query(query)

        if not query.strip():
            return [], 0.0, 0.0, False, False
        embedding_hash = self._hash_query_vector(q_vec)
        lexical_hash = self._hash_lexical_query(query)
        cache_key = self._make_retrieval_cache_key(
            embedding_hash=embedding_hash,
            lexical_hash=lexical_hash,
            top_k=top_k,
        )
        cached = self._get_cached_retrieval(cache_key)
        if cached is not None:
            total_hybrid_retrieve_time = (time.perf_counter() - hybrid_retrieve_start) * 1000
            return list(cached), embed_time, total_hybrid_retrieve_time, embedding_cache_hit, True
        
        dense_hits, base_retrieve_time = self.dense.search_with_query_vector(
                    q_vec,
                    top_k=max(self.dense_candidate_k, top_k),
                )
        
        dense_by_id:dict[str,RetrievedChunk]={ch.chunk_id:ch for ch in dense_hits }
        
        dense_scores:dict[str,float]={ch.chunk_id : float(ch.score) for ch in dense_hits}

        bm25_hits= self.bm25.search(query,top_k)
        bm_25_scores={cid:float(scores) for cid,scores in bm25_hits}

        blended=blend_scores(dense_scores,bm_25_scores,alpha=self.alpha)
        ranked_ids = sorted(blended.items(), key=lambda x:x[1],reverse=True)[:top_k]
        out=[]
        for rank_idx, (cid,score) in enumerate(ranked_ids,start=1):
            ch=dense_by_id.get(cid)
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
        self._store_retrieval(cache_key, out)

        total_hybrid_retrieve_time=(time.perf_counter()-hybrid_retrieve_start)*1000
        return out,embed_time,total_hybrid_retrieve_time,embedding_cache_hit ,False
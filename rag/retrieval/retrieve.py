# query → embedding → FAISS search → map indices to chunk_ids → fetch text/metadata → return results.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd 
import faiss
import json
from collections import OrderedDict
from threading import Lock
from rag.embedding.embedding_provider import ProviderSpec
from rag.embedding.hf_embeddings import SentTransEmb
from rag.utils.helper_functions import _l2_normalize, read_meta
from rag.utils.contracts import Citation,RetrievedChunk
from rag.retrieval.utils import build_retrieved_chunk_from_row
import time
REQUIRED_CITATION_KEYS = ("source", "title", "section", "chunk_id","doc_id","url")



def _load_chunk_ids(ids_path:Path)->List[str]:
    return ids_path.read_text(encoding="utf-8").splitlines()

def _load_chunks_df(paraquet_path:str)->pd.DataFrame:
    df=pd.read_parquet(paraquet_path)
    # chunk_id must be comparable to ids list type
    df["chunk_id"]=df["chunk_id"].astype(str)
    if df["chunk_id"].duplicated().any():
        raise ValueError("processed_chunks.parquet has duplicate chunk_id values")
    return df 

def _normalize_citation_fields(row: Dict[str, Any]) -> Tuple[Citation, Dict[str, Any]]:
    """
    Ensures citation has source/title/section/chunk_id.
    If parquet uses different keys, map them here.
    """

    # Aliases you might have in your parquet
    key_map = {
        "source_url": "url",
        "doc_url": "source",
        "document_url": "source",
        "heading": "title",
        "section_path":"section",
        "header": "section",
        "source_url": "url",
        "document_url": "url",
        "doc_url": "url",
    }

    norm = dict(row)
    for k_from, k_to in key_map.items():
        if k_to not in norm and k_from in norm:
            norm[k_to] = norm[k_from]

    missing = [k for k in ("source", "title", "section", "chunk_id","doc_id","url") if k not in norm or norm[k] is None]
    if missing:
        raise ValueError(
            f"Missing required citation fields {missing}. "
            f"Ensure processed_chunks.parquet includes {REQUIRED_CITATION_KEYS} (or add alias mapping)."
        )

    citation = Citation(
        source=str(norm["source"]),
        title=str(norm["title"]),
        section=str(norm["section"]),
        chunk_id=str(norm["chunk_id"]),
        doc_id=str(norm["doc_id"]),
        url=str(norm["url"])
    )

    # metadata = everything else except required fields + chunk_text
    for k in ("source", "title", "section","doc_id","url"):
        norm.pop(k, None)
    # chunk_id stays in metadata? keep it only once
    norm.pop("chunk_id", None)
    norm.pop("chunk_text", None)

    return citation, norm

def _standardize_score(raw_score: float, score_mode: str) -> float:
    """
    Standardize scores to "higher is better".
    score_mode:
      - "ip" / "cosine": higher is better
      - "l2": lower is better -> convert to negative distance
    """
    mode = (score_mode or "").lower()
    if mode in ("l2", "euclidean"):
        return -float(raw_score)
    # default: treat as similarity
    return float(raw_score)

def _match_filters(md: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Minimal filter language:
      - exact match: {"doc_id": "X"}
      - membership: {"source": {"a","b"}} or {"source": ["a","b"]}
      - callable: {"year": lambda y: int(y) >= 2020}
    """
    for k, want in filters.items():
        have = md.get(k)

        if callable(want):
            try:
                if not want(have):
                    return False
            except Exception:
                return False
            continue

        if isinstance(want, (set, list, tuple)):
            if have not in want:
                return False
            continue

        # exact
        if have != want:
            return False
    return True

class Retriever:
    def __init__(
        self,
        index_dir:str,
        embeddings_dir:str,
        chunks_path:str="data/processed_chunks.parquet",
        device:str="cpu",
        embedding_cache_size:int=512,
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
        self.score_mode = str(self.index_meta.get("score_mode", "")).lower().strip()
        if not self.score_mode:
            # Heuristic fallback
            # Many cosine/IP setups use IndexFlatIP / IVF with METRIC_INNER_PRODUCT.
            # If METRIC_L2 => distances.
            metric = getattr(self.index, "metric_type", None)
            if metric == faiss.METRIC_L2:
                self.score_mode = "l2"
            else:
                self.score_mode = "ip"
        # in memory caching 
        self.embedding_cache_size=embedding_cache_size
        self._embedding_cache:OrderedDict[tuple[str,str],np.ndarray]=OrderedDict()
        self._embedding_cache_lock=Lock()
        self._embedding_cache_hits = 0
        self._embedding_cache_misses = 0


    def _normalize_query_for_cache(self,query:str)->str:
        return " ".join(query.strip().split())

    def _prepare_query_text(self, query: str) -> str:
        q_text = query.strip()
        if self.emb_meta["model_name"].startswith("BAAI/bge"):
            q_text = f"query: {q_text}"
        return q_text
    
    def _embedding_cache_key(self, query: str) -> tuple[str, str]:
        return (
            str(self.emb_meta["model_name"]),
            self._normalize_query_for_cache(query),
        )
    
    def _get_cached_embedding(self, cache_key: tuple[str, str]) -> Optional[np.ndarray]:
        with self._embedding_cache_lock:
            cached = self._embedding_cache.get(cache_key)
            if cached is None:
                return None

            self._embedding_cache.move_to_end(cache_key)
            self._embedding_cache_hits += 1
            return np.ascontiguousarray(cached.copy())
    
    def _store_embedding(self, cache_key: tuple[str, str], query_vector: np.ndarray) -> None:
        with self._embedding_cache_lock:
            self._embedding_cache_misses += 1
            self._embedding_cache[cache_key] = np.ascontiguousarray(query_vector.copy())
            self._embedding_cache.move_to_end(cache_key)

            if len(self._embedding_cache) > self.embedding_cache_size:
                self._embedding_cache.popitem(last=False)

    def _embed_query(self, query: str) -> tuple[np.ndarray, float, bool]:
        cache_key = self._embedding_cache_key(query)

        cached = self._get_cached_embedding(cache_key)
        if cached is not None:
            return cached, 0.0, True

        q_text = self._prepare_query_text(query)

        embed_start = time.perf_counter()
        q_vec = self.provider.embed_texts([q_text])
        q = np.array(q_vec, dtype=np.float32)

        if q.ndim != 2 or q.shape[0] != 1 or q.shape[1] != self.dim:
            raise ValueError(f"Bad query shape {q.shape}; expected (1, {self.dim})")

        q = np.ascontiguousarray(q)

        if self.normalized:
            q = _l2_normalize(q)

        embed_time_ms = (time.perf_counter() - embed_start) * 1000
        self._store_embedding(cache_key, q)
        return q, embed_time_ms, False
    
    def get_embedding_cache_stats(self) -> Dict[str, int]:
        with self._embedding_cache_lock:
            return {
                "hits": self._embedding_cache_hits,
                "misses": self._embedding_cache_misses,
                "size": len(self._embedding_cache),
                "capacity": self.embedding_cache_size,
            }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        oversample: int = 2,
    ) -> tuple[List[RetrievedChunk], float, float, bool]:
        start_time = time.perf_counter()

        if not query.strip():
            return [], 0.0, 0.0, False
        q, embed_time_ms, embedding_cache_hit = self._embed_query(query)

        k_search = max(top_k, 1) * max(oversample, 1)
        D, I = self.index.search(q, k_search)

        candidates: List[RetrievedChunk] = []
        filt = filters or {}

        for raw_score, row_idx in zip(D[0].tolist(), I[0].tolist()):
            if row_idx < 0 or row_idx >= len(self.chunk_ids):
                continue

            cid = self.chunk_ids[row_idx]
            cid_str = str(cid)

            if cid_str not in self.chunks_df.index:
                continue

            r0 = self.chunks_df.loc[cid_str].to_dict()
            r0["chunk_id"] = str(r0.get("chunk_id", cid))

            citation, extra_md = _normalize_citation_fields(r0)

            md_view = {
                **extra_md,
                "source": citation.source,
                "title": citation.title,
                "section": citation.section,
                "doc_id": citation.doc_id,
                "url": citation.url,
                "chunk_id": citation.chunk_id,
            }

            if filt and not _match_filters(md_view, filt):
                continue

            score = _standardize_score(raw_score, self.score_mode)
            chunk = build_retrieved_chunk_from_row(r0, score=score)
            candidates.append(chunk)

            if len(candidates) >= top_k and not filt:
                break

        candidates.sort(key=lambda r: (-r.score, r.chunk_id))

        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        return candidates[:top_k], embed_time_ms, total_time_ms, embedding_cache_hit
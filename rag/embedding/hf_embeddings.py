from __future__ import annotations
from typing import List,Optional
import numpy as np 
from rag.embedding.embedding_provider import EmbeddingProvider, ProviderSpec

class SentTransEmb(EmbeddingProvider):
    def __init__(self,spec:ProviderSpec, device:Optional[str]=None):
        self.spec=spec

        from sentence_transformers import SentenceTransformer
        import torch
        if device is None:
            device="cuda" if torch.cuda.is_available() else "cpu"
        
        self.model=SentenceTransformer(self.spec.model_name,device=device)
    
    def embed_texts(self, texts):
        if not texts:
            return []
        arr=self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=False # Handeled by rag.embed
        )

        arr=np.asarray(arr, dtype=np.float32)

        if arr.ndim==1:
            arr=arr.reshape(1,-1)
        return arr.tolist()
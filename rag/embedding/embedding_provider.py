from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol

class EmbeddingProvider(Protocol):
    def embed_texts(self,texts:List[str])->List[List[float]]:
        ...

@dataclass(frozen=True)
class ProviderSpec:
    model_name:str
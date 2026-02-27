from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol,Union,Dict

class ModelProvider(Protocol):
    def get_response(self, message: Union[str, List[Dict[str, str]]]) -> str:
        ...

@dataclass(frozen=True)
class ModelSpec:
    model_name:str
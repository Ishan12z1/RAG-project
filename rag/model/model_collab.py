import requests
from typing import Union, List, Dict

class CollabModel:
    def __init__(self, url: str, max_new_tokens: int = 200, temperature: float = 0.6):
        self.url = url.rstrip("/") + "/generate"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def get_response(self, message: Union[str, List[Dict[str, str]], dict]) -> str:
        payload = {
            "prompt":         message,
            "max_new_tokens": self.max_new_tokens,
            "temperature":    self.temperature,
        }
        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]
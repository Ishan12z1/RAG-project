import requests
from typing import Union, List, Dict,Any
from rag.utils.http_retry import HTTPRetryPolicy, post_json_with_retry

class CollabModel:
    def __init__(
        self,
        url: str,
        max_new_tokens: int = 200,
        temperature: float = 0.6,
        timeout_s: float = 60.0,
        max_retries: int = 0,
        backoff_s: float = 0.0,
    ):
        self.url = url.rstrip("/") + "/generate"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.policy = HTTPRetryPolicy(
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                    backoff_s=backoff_s,
                )
        self.last_call_meta: Dict[str, Any] = {
                    "stage": "generation",
                    "attempts": 0,
                    "retries_used": 0,
                    "timeouts": 0,
                    "timeout_s": timeout_s,
        }
    def get_response(self, message: Union[str, List[Dict[str, str]], dict]) -> str:
        payload = {
            "prompt":         message,
            "max_new_tokens": self.max_new_tokens,
            "temperature":    self.temperature,
        }
        response, meta = post_json_with_retry(
            url=self.url,
            payload=payload,
            stage="generation",
            policy=self.policy,
        )
        self.last_call_meta=meta
        return response.json()["response"]
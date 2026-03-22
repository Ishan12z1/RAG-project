from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import requests


logger = logging.getLogger("rag_api")


@dataclass(frozen=True)
class HTTPRetryPolicy:
    timeout_s: float
    max_retries: int = 0
    backoff_s: float = 0.5
    backoff_multiplier: float = 2.0
    retry_on_statuses: tuple[int, ...] = (502, 503, 504)


def _log_retry_event(payload: Dict[str, Any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def post_json_with_retry(
    *,
    url: str,
    payload: Any,
    stage: str,
    policy: HTTPRetryPolicy,
) -> Tuple[requests.Response, Dict[str, Any]]:
    attempt = 0
    retries_used = 0
    timeout_count = 0
    current_backoff = policy.backoff_s

    while True:
        attempt += 1
        started = time.perf_counter()

        try:
            response = requests.post(url, json=payload, timeout=policy.timeout_s)
            elapsed_ms = (time.perf_counter() - started) * 1000

            if response.status_code in policy.retry_on_statuses and attempt <= policy.max_retries + 1:
                if attempt <= policy.max_retries:
                    retries_used += 1
                    _log_retry_event(
                        {
                            "event": "http_retry",
                            "stage": stage,
                            "attempt": attempt,
                            "reason": f"http_{response.status_code}",
                            "timeout_s": policy.timeout_s,
                            "elapsed_ms": elapsed_ms,
                            "will_retry": True,
                        }
                    )
                    time.sleep(current_backoff)
                    current_backoff *= policy.backoff_multiplier
                    continue

            response.raise_for_status()

            return response, {
                "stage": stage,
                "attempts": attempt,
                "retries_used": retries_used,
                "timeouts": timeout_count,
                "timeout_s": policy.timeout_s,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
            }

        except requests.Timeout:
            elapsed_ms = (time.perf_counter() - started) * 1000
            timeout_count += 1

            if attempt <= policy.max_retries:
                retries_used += 1
                _log_retry_event(
                    {
                        "event": "http_retry",
                        "stage": stage,
                        "attempt": attempt,
                        "reason": "timeout",
                        "timeout_s": policy.timeout_s,
                        "elapsed_ms": elapsed_ms,
                        "will_retry": True,
                    }
                )
                time.sleep(current_backoff)
                current_backoff *= policy.backoff_multiplier
                continue

            _log_retry_event(
                {
                    "event": "http_retry",
                    "stage": stage,
                    "attempt": attempt,
                    "reason": "timeout",
                    "timeout_s": policy.timeout_s,
                    "elapsed_ms": elapsed_ms,
                    "will_retry": False,
                }
            )
            raise

        except requests.RequestException as e:
            elapsed_ms = (time.perf_counter() - started) * 1000
            status_code: Optional[int] = None
            if getattr(e, "response", None) is not None:
                status_code = e.response.status_code

            retriable = status_code in policy.retry_on_statuses if status_code is not None else False

            if retriable and attempt <= policy.max_retries:
                retries_used += 1
                _log_retry_event(
                    {
                        "event": "http_retry",
                        "stage": stage,
                        "attempt": attempt,
                        "reason": f"http_{status_code}",
                        "timeout_s": policy.timeout_s,
                        "elapsed_ms": elapsed_ms,
                        "will_retry": True,
                    }
                )
                time.sleep(current_backoff)
                current_backoff *= policy.backoff_multiplier
                continue

            _log_retry_event(
                {
                    "event": "http_retry",
                    "stage": stage,
                    "attempt": attempt,
                    "reason": f"http_{status_code}" if status_code is not None else type(e).__name__,
                    "timeout_s": policy.timeout_s,
                    "elapsed_ms": elapsed_ms,
                    "will_retry": False,
                }
            )
            raise
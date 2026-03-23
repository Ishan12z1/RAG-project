from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Optional


class RuntimeMetrics:
    def __init__(self, max_latency_samples: int = 500):
        self._lock = Lock()

        self.requests_total = 0
        self.errors_total = 0
        self.answer_total = 0
        self.abstain_total = 0
        self.parse_error_total = 0
        self.schema_valid_total = 0

        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0

        self.retrieval_cache_hits = 0
        self.retrieval_cache_misses = 0

        self.latency_samples_ms = deque(maxlen=max_latency_samples)

    def record_request(
        self,
        *,
        total_ms: Optional[float],
        mode: str,
        schema_valid: bool,
        embedding_cache_hit: Optional[bool],
        retrieval_cache_hit: Optional[bool],
    ) -> None:
        with self._lock:
            self.requests_total += 1

            if mode == "answer":
                self.answer_total += 1
            elif mode == "abstain":
                self.abstain_total += 1
            elif mode == "parse_error":
                self.parse_error_total += 1

            if schema_valid:
                self.schema_valid_total += 1

            if total_ms is not None:
                self.latency_samples_ms.append(float(total_ms))

            if embedding_cache_hit is True:
                self.embedding_cache_hits += 1
            elif embedding_cache_hit is False:
                self.embedding_cache_misses += 1

            if retrieval_cache_hit is True:
                self.retrieval_cache_hits += 1
            elif retrieval_cache_hit is False:
                self.retrieval_cache_misses += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors_total += 1

    def _percentile(self, values: list[float], pct: float) -> Optional[float]:
        if not values:
            return None
        values = sorted(values)
        if len(values) == 1:
            return values[0]

        idx = (len(values) - 1) * pct
        lower = int(idx)
        upper = min(lower + 1, len(values) - 1)
        fraction = idx - lower
        return values[lower] + (values[upper] - values[lower]) * fraction

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self.latency_samples_ms)

            embedding_total = self.embedding_cache_hits + self.embedding_cache_misses
            retrieval_total = self.retrieval_cache_hits + self.retrieval_cache_misses

            return {
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "answer_total": self.answer_total,
                "abstain_total": self.abstain_total,
                "parse_error_total": self.parse_error_total,
                "schema_valid_rate": (
                    self.schema_valid_total / self.requests_total if self.requests_total > 0 else None
                ),
                "embedding_cache_hit_rate": (
                    self.embedding_cache_hits / embedding_total if embedding_total > 0 else None
                ),
                "retrieval_cache_hit_rate": (
                    self.retrieval_cache_hits / retrieval_total if retrieval_total > 0 else None
                ),
                "p50_ms": self._percentile(latencies, 0.50),
                "p95_ms": self._percentile(latencies, 0.95),
            }


runtime_metrics = RuntimeMetrics()

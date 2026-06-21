"""In-memory metrics store for compression runs (thread-safe)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import AgentResult


@dataclass
class MetricEntry:
    reduction_pct: float
    meaning_score: int
    total_steps: int
    tokens_saved: int
    success: bool


class MetricsStore:
    def __init__(self):
        self._entries: list[MetricEntry] = []
        self._lock = Lock()

    def record(self, result: "AgentResult"):
        with self._lock:
            self._entries.append(
                MetricEntry(
                    reduction_pct=result.reduction_pct,
                    meaning_score=result.meaning_score,
                    total_steps=result.total_steps,
                    tokens_saved=result.tokens_saved,
                    success=result.success,
                )
            )

    def summary(self) -> dict:
        with self._lock:
            if not self._entries:
                return {
                    "total_requests": 0,
                    "avg_reduction_pct": 0,
                    "avg_meaning_score": 0,
                    "avg_steps": 0,
                    "avg_tokens_saved": 0,
                    "success_rate": 0,
                }
            return {
                "total_requests": len(self._entries),
                "avg_reduction_pct": round(
                    statistics.mean(e.reduction_pct for e in self._entries), 1
                ),
                "avg_meaning_score": round(
                    statistics.mean(e.meaning_score for e in self._entries), 1
                ),
                "avg_steps": round(
                    statistics.mean(e.total_steps for e in self._entries), 1
                ),
                "avg_tokens_saved": round(
                    statistics.mean(e.tokens_saved for e in self._entries), 1
                ),
                "success_rate": round(
                    sum(1 for e in self._entries if e.success)
                    / len(self._entries)
                    * 100,
                    1,
                ),
            }


metrics_store = MetricsStore()

# Latency Monitor — tracks wall-clock time at each pipeline stage per turn.
# Logs: VAD duration, STT duration, LLM first-token latency, TTS first-chunk latency, total.
# Used for both the fallback threshold check and the benchmark report.
# Implementation: Part 6
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class LatencyMonitor:
    """Per-turn stage timing. Call start_turn() once, mark(stage) at each
    checkpoint in the order stages actually occur, then log_summary() for a
    breakdown like "[stt: 68ms | llm_first_token: 587ms | total: 800ms]".

    mark() records elapsed time since start_turn(), not since the previous
    mark — log_summary() derives each stage's individual duration from the
    gap between consecutive marks, since the pipeline's stages are inherently
    sequential (one stage's start is the previous stage's end).
    """

    def __init__(self, config: dict) -> None:
        self._log_latency: bool = config["logging"]["log_latency"]
        self._t0: float | None = None
        self._marks: dict[str, float] = {}  # stage -> elapsed ms since start_turn()

    def start_turn(self) -> None:
        """Reset and begin timing a new conversation turn."""
        self._t0 = time.perf_counter()
        self._marks = {}

    def mark(self, stage: str) -> float:
        """Record elapsed ms since start_turn() for `stage`. Returns that value."""
        if self._t0 is None:
            raise RuntimeError("start_turn() must be called before mark()")
        elapsed_ms = (time.perf_counter() - self._t0) * 1000
        self._marks[stage] = elapsed_ms
        return elapsed_ms

    def elapsed_ms(self, stage: str) -> float | None:
        """ms since start_turn() at the time `stage` was marked, or None if unmarked."""
        return self._marks.get(stage)

    def log_summary(self) -> None:
        """Log a per-stage duration breakdown, gated on config.logging.log_latency."""
        if not self._log_latency or not self._marks:
            return

        parts = []
        previous = 0.0
        for stage, cumulative in self._marks.items():
            parts.append(f"{stage}: {cumulative - previous:.0f}ms")
            previous = cumulative
        parts.append(f"total: {previous:.0f}ms")
        logger.info("[%s]", " | ".join(parts))

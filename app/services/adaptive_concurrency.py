"""
Adaptive Concurrency Controller — Dynamic LLM call throttling tied to vLLM /metrics.

Reads real-time hardware state from vLLM's Prometheus /metrics endpoint
(via the VLLMEndpoint objects that poll every 5s) and dynamically adjusts
how many parallel LLM requests callers can fire.

Three vLLM signals drive the limit:
    1. gpu_cache_usage_perc  — KV cache pressure (0.0–1.0)
    2. num_requests_waiting  — server-side queue depth (requests backed up)
    3. num_requests_running  — requests being processed right now

Decision matrix:
    ┌─────────────────────────┬──────────────┬──────────────────────────────┐
    │ Condition               │ Limit        │ Rationale                    │
    ├─────────────────────────┼──────────────┼──────────────────────────────┤
    │ waiting > running       │ MIN (8)      │ vLLM is backed up            │
    │ cache > 80%             │ MIN (8)      │ KV cache about to evict      │
    │ cache > 60%             │ interpolated │ Moderate pressure            │
    │ waiting == 0, cache<60% │ MAX (16)     │ Plenty of room               │
    └─────────────────────────┴──────────────┴──────────────────────────────┘

Usage:
    from app.services.adaptive_concurrency import concurrency_controller

    results = await concurrency_controller.gather(tasks, label="data_janitor")

This is a drop-in replacement for ``asyncio.gather()`` with automatic
back-pressure. The limit re-evaluates every 5 seconds.
"""

import asyncio
import logging
import time
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ── Configurable bounds (via .env) ───────────────────────────────────
_MIN = getattr(settings, "ADAPTIVE_MIN_CONCURRENCY", 8)
_MAX = getattr(settings, "ADAPTIVE_MAX_CONCURRENCY", 16)

# How often (seconds) the controller re-evaluates the limit.
_REEVALUATE_INTERVAL = 5.0


class AdaptiveConcurrencyController:
    """Dynamic concurrency limiter tied to vLLM /metrics hardware state."""

    def __init__(
        self,
        min_concurrency: int = _MIN,
        max_concurrency: int = _MAX,
    ):
        self.min_concurrency = max(1, min_concurrency)
        self.max_concurrency = max(self.min_concurrency, max_concurrency)
        self._current_limit: int = self.max_concurrency
        self._last_eval: float = 0.0
        # Track per-label active counts for observability
        self._label_active: dict[str, int] = {}
        # Global concurrency tracking across all gathers
        self._active_tasks_count = 0
        self._cv = asyncio.Condition()

    async def _acquire_slot(self):
        async with self._cv:
            while self._active_tasks_count >= self._maybe_update_limit():
                await self._cv.wait()
            self._active_tasks_count += 1

    async def _release_slot(self):
        async with self._cv:
            self._active_tasks_count = max(0, self._active_tasks_count - 1)
            self._cv.notify_all()

    # ── vLLM /metrics readers ────────────────────────────────────────

    def _read_endpoints(self) -> list:
        """Fetch live VLLMEndpoint objects from the vLLM client singleton.

        Returns an empty list if the client isn't initialized yet
        (safe default = full concurrency).
        """
        try:
            from app.services.vllm_client import llm

            return [
                ep for ep in llm._endpoints.values()
                if ep.enabled and ep.model
            ]
        except Exception:
            return []

    def _avg_cache_usage(self) -> float:
        """Average KV cache usage (0.0–1.0) across all active endpoints."""
        endpoints = self._read_endpoints()
        if not endpoints:
            return 0.0
        return sum(ep.cache_usage for ep in endpoints) / len(endpoints)

    def _total_waiting(self) -> int:
        """Total requests currently waiting in vLLM server queues."""
        return sum(ep.requests_waiting for ep in self._read_endpoints())

    def _total_running(self) -> int:
        """Total requests currently being processed by vLLM servers."""
        return sum(ep.requests_running for ep in self._read_endpoints())

    def _total_capacity(self) -> int:
        """Sum of max_concurrent across all active endpoints."""
        return sum(ep.max_concurrent for ep in self._read_endpoints())

    # ── Limit calculation ────────────────────────────────────────────

    def _compute_limit(self) -> int:
        """Compute concurrency limit from live vLLM /metrics data.

        Uses three signals:
          1. Queue backpressure: if vLLM is queueing more requests than
             it's processing, we're sending too fast → back off to MIN.
          2. KV cache saturation: high cache % means the GPU is running
             out of room for new request KV blocks → throttle.
          3. Available capacity: if the server has plenty of room
             (low waiting, low cache) → allow MAX.

        The limit is a single integer in [min_concurrency, max_concurrency].
        """
        cache_pct = self._avg_cache_usage()
        waiting = self._total_waiting()
        running = self._total_running()
        capacity = self._total_capacity()

        # ── Signal 1: Queue backpressure ─────────────────────────────
        # If more requests are waiting than running, vLLM is saturated.
        # This is the strongest signal — override everything else.
        if waiting > 0 and running > 0 and waiting >= running:
            return self.min_concurrency

        # ── Signal 2: KV cache pressure ──────────────────────────────
        if cache_pct >= 0.80:
            return self.min_concurrency

        # ── Signal 3: Combined score ─────────────────────────────────
        # Blend cache pressure + queue ratio into a 0.0–1.0 pressure score.
        # 0.0 = completely idle, 1.0 = fully saturated.
        cache_pressure = max(0.0, (cache_pct - 0.40) / 0.40)  # 0 at 40%, 1 at 80%
        queue_pressure = 0.0
        if capacity > 0:
            # How full is the server? running/capacity → 0.0–1.0
            queue_pressure = min(1.0, (running + waiting) / capacity)

        # Weighted blend: cache matters more than queue count
        pressure = (cache_pressure * 0.6) + (queue_pressure * 0.4)

        # Linear interpolation: pressure 0 → MAX, pressure 1 → MIN
        span = self.max_concurrency - self.min_concurrency
        limit = self.max_concurrency - int(pressure * span)
        return max(self.min_concurrency, min(self.max_concurrency, limit))

    def _maybe_update_limit(self) -> int:
        """Re-evaluate the limit if enough time has passed."""
        now = time.monotonic()
        if now - self._last_eval >= _REEVALUATE_INTERVAL:
            new_limit = self._compute_limit()
            if new_limit != self._current_limit:
                cache_pct = self._avg_cache_usage() * 100
                waiting = self._total_waiting()
                running = self._total_running()
                logger.info(
                    "[CONCURRENCY] Limit adjusted %d → %d "
                    "(cache=%.1f%%, running=%d, waiting=%d)",
                    self._current_limit,
                    new_limit,
                    cache_pct,
                    running,
                    waiting,
                )
            self._current_limit = new_limit
            self._last_eval = now
        return self._current_limit

    # ── Public API ───────────────────────────────────────────────────

    @property
    def current_limit(self) -> int:
        """Current concurrency limit (may be stale by up to REEVALUATE_INTERVAL)."""
        return self._current_limit

    @property
    def total_active(self) -> int:
        """Total tasks currently in-flight across all labels."""
        return sum(self._label_active.values())

    def status(self) -> dict:
        """Return a snapshot for monitoring dashboards / /monitor/concurrency."""
        endpoints = self._read_endpoints()
        per_endpoint = {}
        for ep in endpoints:
            per_endpoint[ep.name] = {
                "cache_pct": round(ep.cache_usage * 100, 1),
                "requests_running": ep.requests_running,
                "requests_waiting": ep.requests_waiting,
                "max_concurrent": ep.max_concurrent,
            }
        return {
            "current_limit": self._current_limit,
            "min": self.min_concurrency,
            "max": self.max_concurrency,
            "cache_avg_pct": round(self._avg_cache_usage() * 100, 1),
            "total_running_on_vllm": self._total_running(),
            "total_waiting_on_vllm": self._total_waiting(),
            "total_capacity": self._total_capacity(),
            "total_active_tasks": self.total_active,
            "per_label": dict(self._label_active),
            "per_endpoint": per_endpoint,
        }

    async def gather(
        self,
        tasks: list,
        *,
        label: str = "unknown",
        return_exceptions: bool = True,
    ) -> list[Any]:
        """Drop-in replacement for asyncio.gather with adaptive concurrency.

        Args:
            tasks: List of coroutines or awaitables.
            label: Human-readable label for logging (e.g. "data_janitor").
            return_exceptions: If True, exceptions are returned in the
                result list instead of being raised (same as asyncio.gather).

        Returns:
            List of results in the same order as the input tasks.
        """
        if not tasks:
            return []

        limit = self._maybe_update_limit()
        cache_pct = self._avg_cache_usage() * 100
        waiting = self._total_waiting()
        running = self._total_running()
        logger.info(
            "[CONCURRENCY] %s: dispatching %d tasks (limit=%d | "
            "vLLM: cache=%.0f%%, running=%d, waiting=%d)",
            label,
            len(tasks),
            limit,
            cache_pct,
            running,
            waiting,
        )

        results: list[Any] = [None] * len(tasks)
        errors: list[Exception | None] = [None] * len(tasks)

        async def _run(idx: int, coro):
            await self._acquire_slot()
            self._label_active[label] = self._label_active.get(label, 0) + 1
            try:
                results[idx] = await coro
            except Exception as e:
                if return_exceptions:
                    results[idx] = e
                else:
                    errors[idx] = e
            finally:
                self._label_active[label] = max(
                    0, self._label_active.get(label, 1) - 1
                )
                # Clean up zero-count labels
                if self._label_active.get(label, 0) == 0:
                    self._label_active.pop(label, None)
                await self._release_slot()

        await asyncio.gather(
            *[_run(i, t) for i, t in enumerate(tasks)],
            return_exceptions=True,  # Inner gather always catches
        )

        # If not return_exceptions, raise the first error encountered
        if not return_exceptions:
            for e in errors:
                if e is not None:
                    raise e

        return results


# ── Module-level singleton ───────────────────────────────────────────
# Lazy-initialized on first import. All subsystems share this instance.
concurrency_controller = AdaptiveConcurrencyController()

"""
LLM Call Tracker — Centralized monitoring for vLLM usage.

Ring buffer of last 1000 calls in memory + optional PostgreSQL persistence.
Captured: User prompt, model reply, latencies, queue metrics, token usage.
"""

import datetime
import uuid
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LLMCallRecord:
    """Single LLM call audit record."""

    call_id: str
    timestamp: str
    agent_name: str
    ticker: str
    cycle_id: str
    bot_id: str
    model: str

    # Input
    system_prompt: str
    user_prompt: str
    prompt_tokens: int

    # Output
    response_text: str
    completion_tokens: int
    total_tokens: int

    # Performance
    latency_ms: int
    semaphore_active: int  # how many slots were in use when this call started
    semaphore_max: int

    # Routing
    endpoint_name: str = ""  # which vLLM box handled this call

    # Status
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"{status} [{self.agent_name:18s}] {self.ticker:6s} "
            f"→ {self.prompt_tokens:>5d} in / {self.completion_tokens:>4d} out  "
            f"{self.latency_ms:>5d}ms"
        )


class LLMTracker:
    """
    Central audit log for all LLM calls.

    - Ring buffer: last N calls in memory (default 1000)
    - Aggregate stats: total calls, tokens, latency
    - Per-agent breakdown
    - Event listeners for live streaming
    """

    def __init__(self, max_history: int = 1000):
        self._history: deque[LLMCallRecord] = deque(maxlen=max_history)
        self._listeners: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()

        # Aggregate stats
        self.total_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_latency_ms: int = 0
        self.failed_calls: int = 0

        # Per-agent stats
        self._agent_stats: dict[str, dict] = {}

    async def record(
        self,
        agent_name: str,
        ticker: str,
        cycle_id: str,
        bot_id: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: int,
        semaphore_active: int,
        semaphore_max: int,
        success: bool = True,
        error: str = "",
        endpoint_name: str = "",
    ) -> LLMCallRecord:
        """Record a completed LLM call."""
        record = LLMCallRecord(
            call_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            agent_name=agent_name,
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_tokens=prompt_tokens,
            response_text=response_text,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            semaphore_active=semaphore_active,
            semaphore_max=semaphore_max,
            endpoint_name=endpoint_name,
            success=success,
            error=error,
        )

        async with self._lock:
            self._history.append(record)
            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.total_latency_ms += latency_ms
            if not success:
                self.failed_calls += 1

            # Per-agent stats
            if agent_name not in self._agent_stats:
                self._agent_stats[agent_name] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "total_latency_ms": 0,
                    "failures": 0,
                }
            stats = self._agent_stats[agent_name]
            stats["calls"] += 1
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion_tokens
            stats["total_tokens"] += total_tokens
            stats["total_latency_ms"] += latency_ms
            if not success:
                stats["failures"] += 1

        # Notify live listeners
        for q in self._listeners:
            try:
                q.put_nowait(record)
            except asyncio.QueueFull:
                pass  # drop if listener is behind
        return record

    def get_calls(self, limit: int = 50, agent: str | None = None) -> list[dict]:
        """Get recent calls, optionally filtered by agent."""
        calls = list(self._history)
        if agent:
            calls = [c for c in calls if c.agent_name == agent]
        return [c.to_dict() for c in calls[-limit:]]

    def get_call(self, call_id: str) -> dict | None:
        """Get a single call by ID."""
        for c in self._history:
            if c.call_id == call_id:
                return c.to_dict()
        return None

    def get_stats(self) -> dict:
        """Aggregate stats across all calls."""
        avg_latency = (
            self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0
        )
        avg_prompt = (
            self.total_prompt_tokens / self.total_calls if self.total_calls > 0 else 0
        )
        avg_completion = (
            self.total_completion_tokens / self.total_calls
            if self.total_calls > 0
            else 0
        )
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_prompt_tokens": round(avg_prompt, 1),
            "avg_completion_tokens": round(avg_completion, 1),
        }

    def get_recent_tps(self, window_seconds: int = 60) -> float:
        import datetime

        try:
            now = datetime.datetime.now(datetime.UTC)
            tokens = 0
            latency_ms = 0
            for c in reversed(self._history):
                ts = datetime.datetime.fromisoformat(c.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.UTC)
                if (now - ts).total_seconds() > window_seconds:
                    break
                tokens += c.total_tokens
                latency_ms += c.latency_ms
            if latency_ms == 0:
                return 0.0
            return round(tokens / (latency_ms / 1000.0), 1)
        except Exception:
            return 0.0

    def get_recent_tps_by_endpoint(self, window_seconds: int = 60) -> dict[str, float]:
        """Return per-endpoint TPS for calls within the last `window_seconds`."""
        import datetime

        try:
            now = datetime.datetime.now(datetime.UTC)
            buckets: dict[str, dict] = {}  # endpoint → {tokens, latency_ms}
            for c in reversed(self._history):
                ts = datetime.datetime.fromisoformat(c.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.UTC)
                if (now - ts).total_seconds() > window_seconds:
                    break
                ep = c.endpoint_name or "unknown"
                if ep not in buckets:
                    buckets[ep] = {"tokens": 0, "latency_ms": 0}
                buckets[ep]["tokens"] += c.total_tokens
                buckets[ep]["latency_ms"] += c.latency_ms
            result = {}
            for ep, data in buckets.items():
                if data["latency_ms"] > 0:
                    result[ep] = round(
                        data["tokens"] / (data["latency_ms"] / 1000.0), 1
                    )
                else:
                    result[ep] = 0.0
            return result
        except Exception:
            return {}

    def get_agent_stats(self) -> dict:
        """Per-agent breakdown."""
        result = {}
        for agent, stats in self._agent_stats.items():
            calls = stats["calls"]
            result[agent] = {
                **stats,
                "avg_latency_ms": round(stats["total_latency_ms"] / calls, 1)
                if calls > 0
                else 0,
                "avg_tokens_per_call": round(stats["total_tokens"] / calls, 1)
                if calls > 0
                else 0,
            }
        return result

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to live call events. Returns a queue that receives LLMCallRecord objects."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._listeners.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a live listener."""
        if q in self._listeners:
            self._listeners.remove(q)


# Singleton
tracker = LLMTracker()

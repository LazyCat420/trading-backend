"""
Metrics Collector — Polls Jetson /metrics and /load endpoints.

Parses Prometheus text format into structured data.
Stores time-series snapshots for trend visualization.
"""

import asyncio
import datetime
import re
from collections import deque
from dataclasses import dataclass, asdict

import httpx
from app.config import settings


@dataclass
class MetricsSnapshot:
    """Single point-in-time metrics snapshot from the Jetson."""

    timestamp: str
    server_load: int

    # Request stats
    num_requests_running: float
    num_requests_waiting: float

    # Throughput
    avg_generation_throughput: float  # tok/s

    # KV Cache
    gpu_cache_usage_pct: float

    # Latency percentiles (seconds)
    e2e_latency_p50: float
    e2e_latency_p95: float
    e2e_latency_p99: float

    # Token counts from vLLM
    prompt_tokens_total: float
    generation_tokens_total: float

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsCollector:
    """
    Polls the Jetson vLLM /metrics endpoint and parses Prometheus output.
    Stores snapshots in a ring buffer for time-series queries.
    """

    # Prometheus metric patterns we care about
    GAUGE_PATTERNS = {
        "vllm:num_requests_running": "num_requests_running",
        "vllm:num_requests_waiting": "num_requests_waiting",
        "vllm:avg_generation_throughput_toks_per_s": "avg_generation_throughput",
        "vllm:gpu_cache_usage_perc": "gpu_cache_usage_pct",
        "vllm:prompt_tokens_total": "prompt_tokens_total",
        "vllm:generation_tokens_total": "generation_tokens_total",
    }

    # Histogram quantile patterns
    HISTOGRAM_PATTERNS = {
        ("vllm:e2e_request_latency_seconds", "0.5"): "e2e_latency_p50",
        ("vllm:e2e_request_latency_seconds", "0.95"): "e2e_latency_p95",
        ("vllm:e2e_request_latency_seconds", "0.99"): "e2e_latency_p99",
    }

    def __init__(self, max_history: int = 360):
        # 360 snapshots × 10s interval = 1 hour of history
        self._history: deque[MetricsSnapshot] = deque(maxlen=max_history)
        self._base_url = settings.JETSON_VLLM_URL
        self._polling = False
        self._poll_task: asyncio.Task | None = None

    def _parse_prometheus(self, text: str) -> dict[str, float]:
        """Parse Prometheus text format into a flat dict of metric_name → value."""
        result = {}
        for line in text.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            # Match: metric_name{labels} value
            # or:    metric_name value
            match = re.match(
                r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?([^}]*)\}?\s+([\d.eE+-]+|NaN|Inf|-Inf)$",
                line,
            )
            if match:
                name = match.group(1)
                labels = match.group(2)
                try:
                    value = float(match.group(3))
                except ValueError:
                    continue

                # For histograms with quantile labels
                quantile_match = re.search(r'quantile="([^"]+)"', labels)
                if quantile_match:
                    key = (name, quantile_match.group(1))
                    if key in self.HISTOGRAM_PATTERNS:
                        result[self.HISTOGRAM_PATTERNS[key]] = value
                elif name in self.GAUGE_PATTERNS:
                    result[self.GAUGE_PATTERNS[name]] = value

        return result

    async def collect_once(self) -> MetricsSnapshot | None:
        """Collect a single metrics snapshot from the Jetson."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get /metrics (Prometheus format)
                metrics_resp = await client.get(f"{self._base_url}/metrics")
                parsed = self._parse_prometheus(metrics_resp.text)

                # Get /load
                load_resp = await client.get(f"{self._base_url}/load")
                load_data = load_resp.json()

                snapshot = MetricsSnapshot(
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                    server_load=load_data.get("server_load", 0),
                    num_requests_running=parsed.get("num_requests_running", 0),
                    num_requests_waiting=parsed.get("num_requests_waiting", 0),
                    avg_generation_throughput=parsed.get(
                        "avg_generation_throughput", 0
                    ),
                    gpu_cache_usage_pct=parsed.get("gpu_cache_usage_pct", 0),
                    e2e_latency_p50=parsed.get("e2e_latency_p50", 0),
                    e2e_latency_p95=parsed.get("e2e_latency_p95", 0),
                    e2e_latency_p99=parsed.get("e2e_latency_p99", 0),
                    prompt_tokens_total=parsed.get("prompt_tokens_total", 0),
                    generation_tokens_total=parsed.get("generation_tokens_total", 0),
                )
                self._history.append(snapshot)
                return snapshot

        except Exception as e:
            print(f"[MetricsCollector] Failed to collect: {e}")
            return None

    async def start_polling(self, interval_seconds: int = 10):
        """Start background polling loop."""
        if self._polling:
            return
        self._polling = True
        self._poll_task = asyncio.create_task(self._poll_loop(interval_seconds))

    async def _poll_loop(self, interval: int):
        """Background loop that collects metrics periodically."""
        while self._polling:
            await self.collect_once()
            await asyncio.sleep(interval)

    async def stop_polling(self):
        """Stop background polling."""
        self._polling = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    def get_latest(self) -> dict | None:
        """Get the most recent snapshot."""
        if self._history:
            return self._history[-1].to_dict()
        return None

    def get_history(self, limit: int = 60) -> list[dict]:
        """Get recent snapshots for time-series graphing."""
        snapshots = list(self._history)[-limit:]
        return [s.to_dict() for s in snapshots]


# Singleton
metrics = MetricsCollector()

"""
vLLM Client — routes ALL LLM calls through Prism AI Gateway's /agent endpoint.

Priority-queue dispatcher: keeps all VLLM_MAX_CONCURRENT slots filled.
User-interactive requests (chat, collaboration) get HIGH priority and
jump the queue ahead of pipeline work.

Architecture:
  Trading Bot → Prism /agent?stream=false → AgenticLoopService → vLLM provider → vLLM server
  Prism handles: agentic loop, tool execution, request logging, conversations, sessions, metrics

  When Prism is unavailable, falls back to direct vLLM calls with offline shadow-logging.

Multi-Endpoint Architecture:
  Each vLLM endpoint (Jetson, DGX Spark 1, DGX Spark 2, ...) is represented
  as a VLLMEndpoint with its own queue, concurrency slots, and auto-discovered
  model.  Models are fetched via the vLLM /v1/models API — no hardcoded names.

Hardware Tier Routing (auto-assigned, not user-configurable):
  Tier 0 = Jetson Orin AGX 64GB  (lightweight: data collection, curation, agents)
  Tier 1 = DGX Spark(s)     (heavy: deep analysis, RLM, debate, decisions)
  All endpoints participate as equal workers in the agentic pipeline.
  Tasks are routed by hardware capability, not by manually assigned "roles".

Priority levels:
  HIGH   = user chat, collaboration Q&A  (served first)
  NORMAL = pipeline agents, RLM, debate  (standard FIFO)
  LOW    = LLM curation, background      (best-effort)

Every agent imports `llm` from here. No other file makes HTTP calls to vLLM.
Monitoring: every chat() call is auto-logged to the LLM tracker (PostgreSQL)
AND recorded by Prism (MongoDB) automatically through the /agent gateway.
"""

import asyncio
import logging
import time
import uuid
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings
from app.config.config_models import get_model_limits
from app.config.context_budget import register_model_context
from app.db.connection import get_db
from app.monitoring.llm_tracker import tracker
from app.services.prism_client import PrismClient
from app.utils.text_utils import strip_think_tags

logger = logging.getLogger(__name__)


def _is_qwen_model(model_id: str) -> bool:
    """Check if a model ID belongs to the Qwen family.

    Only Qwen models support chat_template_kwargs with enable_thinking.
    Sending this parameter to non-Qwen models (e.g., NVIDIA Nemotron)
    causes 500 errors.
    """
    if not model_id:
        return False
    lower = model_id.lower()
    return "qwen" in lower


# ── Priority Levels ────────────────────────────────────────────────────
class Priority(IntEnum):
    """Lower number = higher priority."""

    HIGH = 0  # user-interactive (chat, collaboration)
    NORMAL = 1  # pipeline (agents, RLM, debate)
    LOW = 2  # background (curation, maintenance)


@dataclass(order=True)
class QueueItem:
    """An item in the priority queue.

    Ordering: (priority, seq) ensures FIFO within the same priority level.
    The payload/future/metadata are excluded from comparison.
    """

    priority: int
    seq: int
    future: asyncio.Future = field(compare=False)
    payload: dict = field(compare=False)
    metadata: dict = field(compare=False)


# ── Valid Roles (internal routing tiers — NOT user-assignable) ───────────
VALID_ROLES = ("collector", "analyst", "trader", "training")


# ── Role-Based Routing Taxonomy ─────────────────────────────────────────
# Maps agent names to their required hardware role/tier.
# 'collector' = Jetson (35B), 'analyst' / 'trader' = DGX Spark (120B)
AGENT_ROLE_ROUTING = {
    # Lightweight tasks -> Jetson
    "hermes_research": "collector",
    "summarizer": "collector",
    "data_janitor": "collector",
    "data_curator": "collector",
    "retriever": "collector",
    "user_chat": "collector",
    "collector": "collector",
    
    # Complex reasoning -> DGX Spark
    "technical": "analyst",
    "fundamental": "analyst",
    "sentiment": "analyst",
    "fund_flow": "analyst",
    "risk": "analyst",
    "comparative": "analyst",
    "trading_phase": "trader",
    "decision_engine": "trader",
    "deepeval_judge": "analyst",
    "strategy_evaluator": "analyst",
    "judge_evaluator": "analyst",
    "debate": "analyst",
}


# ── Smart Queue Constants ──────────────────────────────────────────────
# Max queue depth per endpoint = max_concurrent * this multiplier.
# Beyond this, requests overflow to the next-best endpoint.
MAX_QUEUE_DEPTH_MULTIPLIER = 3

# Seconds to penalize an endpoint after a timeout (makes it less preferred)
TIMEOUT_PENALTY_SECONDS = 30

# Seconds to disable an endpoint after circuit breaker trips
CIRCUIT_BREAKER_COOLDOWN = 60


# ── VLLMEndpoint ───────────────────────────────────────────────────────
@dataclass
class VLLMEndpoint:
    """Represents a single vLLM server instance.

    Each endpoint manages its own priority queue, concurrency semaphores,
    and dispatcher task independently of others.
    """

    name: str  # "jetson", "dgx_spark", "dgx_spark_2"
    url: str  # e.g. "http://10.0.0.30:8000"
    role: str  # "collector", "analyst", or "training"
    max_concurrent: int  # max parallel requests
    purpose: str  # human-readable description

    # Toggle: if False, endpoint is excluded from pipeline routing
    enabled: bool = True
    # True when the system auto-disabled this endpoint because it was
    # unreachable during model discovery. Distinguished from manual disable
    # so the re-discovery task knows it can re-enable it automatically.
    auto_disabled: bool = field(default=False, repr=False)
    # True when the vLLM server is reachable but has no model loaded yet
    # (e.g., Jetson Orin AGX takes 10+ min to load a large model).
    # Distinguished from offline (unreachable) so the system can give
    # informative errors and poll faster for model readiness.
    loading: bool = field(default=False, repr=False)

    # Auto-populated by discover_roles()
    model: str | None = field(default=None, repr=False)
    max_model_len: int = field(default=0, repr=False)  # Context window from vLLM

    # Runtime state (not part of repr/compare)
    active_count: int = field(default=0, repr=False)
    cache_usage: float = field(default=0.0, repr=False)
    # vLLM server-side metrics (polled from /metrics endpoint)
    requests_running: int = field(default=0, repr=False)   # vllm:num_requests_running
    requests_waiting: int = field(default=0, repr=False)   # vllm:num_requests_waiting
    queue: asyncio.PriorityQueue = field(default=None, repr=False, compare=False)
    slots: asyncio.Semaphore = field(default=None, repr=False, compare=False)
    pipeline_slots: asyncio.Semaphore = field(default=None, repr=False, compare=False)
    dispatcher_task: asyncio.Task | None = field(
        default=None, repr=False, compare=False
    )
    metrics_task: asyncio.Task | None = field(
        default=None, repr=False, compare=False
    )
    # Timeout penalty: when an endpoint times out, penalize it temporarily
    timeout_penalty_until: float = field(default=0.0, repr=False)
    # Batch dispatch: max items pulled from queue per batch
    batch_size: int = field(default=24, repr=False)
    # Circuit breaker: track consecutive failed batches
    consecutive_batch_failures: int = field(default=0, repr=False)
    circuit_open_until: float = field(default=0.0, repr=False)

    def init_concurrency(self, reserved_high: int = 1):
        """Initialize queue and semaphores. Safe to call at import time."""
        if not hasattr(self, "queue") or self.queue is None:
            self.queue = asyncio.PriorityQueue()
        if not hasattr(self, "slots") or self.slots is None:
            self.slots = asyncio.Semaphore(self.max_concurrent)
        pipe_max = max(1, self.max_concurrent - reserved_high)
        if not hasattr(self, "pipeline_slots") or self.pipeline_slots is None:
            self.pipeline_slots = asyncio.Semaphore(pipe_max)

    @property
    def load_score(self) -> float:
        """Combined load score: active + queued + penalty + circuit breaker.

        Lower score = more available capacity.
        Penalty adds a large constant when the endpoint recently timed out.
        Circuit breaker adds infinity when the endpoint is tripped.
        """
        # Circuit breaker open → infinite score (never route here)
        if self.circuit_open_until > time.monotonic():
            return float('inf')
        qs = self.queue.qsize() if self.queue else 0
        score = float(self.active_count + qs)
        # Add penalty if endpoint recently timed out
        if self.timeout_penalty_until > time.monotonic():
            score += self.max_concurrent  # effectively deprioritize
        return score

    @property
    def is_overloaded(self) -> bool:
        """True if queue depth exceeds safe threshold."""
        qs = self.queue.qsize() if self.queue else 0
        return qs > self.max_concurrent * MAX_QUEUE_DEPTH_MULTIPLIER


_last_failure_times = {}

def _record_endpoint_failure(ep: VLLMEndpoint, cb_threshold: int, reason: str):
    now = time.monotonic()
    if ep.circuit_open_until > now:
        return
        
    last_t = _last_failure_times.get(ep.name, 0.0)
    if now - last_t < 0.2:
        return
        
    _last_failure_times[ep.name] = now
    ep.consecutive_batch_failures += 1
    if ep.consecutive_batch_failures >= cb_threshold:
        ep.circuit_open_until = now + CIRCUIT_BREAKER_COOLDOWN
        logger.error(
            "[BATCH] 🔴 %s CIRCUIT BREAKER OPEN after %s — disabling for %ds",
            ep.name,
            reason,
            CIRCUIT_BREAKER_COOLDOWN,
        )
        ep.consecutive_batch_failures = 0


class VLLMClient:
    # Reserve 1 slot for HIGH priority (user chat) — pipeline can use max_concurrent-1
    RESERVED_HIGH_SLOTS = 1

    def __init__(self):
        # ── Build endpoint registry ──
        self._endpoints: dict[str, VLLMEndpoint] = {}

        if settings.JETSON_VLLM_URL:
            ep = VLLMEndpoint(
                name="jetson",
                url=settings.JETSON_VLLM_URL,
                role="collector",
                max_concurrent=settings.JETSON_MAX_CONCURRENT,
                purpose="Data collection, summarization, LLM curation, agents",
                batch_size=settings.JETSON_BATCH_SIZE,
            )
            ep.init_concurrency(self.RESERVED_HIGH_SLOTS)
            self._endpoints["jetson"] = ep

        if settings.DGX_SPARK_VLLM_URL:
            ep = VLLMEndpoint(
                name="dgx_spark",
                url=settings.DGX_SPARK_VLLM_URL,
                role="trader",
                max_concurrent=settings.DGX_MAX_CONCURRENT,
                purpose="Final trading decisions — uses most capable model",
                batch_size=settings.DGX_BATCH_SIZE,
            )
            ep.init_concurrency(self.RESERVED_HIGH_SLOTS)
            self._endpoints["dgx_spark"] = ep

        if settings.DGX_SPARK_2_VLLM_URL:
            ep = VLLMEndpoint(
                name="dgx_spark_2",
                url=settings.DGX_SPARK_2_VLLM_URL,
                role="analyst",
                max_concurrent=settings.DGX_SPARK_2_MAX_CONCURRENT,
                purpose="Deep analysis, RLM decisions, debate engine",
                batch_size=settings.DGX_SPARK_2_BATCH_SIZE,
            )
            ep.init_concurrency(self.RESERVED_HIGH_SLOTS)
            self._endpoints["dgx_spark_2"] = ep

        # ── Prism gateway ──
        self.prism_client = PrismClient()

        # Active/default model — auto-filled by discover_roles()
        self.model = settings.ACTIVE_MODEL or ""

        self._seq = 0  # monotonic counter for FIFO within priority

        # Reuse one client for connection pooling
        self._client: httpx.AsyncClient | None = None
        # Model → endpoint name cache (refreshed on cache miss)
        self._model_endpoint_cache: dict[str, str] = {}
        # Background rediscovery task handle
        self._rediscovery_task: asyncio.Task | None = None
        # Role discovery flag
        self._roles_discovered: bool = False
        # Active background tasks representing running requests
        self._active_tasks: set[asyncio.Task] = set()

    # ── Endpoint configuration ─────────────────────────────────────────

    def _active_endpoints(self) -> list[VLLMEndpoint]:
        """Return all enabled, non-training endpoints.

        All endpoints (including trader) participate in general load
        balancing since they share the same model. Trader-specific
        routing is still available via get_trader_model().
        """
        return [
            ep
            for ep in self._endpoints.values()
            if ep.enabled and ep.role != "training"
        ]

    def _pick_best_endpoint(self, requested_model: str | None = None, agent_name: str | None = None) -> VLLMEndpoint:
        """Capacity-aware endpoint selection with cross-role fallback.

        Picks the endpoint with the lowest combined load (active + queued),
        accounting for timeout penalties and queue overflow.
        
        If requested_model is provided, strictly filters candidates to only 
        those endpoints hosting that specific model.
        
        If agent_name is provided, routes to the appropriate hardware tier 
        (Jetson vs DGX Spark) based on task complexity.
        
        CROSS-ROLE FALLBACK: If the preferred tier has no ready endpoints
        (e.g., Jetson model still loading), routes to ANY available endpoint
        regardless of role assignment. This prevents timeouts when one box
        is starting up.
        """
        # Step 1: All endpoints with models loaded
        all_ready = [
            ep
            for ep in self._endpoints.values()
            if ep.enabled and ep.role != "training" and ep.model
        ]
        candidates = list(all_ready)
        
        # Strictly filter by model first if requested
        model_filtered = False
        if requested_model:
            model_candidates = [ep for ep in candidates if ep.model == requested_model]
            if model_candidates:
                candidates = model_candidates
                model_filtered = True
            else:
                logger.warning("[VLLM] Requested model '%s' not found on any active endpoint. Falling back.", requested_model)
        
        # Role-based routing: filter by required role if defined in taxonomy
        required_role = None
        if agent_name:
            for key, role in AGENT_ROLE_ROUTING.items():
                if agent_name.startswith(key) or f"_{key}" in agent_name:
                    required_role = role
                    break
            
            if required_role:
                # If analyst or trader, we can use any DGX (they both run 120B). 
                # If collector, we use Jetson.
                if required_role in ("analyst", "trader"):
                    acceptable_roles = ("analyst", "trader")
                else:
                    acceptable_roles = ("collector",)
                
                role_candidates = [ep for ep in candidates if ep.role in acceptable_roles]
                if role_candidates:
                    candidates = role_candidates
                else:
                    # CROSS-ROLE FALLBACK: preferred tier has no ready endpoints.
                    # Route to ANY available endpoint with a model loaded.
                    if model_filtered:
                        # Do not fallback to all active models since we filtered by model
                        pass
                    elif all_ready:
                        fallback_names = ', '.join(ep.name for ep in all_ready)
                        logger.warning(
                            "[VLLM] 🔀 No %s-tier endpoints have models loaded. "
                            "Cross-routing '%s' to available endpoints: [%s]",
                            required_role, agent_name, fallback_names,
                        )
                        candidates = all_ready
                    else:
                        logger.warning(
                            "[VLLM] No active endpoint found for role tier '%s' and no fallbacks. ",
                            required_role,
                        )

        if not candidates:
            # Fallback: any enabled endpoint with a model
            candidates = [ep for ep in self._endpoints.values() if ep.enabled and ep.model]
        
        if not candidates:
            # Check if any endpoints are in loading state
            loading_eps = [
                ep for ep in self._endpoints.values()
                if ep.loading
            ]
            if loading_eps:
                loading_names = ', '.join(ep.name for ep in loading_eps)
                raise RuntimeError(
                    f"No models ready — {len(loading_eps)} endpoint(s) still loading: "
                    f"[{loading_names}]. Requests will succeed once model loading completes."
                )
            raise RuntimeError("No vLLM endpoints available")

        # Prefer non-overloaded endpoints
        non_overloaded = [ep for ep in candidates if not ep.is_overloaded]
        pool = non_overloaded if non_overloaded else candidates

        best = min(pool, key=lambda ep: ep.load_score)
        # Log cross-routing at INFO level so it's visible in docker logs
        if required_role and best.role not in (required_role,):
            # Check if the original role group was also acceptable
            if required_role in ("analyst", "trader") and best.role in ("analyst", "trader"):
                pass  # Same tier, not a cross-route
            else:
                logger.info(
                    "[QUEUE] 🔀 Cross-route: %s (needs %s-tier) → %s (%s-tier, model=%s)",
                    agent_name, required_role, best.name, best.role, best.model,
                )
        logger.debug(
            "[QUEUE] Smart route → %s (score=%.0f, active=%d/%d, queued=%d)%s",
            best.name,
            best.load_score,
            best.active_count,
            best.max_concurrent,
            best.queue.qsize() if best.queue else 0,
            " [PENALTY]" if best.timeout_penalty_until > time.monotonic() else "",
        )
        return best

    def configure_endpoint(
        self, name: str, *, enabled: bool | None = None, role: str | None = None
    ) -> dict:
        """Toggle an endpoint on/off and/or change its role.

        Args:
            name: Endpoint name (e.g. "jetson", "dgx_spark", "dgx_spark_2")
            enabled: If provided, set the endpoint's enabled state
            role: If provided, set the endpoint's role ("collector", "analyst", "training")

        Returns dict with the updated endpoint info.
        """
        ep = self._endpoints.get(name)
        if not ep:
            raise ValueError(f"Unknown endpoint: {name}")

        if enabled is not None:
            ep.enabled = enabled
            # Manual toggle clears auto_disabled so rediscovery doesn't
            # override the user's explicit intent.
            ep.auto_disabled = False
            logger.info(
                "[VLLM] Endpoint %s %s (manual)", name, "ENABLED" if enabled else "DISABLED"
            )

        if role is not None:
            if role not in VALID_ROLES:
                raise ValueError(
                    f"Invalid role '{role}'. Must be one of: {VALID_ROLES}"
                )
            old_role = ep.role
            ep.role = role
            # Update purpose description based on role
            if role == "collector":
                ep.purpose = "Data collection, summarization, LLM curation, agents"
            elif role == "analyst":
                ep.purpose = "Deep analysis, RLM decisions, debate engine"
            elif role == "trader":
                ep.purpose = "Final trading decisions — uses most capable model"
            elif role == "training":
                ep.purpose = "Model fine-tuning with Unsloth (placeholder)"
            logger.info(
                "[VLLM] Endpoint %s role changed: %s → %s", name, old_role, role
            )

        # Clear model resolution cache since roles changed
        self._model_endpoint_cache.clear()

        return {
            "name": ep.name,
            "url": ep.url,
            "role": ep.role,
            "enabled": ep.enabled,
            "model": ep.model,
            "purpose": ep.purpose,
        }

    # ── Backward-compat properties ─────────────────────────────────────

    @property
    def base_url(self) -> str:
        """Jetson URL (backward compat)."""
        ep = self._endpoints.get("jetson")
        return ep.url if ep else ""

    @property
    def _dgx_url(self) -> str:
        """First DGX Spark URL (backward compat)."""
        ep = self._endpoints.get("dgx_spark")
        return ep.url if ep else ""

    @property
    def _jetson_active_count(self) -> int:
        ep = self._endpoints.get("jetson")
        return ep.active_count if ep else 0

    @property
    def _dgx_active_count(self) -> int:
        total = 0
        for name, ep in self._endpoints.items():
            if name.startswith("dgx"):
                total += ep.active_count
        return total

    @property
    def _jetson_max(self) -> int:
        ep = self._endpoints.get("jetson")
        return ep.max_concurrent if ep else 0

    @property
    def _dgx_max(self) -> int:
        total = 0
        for name, ep in self._endpoints.items():
            if name.startswith("dgx"):
                total += ep.max_concurrent
        return total

    # ── HTTP client ────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init a persistent async client for connection reuse."""
        if self._client is not None and not self._client.is_closed:
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                self._client = None
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=240.0)
        return self._client

    # ── Dispatchers ────────────────────────────────────────────────────

    def _ensure_dispatcher(self):
        """Start the dispatcher loop for each enabled endpoint if not already running."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                for ep in self._endpoints.values():
                    if not ep.enabled:
                        continue
                    if ep.dispatcher_task is None or ep.dispatcher_task.done():
                        ep.dispatcher_task = loop.create_task(self._dispatch_loop(ep))
                    if ep.metrics_task is None or ep.metrics_task.done():
                        ep.metrics_task = loop.create_task(self._poll_metrics_loop(ep))
                # Start background rediscovery (re-probes offline endpoints every 60s)
                if self._rediscovery_task is None or self._rediscovery_task.done():
                    self._rediscovery_task = loop.create_task(self._rediscovery_loop())
        except RuntimeError:
            pass

    async def _poll_metrics_loop(self, ep: VLLMEndpoint):
        """Background task to poll VLLM /metrics and update live hardware state.

        Reads three critical gauges from the vLLM Prometheus endpoint:
          - vllm:gpu_cache_usage_perc  → ep.cache_usage   (0.0–1.0)
          - vllm:num_requests_running  → ep.requests_running
          - vllm:num_requests_waiting  → ep.requests_waiting

        These values feed into the AdaptiveConcurrencyController to
        dynamically throttle caller-side batch sizes based on how much
        room is actually left on the vLLM docker.
        """
        # Map vLLM metric names to (attribute, converter)
        _METRIC_MAP = {
            "vllm_gpu_cache_usage_perc": ("cache_usage", float),
            "vllm_num_requests_running": ("requests_running", lambda v: int(float(v))),
            "vllm_num_requests_waiting": ("requests_waiting", lambda v: int(float(v))),
        }
        while True:
            try:
                client = await self._get_client()
                r = await client.get(f"{ep.url}/metrics", timeout=5.0)
                if r.status_code == 200:
                    for line in r.text.splitlines():
                        if line.startswith("#") or not line.strip():
                            continue
                        for metric_prefix, (attr, conv) in _METRIC_MAP.items():
                            if line.startswith(metric_prefix):
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        setattr(ep, attr, conv(parts[-1]))
                                    except (ValueError, TypeError):
                                        pass
                                break
            except Exception as e:
                logger.debug("[QUEUE] Failed to fetch metrics from %s: %s", ep.name, e)
            await asyncio.sleep(30.0)  # was 5s — reduced to avoid spamming endpoints

    async def _dispatch_loop(self, ep: VLLMEndpoint):
        """Background task: continuously drain queue, executing tasks concurrently
        using a semaphore to limit in-flight requests. This eliminates the head-of-line
        blocking of the legacy batching design where a single slow request stalled
        the entire batch dispatcher.
        """
        batch_timeout = float(settings.VLLM_FUTURE_TIMEOUT)
        cb_threshold = settings.BATCH_CIRCUIT_BREAKER_THRESHOLD

        while True:
            try:
                # ── Pipeline stop check ────────────────────────────────
                try:
                    from app.pipeline.orchestration.cycle_control import cycle_control
                    if cycle_control.is_stopped:
                        # Drain and cancel all pending NORMAL and LOW priority futures in the queue
                        # Keep HIGH priority items in the queue
                        drained = 0
                        temp_list = []
                        while not ep.queue.empty():
                            try:
                                item = ep.queue.get_nowait()
                                if item.priority > Priority.HIGH:
                                    if not item.future.done():
                                        item.future.cancel()
                                    ep.queue.task_done()
                                    drained += 1
                                else:
                                    temp_list.append(item)
                            except asyncio.QueueEmpty:
                                break
                        
                        # Put HIGH priority items back into the queue
                        for item in temp_list:
                            await ep.queue.put(item)

                        if drained:
                            logger.info(
                                "[QUEUE] %s drained %d items on pipeline stop",
                                ep.name, drained,
                            )
                        
                        # If queue is empty of HIGH priority items, wait and continue
                        if ep.queue.empty():
                            # Wait briefly before checking again (don't spin)
                            await asyncio.sleep(0.5)
                            continue
                except ImportError:
                    pass

                # ── Circuit breaker check ──────────────────────────────
                if ep.circuit_open_until > time.monotonic():
                    remaining = ep.circuit_open_until - time.monotonic()
                    logger.warning(
                        "[QUEUE] ⚡ %s circuit breaker OPEN — paused for %.0fs",
                        ep.name,
                        remaining,
                    )
                    await asyncio.sleep(min(remaining, 10.0))
                    continue

                # ── Smart Queue memory protection ──────────────────────
                while ep.cache_usage >= 0.90:
                    logger.warning(
                        "[QUEUE] 🚨 %s KV cache at %.1f%% (>90%%). Smart Queue pausing dispatch...",
                        ep.name,
                        ep.cache_usage * 100,
                    )
                    await asyncio.sleep(2.0)

                # Acquire the concurrency slot semaphore
                await ep.slots.acquire()

                # Get the highest-priority item from queue
                try:
                    item = await ep.queue.get()
                except Exception:
                    ep.slots.release()
                    raise

                # Check stop again after waking from queue.get()
                try:
                    from app.pipeline.orchestration.cycle_control import cycle_control
                    if cycle_control.is_stopped and item.priority > Priority.HIGH:
                        if not item.future.done():
                            item.future.cancel()
                        ep.queue.task_done()
                        ep.slots.release()
                        continue
                except ImportError:
                    pass

                if item.future.cancelled():
                    ep.queue.task_done()
                    ep.slots.release()
                    continue

                # Run execute_item in background, and release semaphore when finished
                async def run_and_release(item=item):
                    try:
                        # execute individual item with BATCH_TIMEOUT
                        await asyncio.wait_for(
                            self._execute_item(item, ep, release_pipeline=item.priority > Priority.HIGH),
                            timeout=batch_timeout
                        )
                        if item.future.done() and item.future.exception() is not None:
                            raise item.future.exception()
                        ep.consecutive_batch_failures = 0
                    except asyncio.TimeoutError:
                        logger.error(
                            "[VLLM] ⏰ Request TIMEOUT for agent=%s ticker=%s",
                            item.metadata.get("agent_name"),
                            item.metadata.get("ticker"),
                        )
                        if not item.future.done():
                            item.future.set_exception(
                                asyncio.TimeoutError(
                                    f"vLLM request timeout ({batch_timeout}s) for {item.metadata.get('agent_name')}"
                                )
                            )
                        ep.timeout_penalty_until = time.monotonic() + TIMEOUT_PENALTY_SECONDS
                        _record_endpoint_failure(ep, cb_threshold, "request timeout")
                    except asyncio.CancelledError:
                        logger.warning(
                            "[VLLM] 🛑 Active request cancelled for agent=%s ticker=%s",
                            item.metadata.get("agent_name"),
                            item.metadata.get("ticker"),
                        )
                        if not item.future.done():
                            item.future.cancel()
                        raise
                    except Exception as err:
                        logger.exception("[VLLM] Request failed for agent=%s ticker=%s: %s", item.metadata.get("agent_name"), item.metadata.get("ticker"), err)
                        _record_endpoint_failure(ep, cb_threshold, "error")
                    finally:
                        ep.queue.task_done()
                        ep.slots.release()

                task = asyncio.create_task(run_and_release())
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[QUEUE] %s dispatcher error: %s", ep.name, e)
                await asyncio.sleep(1.0)  # Prevent tight error loop

    async def _execute_item(
        self, item: QueueItem, ep: VLLMEndpoint, release_pipeline: bool = False
    ):
        """Execute a single queued request through Prism gateway."""
        ep.active_count += 1
        semaphore_active = ep.active_count
        semaphore_max = ep.max_concurrent

        is_high = item.priority <= Priority.HIGH
        if is_high:
            logger.info(
                "[QUEUE] ⚡ HIGH priority request started on %s (active=%d/%d)",
                ep.name,
                semaphore_active,
                semaphore_max,
            )
        start = time.monotonic()
        meta = item.metadata

        # Per-box telemetry: compute queue wait time
        enqueue_time = meta.get("_enqueue_time", start)
        queue_wait_ms = int((start - enqueue_time) * 1000)
        meta["_endpoint_name"] = ep.name
        meta["_queue_wait_ms"] = queue_wait_ms

        try:
            client = await self._get_client()

            # ── Model Name Sync ──
            # ALWAYS ensure the payload uses the endpoint's dynamically discovered model.
            # This prevents 500 errors when hardcoded models (from settings/overrides)
            # don't match the actual model loaded on the target hardware (e.g., Prism Gateway).
            if ep.model:
                item.payload["model"] = ep.model
                
            # ── Routing: Prism /agent vs direct vLLM ──
            # Controlled by PRISM_AGENT_ROUTING toggle:
            #   True  = route ALL endpoints through Prism /agent
            #           (agentic loop, tool execution, native logging)
            #   False = direct vLLM + offline shadow-log to Prism
            # NOTE: Jetson was previously excluded from Prism routing.
            # As of Phase 3 (Unified Telemetry), ALL endpoints route
            # through Prism so that every request is tracked and visible.
            use_prism_agent = (
                self.prism_client.enabled
                and settings.PRISM_AGENT_ROUTING
            )
            prism_routed = False

            if use_prism_agent:
                prism_is_healthy = await self.prism_client.check_health()
                if prism_is_healthy:
                    logger.info(
                        "[PRISM] Agent call for %s (via %s) | agent=%s",
                        meta.get("agent_name", "unknown"),
                        ep.name,
                        self.prism_client.agent,
                    )
                    content, total_tokens, elapsed_ms = await self._call_prism_agent(
                        client, item.payload, meta, start
                    )
                    prism_routed = True
                else:
                    logger.warning(
                        "[PRISM] ⚠️ Prism unhealthy — falling back to direct vLLM for %s",
                        meta.get("agent_name", "unknown"),
                    )

            if not prism_routed:
                logger.info(
                    "[VLLM] Direct call for %s (via %s) model=%s | Global Active: %d",
                    meta.get("agent_name", "unknown"),
                    ep.name,
                    item.payload.get("model", "?"),
                    self._active_slots,
                )
                content, total_tokens, elapsed_ms = await self._call_vllm_direct(
                    client, item.payload, meta, start, ep=ep
                )

            usage_data = meta.get("_usage", {})
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)

            # Record to local PostgreSQL tracker
            await tracker.record(
                agent_name=meta.get("agent_name", "unknown"),
                ticker=meta.get("ticker", ""),
                cycle_id=meta.get("cycle_id", ""),
                bot_id=meta.get("bot_id", ""),
                model=item.payload.get("model", self.model),
                system_prompt=meta.get("system_prompt", ""),
                user_prompt=meta.get("user_prompt", ""),
                response_text=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=elapsed_ms,
                semaphore_active=semaphore_active,
                semaphore_max=semaphore_max,
                endpoint_name=ep.name,
            )



            # Record Graph Attention from <think> content
            think_text = meta.get("_think_content", "")
            if think_text and hasattr(self, "_cached_nodes") is False:
                # Cache nodes once (or occasionally)
                with get_db() as db:
                    try:
                        rows = db.execute(
                            "SELECT id FROM ontology_nodes WHERE node_type NOT IN ('Asset', 'Sector')"
                        ).fetchall()
                        self._cached_nodes = {row[0].lower(): row[0] for row in rows}
                    except Exception as e:
                        logger.error(
                            f"[VLLMClient] Failed to cache nodes for attention map: {e}"
                        )
                        self._cached_nodes = {}

            if think_text and getattr(self, "_cached_nodes", None):
                think_lower = think_text.lower()
                attention_weights = {}
                # Basic string frequency counter for simple sub-graph attention
                # In a production AI app, could use spaCy NER or exact regex
                for lower_id, real_id in self._cached_nodes.items():
                    if len(lower_id) > 3:  # Ignore tiny unigrams
                        count = think_lower.count(lower_id)
                        if count > 0:
                            attention_weights[real_id] = count

                # Normalize and save Top-10
                if attention_weights:
                    total_attn = sum(attention_weights.values())
                    top_10 = sorted(attention_weights.items(), key=lambda x: -x[1])[:10]
                    with get_db() as db:
                        try:
                            for nid, count in top_10:
                                weight = count / total_attn
                                db.execute(
                                    "INSERT INTO llm_attention_weights (id, cycle_id, agent_step, node_id, weight) VALUES (%s, %s, %s, %s, %s)",
                                    [
                                        str(uuid.uuid4()),
                                        meta.get("cycle_id", ""),
                                        meta.get("agent_name", ""),
                                        nid,
                                        weight,
                                    ],
                                )
                        except Exception as e:
                            logger.error(
                                f"[VLLMClient] Failed to save attention weight: {e}"
                            )

            if not item.future.done():
                result_dict = {
                    "text": content,
                    "tool_calls": item.payload.get(
                        "_tool_calls_result"
                    ),  # Extracted in _call_vllm_direct
                    "total_tokens": total_tokens,
                    "elapsed_ms": elapsed_ms,
                    "endpoint_name": ep.name,
                    "model_name": item.payload.get("model", self.model),
                }
                item.future.set_result(result_dict)

            # Log to DB for DeepEval benchmarking
            # Skip internal auditor/evaluator agents to prevent circular logging
            # (DeepEval judge calls would be re-evaluated as "pending decisions")
            try:
                from app.services.rlm.rlm_audit import log_rlm_audit_trail

                agent_name_str = meta.get("agent_name", "unknown")

                # These agents are internal to the evaluation system — logging
                # their outputs would create a feedback loop where judge responses
                # get picked up as new "pending decisions" on the next audit run.
                _AUDIT_INTERNAL_AGENTS = {
                    "deepeval_judge",
                    "judge_evaluator",
                    "strategy_evaluator",
                    "unknown",
                    "user_chat",
                }

                if (
                    agent_name_str not in _AUDIT_INTERNAL_AGENTS
                    and meta.get("ticker", "UNKNOWN") != "UNKNOWN"
                ):
                    step_map = {
                        "hermes_research": "collection",
                        "technical": "analysis",
                        "fundamental": "analysis",
                        "sentiment": "analysis",
                        "fund_flow": "analysis",
                        "risk": "analysis",
                        "comparative": "analysis",
                        "technical_meta": "analysis",
                        "fundamental_meta": "analysis",
                        "sentiment_meta": "analysis",
                        "fund_flow_meta": "analysis",
                        "risk_meta": "analysis",
                        "comparative_meta": "analysis",
                        "trading_phase": "trading",
                        "decision_engine": "trading",
                    }
                    mapped_step = step_map.get(agent_name_str, agent_name_str)

                    log_rlm_audit_trail(
                        cycle_id=meta.get("cycle_id", "manual"),
                        bot_id=meta.get("bot_id", "default"),
                        ticker=meta.get("ticker", "UNKNOWN"),
                        context=meta.get("user_prompt", ""),
                        trading_system_prompt=meta.get("system_prompt", ""),
                        active_model=item.payload.get("model", self.model),
                        response_text=content,
                        tokens_used=total_tokens,
                        execution_time=elapsed_ms / 1000.0,
                        agent_step=mapped_step,
                        endpoint_name=meta.get("_endpoint_name", ""),
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        queue_wait_ms=meta.get("_queue_wait_ms", 0),
                    )
            except Exception as audit_err:
                logger.error(
                    "[VLLMClient] Failed to write to llm_audit_logs: %s", audit_err
                )

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            await tracker.record(
                agent_name=meta.get("agent_name", "unknown"),
                ticker=meta.get("ticker", ""),
                cycle_id=meta.get("cycle_id", ""),
                bot_id=meta.get("bot_id", ""),
                model=item.payload.get("model", self.model),
                system_prompt=meta.get("system_prompt", ""),
                user_prompt=meta.get("user_prompt", ""),
                response_text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=elapsed_ms,
                semaphore_active=semaphore_active,
                semaphore_max=semaphore_max,
                success=False,
                error=str(e),
                endpoint_name=ep.name,
            )
            if not item.future.done():
                item.future.set_exception(e)
        finally:
            ep.active_count -= 1

    async def _call_endpoint(
        self,
        client: httpx.AsyncClient,
        url: str,
        json_payload: dict,
        headers: dict | None = None,
    ) -> httpx.Response:
        """Shared API wrapper with explicit timeout and retry logic.

        Respects cycle_control.is_stopped between retries — aborts early
        instead of burning GPU time on a request the pipeline no longer needs.
        """
        import asyncio
        from httpx import RequestError, HTTPStatusError

        max_retries = 3
        backoff = 1.0

        for attempt in range(max_retries):
            # Abort retries if pipeline was stopped between attempts
            if attempt > 0:
                try:
                    from app.pipeline.orchestration.cycle_control import cycle_control
                    if cycle_control.is_stopped:
                        raise asyncio.CancelledError("Pipeline stopped — aborting retry")
                except ImportError:
                    pass
            try:
                # Task 9: Explicit timeout configuration (not using default)
                r = await client.post(
                    url,
                    json=json_payload,
                    headers=headers,
                    timeout=240.0,
                )
                r.raise_for_status()
                return r
            except (RequestError, HTTPStatusError) as e:
                # Do not retry on 4xx client errors
                if (
                    isinstance(e, HTTPStatusError)
                    and 400 <= e.response.status_code < 500
                ):
                    raise
                if attempt == max_retries - 1:
                    logger.error(
                        "[VLLM] API call to %s failed after %d attempts: %s",
                        url,
                        max_retries,
                        repr(e),
                    )
                    raise
                logger.warning(
                    "[VLLM] API call to %s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    url,
                    attempt + 1,
                    max_retries,
                    repr(e),
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(
            f"Failed to call endpoint {url} after {max_retries} attempts"
        )

    # ── URL / Endpoint Resolution ──────────────────────────────────────

    def _find_endpoint_by_url(self, url: str) -> VLLMEndpoint | None:
        """Find the endpoint that owns a given URL."""
        for ep in self._endpoints.values():
            if ep.url == url:
                return ep
        return None

    def _find_endpoint_by_name(self, name: str) -> VLLMEndpoint | None:
        """Find an endpoint by name."""
        return self._endpoints.get(name)

    async def _resolve_base_url(self, model_id: str) -> str:
        """Resolve which vLLM instance hosts a given model.

        Checks the cache first, then queries all endpoints.
        Also triggers role discovery on first call to populate
        the model-to-URL mapping.
        Raises an error if the model cannot be found, rather than
        silently defaulting to an endpoint.
        """
        # Check cache first — but only if endpoint is enabled
        if model_id and model_id in self._model_endpoint_cache:
            ep_name = self._model_endpoint_cache[model_id]
            ep = self._endpoints.get(ep_name)
            if ep and ep.enabled and ep.role != "training":
                return ep.url

        # Auto-discover roles on first cache miss (populates cache)
        if not self._roles_discovered:
            await self.discover_roles()
            if model_id in self._model_endpoint_cache:
                ep_name = self._model_endpoint_cache[model_id]
                ep = self._endpoints.get(ep_name)
                if ep and ep.enabled and ep.role != "training":
                    return ep.url

        # Query all enabled, non-training endpoints (INCLUDING 'trader')
        client = await self._get_client()
        for ep in self._endpoints.values():
            if not ep.enabled or ep.role == "training":
                continue
            try:
                r = await client.get(f"{ep.url}/v1/models", timeout=10.0)
                if r.status_code == 200:
                    data = r.json()
                    for m in data.get("data", []):
                        self._model_endpoint_cache[m["id"]] = ep.name
                        if m["id"] == model_id:
                            logger.info(
                                "[VLLM] Model '%s' resolved to %s (%s)",
                                model_id,
                                ep.name,
                                ep.url,
                            )
                            return ep.url
            except Exception as e:
                logger.debug("[VLLM] Failed to query %s for models: %s", ep.url, e)
                continue

        logger.error(
            "[VLLM] Model '%s' not found on any enabled instance! Raising error.",
            model_id,
        )
        raise ValueError(
            f"Model '{model_id}' is not hosted on any available vLLM endpoint."
        )

    async def _call_vllm_direct(
        self,
        client: httpx.AsyncClient,
        payload: dict,
        meta: dict,
        start: float,
        ep: VLLMEndpoint | None = None,
    ) -> tuple[str, int, int]:
        """Direct call to vLLM (bypassing Prism).

        When called from the queue dispatcher, `ep` is the endpoint that
        was already selected — we use its URL directly instead of re-resolving.
        This eliminates redundant GET /v1/models queries.
        """
        if ep:
            # Use the endpoint's URL directly — no re-resolution needed
            base_url = ep.url
            model_id = ep.model or payload.get("model", self.model)
            # Ensure payload model matches what this endpoint actually serves
            payload["model"] = model_id
        else:
            # Legacy/fallback path (e.g., called outside the dispatcher)
            model_id = payload.get("model", self.model)
            base_url = await self._resolve_base_url(model_id)

        # Strip chat_template_kwargs for non-Qwen models — they don't support it
        # and will 500 error (e.g., NVIDIA Nemotron)
        if not _is_qwen_model(model_id) and "chat_template_kwargs" in payload:
            payload = {k: v for k, v in payload.items() if k != "chat_template_kwargs"}
            logger.debug(
                "[VLLM] Stripped chat_template_kwargs for non-Qwen model: %s", model_id
            )

        r = await self._call_endpoint(
            client=client,
            url=f"{base_url}/v1/chat/completions",
            json_payload=payload,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = r.json()

        usage = data.get("usage", {})
        raw_text = data["choices"][0]["message"].get("content") or ""
        content, think_content = strip_think_tags(raw_text, return_think_content=True)
        meta["_think_content"] = think_content
        total_tokens = usage.get("total_tokens", 0)

        # Store usage in metadata for tracker
        meta["_usage"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

        # Capture tool calls if any
        tool_calls = data["choices"][0]["message"].get("tool_calls")
        if tool_calls:
            payload["_tool_calls_result"] = tool_calls

        return content, total_tokens, elapsed_ms

    async def _call_prism_agent(
        self,
        client: httpx.AsyncClient,
        payload: dict,
        meta: dict,
        start: float,
    ) -> tuple[str, int, int]:
        """Route through Prism's /agent endpoint (agentic loop with tool calling).

        Like Lupos bot's generateAgentResponse: sends messages to Prism which
        assembles the persona system prompt, runs the agentic loop, executes
        tools autonomously, and returns the final response. Prism handles all
        request logging, conversation persistence, and metrics natively.
        """
        model_id = payload.get("model", self.model)
        system_prompt = meta.get("system_prompt", "")
        agent_name = meta.get("agent_name", "pipeline")
        ticker = meta.get("ticker", "")
        cycle_id = meta.get("cycle_id", "")

        # Build Prism /agent payload (flat OpenAI-style, like Lupos bot)
        # Pipeline calls disable the agentic loop to prevent the coordinator
        # from spawning team_create workers — these are simple prompt→response tasks.
        # Only interactive chat (user_chat) gets the full agentic loop.
        is_interactive = agent_name == "user_chat"
        prism_payload = self.prism_client.get_chat_payload_and_url(
            model=model_id,
            messages=payload.get("messages", []),
            max_tokens=payload.get("max_tokens", 1024),
            temperature=payload.get("temperature", 0.3),
            system_prompt=system_prompt,
            agent_name=agent_name,
            ticker=ticker,
            cycle_id=cycle_id,
            enable_thinking=payload.get("chat_template_kwargs", {}).get("enable_thinking", False),
            # NEVER forward tool schemas to Prism — tools are executed locally
            # by the trading bot's agent_loop.py. Forwarding them causes
            # Prism's coordinator to attempt execution, creating infinite loops.
            tools=None,
            is_qwen_model=_is_qwen_model(model_id),
            agentic_mode=is_interactive,
        )
        agent_payload, target_url, headers = prism_payload

        # Server-to-server: skip conversation persistence, auto-approve tools
        agent_payload["skipConversation"] = True
        agent_payload["autoApprove"] = True

        r = await self._call_endpoint(
            client=client,
            url=target_url,
            json_payload=agent_payload,
            headers=headers,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        data = r.json()

        # Parse Prism agent response format
        content = data.get("text") or ""
        thinking = data.get("thinking") or ""
        if thinking:
            meta["_think_content"] = thinking

        # Extract usage from Prism response
        usage = data.get("usage", {})
        total_tokens = (
            usage.get("totalTokens")
            or usage.get("total_tokens")
            or (usage.get("inputTokens", 0) + usage.get("outputTokens", 0))
            or 0
        )

        meta["_usage"] = {
            "prompt_tokens": usage.get("inputTokens", usage.get("prompt_tokens", 0)),
            "completion_tokens": usage.get("outputTokens", usage.get("completion_tokens", 0)),
        }

        # Capture tool calls if returned
        tool_calls = data.get("toolCalls")
        if tool_calls:
            payload["_tool_calls_result"] = tool_calls

        return content, total_tokens, elapsed_ms

    @property
    def _active_slots(self) -> int:
        """How many slots are currently in use across all endpoints."""
        return sum(ep.active_count for ep in self._endpoints.values())

    def queue_status(self) -> dict:
        """Return current queue and slot utilization.

        Returns a dict keyed by endpoint name with backward-compatible
        'jetson' and 'dgx' keys for existing consumers.
        """
        result = {}
        now = time.monotonic()
        for name, ep in self._endpoints.items():
            result[name] = {
                "active": ep.requests_running if ep.requests_running > 0 else ep.active_count,
                "max_concurrent": ep.max_concurrent,
                "batch_size": ep.batch_size,
                "queued": ep.requests_waiting if ep.requests_waiting > 0 else (ep.queue.qsize() if ep.queue else 0),
                "pipeline_max": max(1, ep.max_concurrent - self.RESERVED_HIGH_SLOTS),
                "model": ep.model,
                "role": ep.role,
                "enabled": ep.enabled,
                "circuit_breaker_open": ep.circuit_open_until > now,
                "consecutive_failures": ep.consecutive_batch_failures,
            }
        # Backward compat: cli.py references q_status["dgx"]
        if "dgx_spark" in result and "dgx" not in result:
            result["dgx"] = result["dgx_spark"]
        result["reserved_for_chat"] = self.RESERVED_HIGH_SLOTS
        return result

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        priority: Priority = Priority.NORMAL,
        # Monitoring metadata — passed by callers
        agent_name: str = "unknown",
        ticker: str = "",
        cycle_id: str = "",
        bot_id: str = "",
        model_override: str | None = None,
        endpoint_override: str | None = None,
        history: list[dict] | None = None,
        images: list[str] | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[str, int, int]:
        """
        Enqueue a chat completion request routed through Prism.

        Returns: (response_text, total_tokens, elapsed_ms)
        Requests are dispatched by priority. HIGH priority (user chat)
        will be served ahead of NORMAL (pipeline) and LOW (background).

        model_override: if provided, use this model instead of self.model.
        images: list of base64 data URIs for vision support.
        tools: optional list of tool schemas for function calling.
        """
        # ── Stop gate: reject pipeline requests when cycle is stopped ──
        # HIGH priority (user chat) is always allowed through so the user
        # can still interact even during/after a stop.
        if priority > Priority.HIGH:
            try:
                from app.pipeline.orchestration.cycle_control import cycle_control
                if cycle_control.is_stopped:
                    raise asyncio.CancelledError("Pipeline stopped — rejecting new LLM request")
            except ImportError:
                pass

        self._ensure_dispatcher()

        if not self._roles_discovered:
            await self.discover_roles()

        # ── Endpoint-first routing ──
        # Pick the target endpoint FIRST, then derive the model from it.
        # This prevents cross-endpoint model name leakage (e.g., sending a
        # Jetson Qwen model name to a DGX that runs a different model).
        if endpoint_override:
            target_ep = self._find_endpoint_by_name(endpoint_override)
            if not target_ep:
                raise ValueError(
                    f"Endpoint name override '{endpoint_override}' not found"
                )
        elif model_override:
            # Caller explicitly requested a specific model — find an endpoint
            # that hosts it, or fall back to the best available endpoint.
            target_ep = self._pick_best_endpoint(requested_model=model_override, agent_name=agent_name)
        else:
            # No override — just pick the least loaded endpoint
            target_ep = self._pick_best_endpoint(agent_name=agent_name)

        # The model we actually send is ALWAYS from the target endpoint.
        # model_override is only used to influence endpoint selection above,
        # never leaked directly into the payload for a different box.
        effective_model = target_ep.model or model_override or self.model

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)

        # Build user message with optional vision content
        if images and len(images) > 0:
            user_content = []
            if user:
                user_content.append({"type": "text", "text": user})
            for img_uri in images:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": img_uri}}
                )
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user})

        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        # Only inject chat_template_kwargs for Qwen models — non-Qwen models
        # (NVIDIA Nemotron, Llama, etc.) don't support this and will 500 error
        if not enable_thinking and _is_qwen_model(effective_model):
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        metadata = {
            "agent_name": agent_name,
            "ticker": ticker,
            "cycle_id": cycle_id,
            "bot_id": bot_id,
            "system_prompt": system,
            "user_prompt": user,
            "priority": priority.name,
            "_enqueue_time": time.monotonic(),
        }

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict] = loop.create_future()

        self._seq += 1
        item = QueueItem(
            priority=int(priority),
            seq=self._seq,
            future=future,
            payload=payload,
            metadata=metadata,
        )

        await target_ep.queue.put(item)

        qs = target_ep.queue.qsize()
        active = target_ep.active_count
        max_c = target_ep.max_concurrent
        if qs > 0 or active >= max_c:
            logger.debug(
                "[QUEUE] %s enqueued to %s (priority=%s, active=%d/%d, queued=%d)",
                agent_name,
                target_ep.name,
                priority.name,
                active,
                max_c,
                qs,
            )

        try:
            result_dict = await asyncio.wait_for(
                future, timeout=float(settings.VLLM_FUTURE_TIMEOUT)
            )
        except asyncio.TimeoutError:
            # Penalize the endpoint so future requests prefer other boxes
            target_ep.timeout_penalty_until = time.monotonic() + TIMEOUT_PENALTY_SECONDS
            logger.error(
                "[VLLM] ⏰ Future TIMEOUT after %ds | agent=%s ticker=%s endpoint=%s | "
                "active=%d/%d queued=%d | PENALTY applied for %ds",
                settings.VLLM_FUTURE_TIMEOUT,
                agent_name,
                ticker,
                target_ep.name,
                target_ep.active_count,
                target_ep.max_concurrent,
                target_ep.queue.qsize() if target_ep.queue else -1,
                TIMEOUT_PENALTY_SECONDS,
            )
            if not future.done():
                future.cancel()
            raise RuntimeError(
                f"vLLM future timeout ({settings.VLLM_FUTURE_TIMEOUT}s): "
                f"{agent_name}/{ticker} on {target_ep.name}"
            )
        except asyncio.CancelledError:
            logger.warning(
                "[VLLM] 🛑 Task cancelled while waiting for future | agent=%s ticker=%s endpoint=%s",
                agent_name,
                ticker,
                target_ep.name,
            )
            if not future.done():
                future.cancel()
            raise
        return (
            result_dict["text"],
            result_dict["total_tokens"],
            result_dict["elapsed_ms"],
        )

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        priority: Priority = Priority.NORMAL,
        # Monitoring metadata — passed by callers
        agent_name: str = "unknown",
        ticker: str = "",
        cycle_id: str = "",
        bot_id: str = "",
        model_override: str | None = None,
        endpoint_override: str | None = None,
    ) -> dict:
        """
        Enqueue a chat completion request with tool support and raw messages array.

        Returns: dict with keys: 'text', 'total_tokens', 'elapsed_ms', 'tool_calls'
        """
        # ── Stop gate: reject pipeline requests when cycle is stopped ──
        if priority > Priority.HIGH:
            try:
                from app.pipeline.orchestration.cycle_control import cycle_control
                if cycle_control.is_stopped:
                    raise asyncio.CancelledError("Pipeline stopped — rejecting new LLM request")
            except ImportError:
                pass

        self._ensure_dispatcher()

        if not self._roles_discovered:
            await self.discover_roles()

        # ── Endpoint-first routing (same pattern as chat()) ──
        if endpoint_override:
            target_ep = self._find_endpoint_by_name(endpoint_override)
            if not target_ep:
                raise ValueError(
                    f"Endpoint name override '{endpoint_override}' not found"
                )
        elif model_override:
            target_ep = self._pick_best_endpoint(requested_model=model_override, agent_name=agent_name)
        else:
            target_ep = self._pick_best_endpoint(agent_name=agent_name)

        effective_model = target_ep.model or model_override or self.model

        # Sanitize messages to remove non-standard keys like 'service_source' before sending to LLM
        sanitized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                clean_msg = {k: v for k, v in msg.items() if k != "service_source"}
                sanitized_messages.append(clean_msg)
            else:
                sanitized_messages.append(msg)

        payload: dict[str, Any] = {
            "model": effective_model,
            "messages": sanitized_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        if not enable_thinking and _is_qwen_model(effective_model):
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Find the last user message for tracking
        user_prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break

        system_prompt = ""
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content", "")

        metadata = {
            "agent_name": agent_name,
            "ticker": ticker,
            "cycle_id": cycle_id,
            "bot_id": bot_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "priority": priority.name,
        }

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict] = loop.create_future()

        self._seq += 1
        item = QueueItem(
            priority=int(priority),
            seq=self._seq,
            future=future,
            payload=payload,
            metadata=metadata,
        )

        await target_ep.queue.put(item)

        qs = target_ep.queue.qsize()
        active = target_ep.active_count
        max_c = target_ep.max_concurrent
        if qs > 0 or active >= max_c:
            logger.debug(
                "[QUEUE] %s enqueued to %s (priority=%s, active=%d/%d, queued=%d)",
                agent_name,
                target_ep.name,
                priority.name,
                active,
                max_c,
                qs,
            )

        try:
            return await asyncio.wait_for(
                future, timeout=float(settings.VLLM_FUTURE_TIMEOUT)
            )
        except asyncio.TimeoutError:
            # Penalize the endpoint so future requests prefer other boxes
            target_ep.timeout_penalty_until = time.monotonic() + TIMEOUT_PENALTY_SECONDS
            logger.error(
                "[VLLM] ⏰ Future TIMEOUT after %ds | agent=%s ticker=%s endpoint=%s (chat_with_tools) | PENALTY applied for %ds",
                settings.VLLM_FUTURE_TIMEOUT,
                agent_name,
                ticker,
                target_ep.name,
                TIMEOUT_PENALTY_SECONDS,
            )
            if not future.done():
                future.cancel()
            raise RuntimeError(
                f"vLLM future timeout ({settings.VLLM_FUTURE_TIMEOUT}s): "
                f"{agent_name}/{ticker} on {target_ep.name} (chat_with_tools)"
            )
        except asyncio.CancelledError:
            logger.warning(
                "[VLLM] 🛑 Task cancelled while waiting for future | agent=%s ticker=%s endpoint=%s (chat_with_tools)",
                agent_name,
                ticker,
                target_ep.name,
            )
            if not future.done():
                future.cancel()
            raise

    def end_session(self, group_key: str):
        """Clear cached session for a completed cycle."""
        self.prism_client.end_session(group_key)

    # ── Health ─────────────────────────────────────────────────────────

    async def health(self) -> bool:
        """Check if ANY vLLM server is healthy (direct check, bypasses Prism)."""
        client = await self._get_client()
        for ep in self._endpoints.values():
            try:
                r = await client.get(f"{ep.url}/health", timeout=5.0)
                if r.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    async def health_all(self) -> dict[str, bool]:
        """Check health of ALL vLLM endpoints concurrently.

        Returns dict with endpoint name → bool.
        Always includes 'jetson' and 'dgx_spark' keys for backward compat.

        Uses asyncio.gather so all endpoints are checked in parallel,
        capping worst-case latency at ~5s instead of 5s × N endpoints.
        """
        client = await self._get_client()
        result: dict[str, bool] = {}

        async def _check(name: str, ep):
            try:
                r = await client.get(f"{ep.url}/health", timeout=5.0)
                result[name] = r.status_code == 200
            except Exception:
                result[name] = False

        await asyncio.gather(
            *[_check(name, ep) for name, ep in self._endpoints.items()]
        )

        # Ensure backward-compat keys exist
        result.setdefault("jetson", False)
        result.setdefault("dgx_spark", False)
        return result

    # ── Model Discovery ────────────────────────────────────────────────

    async def list_models(self) -> list[str]:
        """List models loaded across ALL endpoints."""
        client = await self._get_client()
        all_models: list[str] = []
        for ep in self._endpoints.values():
            try:
                r = await client.get(f"{ep.url}/v1/models", timeout=10.0)
                if r.status_code == 200:
                    for m in r.json().get("data", []):
                        if m["id"] not in all_models:
                            all_models.append(m["id"])
            except Exception:
                continue
        return all_models

    async def discover_roles(self) -> dict:
        """Auto-discover which model is on which endpoint.

        Queries each endpoint's /v1/models API and caches the mapping.
        Sets self.model to the first available model if not already set.

        Role assignment is based on endpoint configuration:
          - Jetson → COLLECTOR
          - DGX Spark(s) → ANALYST

        Returns dict with per-endpoint model info.
        """
        if not hasattr(self, "_discovery_lock") or self._discovery_lock is None:
            self._discovery_lock = asyncio.Lock()

        async with self._discovery_lock:
            if self._roles_discovered:
                roles = {}
                roles["collector_model"] = None
                roles["analyst_model"] = None
                roles["analyst_models"] = []
                for name, ep in self._endpoints.items():
                    roles[f"{name}_model"] = ep.model
                    roles[f"{name}_url"] = ep.url
                    if ep.role == "collector" and ep.model:
                        roles["collector_model"] = ep.model
                    elif ep.role == "analyst" and ep.model:
                        roles["analyst_models"].append(ep.model)
                return roles
            return await self._discover_roles_unlocked()

    async def _discover_roles_unlocked(self) -> dict:
        client = await self._get_client()
        roles = {}
        successful_endpoints = 0

        async def _check_endpoint(name, ep):
            nonlocal successful_endpoints
            roles[f"{name}_model"] = None
            roles[f"{name}_url"] = ep.url
            try:
                r = await client.get(f"{ep.url}/v1/models", timeout=5.0)
                if r.status_code == 200:
                    models = r.json().get("data", [])
                    if models:
                        ep.model = models[0]["id"]
                        roles[f"{name}_model"] = models[0]["id"]
                        for m in models:
                            self._model_endpoint_cache[m["id"]] = name

                        # Capture max_model_len for context budget system
                        raw_ctx = models[0].get("max_model_len", 0)
                        if raw_ctx > 0:
                            ep.max_model_len = raw_ctx
                            register_model_context(ep.model, raw_ctx)
                            logger.info(
                                "[VLLM] Endpoint %s context window: %d tokens",
                                name,
                                raw_ctx,
                            )

                        # Apply dynamic model limit if configured
                        limits = get_model_limits()
                        if ep.model in limits:
                            new_max = limits[ep.model]
                            if new_max != ep.max_concurrent:
                                logger.info(
                                    "[VLLM] Auto-scaling endpoint %s concurrency: %d -> %d (model: %s)",
                                    name,
                                    ep.max_concurrent,
                                    new_max,
                                    ep.model,
                                )
                                ep.max_concurrent = new_max
                                ep.init_concurrency(self.RESERVED_HIGH_SLOTS)

                        logger.info(
                            "[VLLM] Endpoint %s (%s) → model: %s",
                            name,
                            ep.url,
                            ep.model,
                        )
                        if ep.auto_disabled and not ep.enabled:
                            ep.enabled = True
                            ep.auto_disabled = False
                            ep.consecutive_batch_failures = 0
                            ep.circuit_open_until = 0.0
                            ep.timeout_penalty_until = 0.0
                            ep.init_concurrency(self.RESERVED_HIGH_SLOTS)
                            logger.info(
                                "[VLLM] ✅ Endpoint %s at %s back online — re-enabled via discovery",
                                name, ep.url,
                            )
                        ep.loading = False
                        successful_endpoints += 1
                    else:
                        # Server responded but no models loaded — LOADING state
                        ep.loading = True
                        ep.model = None
                        logger.info(
                            "[VLLM] ⏳ Endpoint %s at %s is online but model is still loading",
                            name, ep.url,
                        )
            except Exception as e:
                ep.loading = False
                logger.debug("[VLLM] Failed to discover models on %s: %s", name, e)

        # Gather concurrently so 3 offline endpoints don't block for 15s sequentially
        await asyncio.gather(
            *[_check_endpoint(name, ep) for name, ep in self._endpoints.items()]
        )

        # ── Auto-disable endpoints that failed model discovery ──
        # If an endpoint is configured but has no model after probing, it means
        # the vLLM docker is offline or not serving any model. Auto-disable it
        # so the dispatcher doesn't try to route requests there.
        # Endpoints in 'loading' state stay enabled — they'll have a model soon.
        for name, ep in self._endpoints.items():
            if not ep.model and ep.enabled and not ep.loading:
                ep.enabled = False
                ep.auto_disabled = True
                logger.warning(
                    "[VLLM] ⚠️ Endpoint %s at %s is unreachable — auto-disabled. "
                    "It will be re-enabled automatically when it comes back online.",
                    name,
                    ep.url,
                )
            elif not ep.model and ep.enabled and ep.loading:
                logger.info(
                    "[VLLM] ⏳ Endpoint %s at %s is loading — keeping enabled, "
                    "requests will cross-route to other boxes until model is ready.",
                    name, ep.url,
                )

        if len(self._endpoints) > 0 and successful_endpoints == 0:
            # Check if any endpoints are loading (not truly dead)
            loading_count = sum(1 for ep in self._endpoints.values() if ep.loading)
            if loading_count > 0:
                logger.warning(
                    "[VLLM] ⚠️ No endpoints have models loaded yet, but %d are still loading. "
                    "Will retry discovery in background. Requests will fail until a model is ready.",
                    loading_count,
                )
                # Don't raise — let the system boot and rediscovery will pick them up
            else:
                raise RuntimeError(
                    "All configured vLLM endpoints failed to respond or returned connection errors. "
                    "Please verify the inference servers are running."
                )

        self._roles_discovered = True
        self._ensure_dispatcher()

        # ALWAYS set self.model from discovery — the ACTIVE_MODEL env var
        # is only a seed value and must NOT override what's actually loaded
        # on the hardware. This prevents stale model names from leaking
        # into payloads when the user swaps models on the vLLM docker.
        old_model = self.model
        self.model = ""  # Reset — will be populated from live endpoints
        # Prefer collector (Jetson) model
        for ep in self._endpoints.values():
            if ep.role == "collector" and ep.model and ep.enabled:
                self.model = ep.model
                break
        # Fallback to any available model
        if not self.model:
            for ep in self._endpoints.values():
                if ep.model and ep.enabled:
                    self.model = ep.model
                    break
        if self.model and self.model != old_model:
            logger.info(
                "[VLLM] Active model updated: %s → %s (from live discovery)",
                old_model or '(empty)', self.model,
            )
        elif self.model:
            logger.info("[VLLM] Active model confirmed: %s", self.model)

        # Build backward-compat summary
        collector_model = None
        analyst_models = []
        for ep in self._endpoints.values():
            if ep.role == "collector" and ep.model:
                collector_model = ep.model
            elif ep.role == "analyst" and ep.model:
                analyst_models.append(ep.model)

        roles["collector_model"] = None
        roles["analyst_model"] = None
        roles["analyst_models"] = []

        logger.info("[VLLM] Swarm endpoints auto-discovered.")
        return roles

    def get_least_busy_model(self) -> str:
        """Return the model ID from the endpoint with the lowest load score.

        Uses the same capacity-aware scoring as _pick_best_endpoint(),
        including timeout penalties and queue depth awareness.
        """
        active_eps = [
            ep
            for ep in self._endpoints.values()
            if ep.enabled and ep.model and ep.role != "training"
        ]

        if not active_eps:
            if not self._roles_discovered:
                return self.model

            # Try to return the default model if endpoints are present but missing roles
            fallback_eps = [
                ep for ep in self._endpoints.values() if ep.enabled and ep.model
            ]
            if fallback_eps:
                return fallback_eps[0].model

            raise RuntimeError("No vLLM endpoints available for load balancing.")

        best = min(active_eps, key=lambda ep: ep.load_score)
        logger.debug(
            "[VLLM] Balanced model selection: %s (score=%.0f, active=%d, queued=%d)",
            best.name,
            best.load_score,
            best.active_count,
            best.queue.qsize() if best.queue else 0,
        )
        return best.model

    def get_trader_model(self) -> str | None:
        """Return the model ID for the dedicated trader endpoint.

        The trader endpoint (role='trader') hosts the most capable model
        and is reserved for final trading decisions. Returns fallback to
        the least-busy model if no dedicated trader is configured.
        """
        for ep in self._endpoints.values():
            if ep.role == "trader" and ep.enabled and ep.model:
                return ep.model
        # Fallback: no dedicated trader — use least busy
        return self.get_least_busy_model()

    def get_trader_url(self) -> str | None:
        """Return the base URL for the dedicated trader endpoint."""
        for ep in self._endpoints.values():
            if ep.role == "trader" and ep.enabled:
                return ep.url
        # Fallback: first enabled endpoint
        for ep in self._endpoints.values():
            if ep.enabled:
                return ep.url
        return None

    def get_analyst_model_balanced(self) -> tuple[str | None, str | None]:
        """Return (model_id, endpoint_name) from the least-busy analyst endpoint.

        Excludes trader and training endpoints. Used by rlm_wrapper.py for
        the RLM analysis sessions. Returns a tuple so the caller can also
        identify which box was selected for routing.
        """
        analyst_eps = [
            ep
            for ep in self._endpoints.values()
            if ep.enabled and ep.model and ep.role not in ("training", "trader")
        ]
        if not analyst_eps:
            # No analyst endpoints — fall back to any enabled endpoint
            fallback = [
                ep for ep in self._endpoints.values() if ep.enabled and ep.model
            ]
            if fallback:
                best = min(
                    fallback,
                    key=lambda ep: (
                        ep.active_count + (ep.queue.qsize() if ep.queue else 0)
                    ),
                )
                return best.model, best.name
            return self.model, "jetson"

        best = min(
            analyst_eps,
            key=lambda ep: ep.active_count + (ep.queue.qsize() if ep.queue else 0),
        )
        return best.model, best.name

    @property
    def _analyst_url(self) -> str | None:
        """Return the base URL for the first active endpoint.

        Used by rlm_wrapper.py to point the RLM OpenAI client at the
        correct vLLM server.
        """
        for ep in self._endpoints.values():
            if ep.enabled and ep.role != "training":
                return ep.url
        return None

    def get_role_info(self) -> dict:
        """Return current role assignments for API consumption.

        Returns per-endpoint info grouped by role.
        """
        endpoints_info = {}
        for name, ep in self._endpoints.items():
            endpoints_info[name] = {
                "endpoint": name,
                "url": ep.url,
                "model": ep.model,
                "role": "swarm_node" if ep.role != "training" else "training",
                "enabled": ep.enabled,
                "auto_disabled": ep.auto_disabled,
                "purpose": "Swarm Load Balancing Node",
                "max_concurrent": ep.max_concurrent,
            }

        return {
            "endpoints": endpoints_info,
            "swarm_mode": True,
            "collector": None,
            "analyst": None,
            "analysts": [],
            "trader": None,
        }

    # ── Background Rediscovery ─────────────────────────────────────────

    async def _rediscovery_loop(self):
        """Background task: re-probe ALL endpoints (including auto-disabled ones) every 60s.

        This allows endpoints to come back online automatically when:
        - A vLLM docker is restarted with a different model
        - A box that was offline comes back
        - A model is swapped out on a running instance

        Only auto-disabled endpoints are re-enabled. Manually disabled endpoints
        are left alone (user explicitly turned them off).
        """
        await asyncio.sleep(30.0)  # Initial delay — let startup finish
        while True:
            try:
                await self.rediscover_endpoints()
            except Exception as e:
                logger.debug("[VLLM] Rediscovery loop error: %s", e)
            
            # Fast poll when any endpoint is in loading state (model being loaded)
            # so we detect model readiness within 15s instead of waiting 2 minutes
            any_loading = any(ep.loading for ep in self._endpoints.values())
            any_auto_disabled = any(ep.auto_disabled for ep in self._endpoints.values())
            if any_loading:
                logger.debug("[VLLM] Fast rediscovery (15s) — endpoint(s) loading")
                await asyncio.sleep(15.0)
            elif any_auto_disabled:
                await asyncio.sleep(60.0)  # Medium poll for offline endpoints
            else:
                await asyncio.sleep(120.0)  # Normal poll when everything is healthy

    async def rediscover_endpoints(self):
        """Re-probe all endpoints for model availability.

        - Auto-disabled endpoints that respond with a model → re-enabled
        - Enabled endpoints whose model changed → updated
        - Enabled endpoints that went offline → auto-disabled
        """
        client = await self._get_client()
        changes = []

        async def _probe(name: str, ep: VLLMEndpoint):
            try:
                r = await client.get(f"{ep.url}/v1/models", timeout=5.0)
                if r.status_code == 200:
                    models = r.json().get("data", [])
                    if models:
                        new_model = models[0]["id"]
                        old_model = ep.model

                        # Re-enable auto-disabled endpoint
                        if ep.auto_disabled and not ep.enabled:
                            ep.enabled = True
                            ep.auto_disabled = False
                            ep.model = new_model
                            # Reset circuit breaker on recovery
                            ep.consecutive_batch_failures = 0
                            ep.circuit_open_until = 0.0
                            ep.timeout_penalty_until = 0.0
                            # Re-init concurrency primitives
                            ep.init_concurrency(self.RESERVED_HIGH_SLOTS)
                            for m in models:
                                self._model_endpoint_cache[m["id"]] = name
                            changes.append(
                                f"{name}: RE-ENABLED with model {new_model}"
                            )
                            logger.info(
                                "[VLLM] ✅ Endpoint %s at %s back online with model: %s — re-enabled",
                                name, ep.url, new_model,
                            )
                            # Capture context window
                            raw_ctx = models[0].get("max_model_len", 0)
                            if raw_ctx > 0:
                                ep.max_model_len = raw_ctx
                                register_model_context(new_model, raw_ctx)
                            # Apply dynamic model limits
                            limits = get_model_limits()
                            if new_model in limits:
                                new_max = limits[new_model]
                                if new_max != ep.max_concurrent:
                                    ep.max_concurrent = new_max
                                    ep.init_concurrency(self.RESERVED_HIGH_SLOTS)
                            # Start dispatcher for this endpoint
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    if ep.dispatcher_task is None or ep.dispatcher_task.done():
                                        ep.dispatcher_task = loop.create_task(self._dispatch_loop(ep))
                                    if ep.metrics_task is None or ep.metrics_task.done():
                                        ep.metrics_task = loop.create_task(self._poll_metrics_loop(ep))
                            except RuntimeError:
                                pass
                        elif ep.enabled and old_model != new_model:
                            # Model changed on a running endpoint (hot-swap)
                            # Remove old model from cache
                            if old_model and old_model in self._model_endpoint_cache:
                                del self._model_endpoint_cache[old_model]
                            ep.model = new_model
                            for m in models:
                                self._model_endpoint_cache[m["id"]] = name
                            raw_ctx = models[0].get("max_model_len", 0)
                            if raw_ctx > 0:
                                ep.max_model_len = raw_ctx
                                register_model_context(new_model, raw_ctx)
                            # Apply dynamic model limits for new model
                            limits = get_model_limits()
                            if new_model in limits:
                                new_max = limits[new_model]
                                if new_max != ep.max_concurrent:
                                    ep.max_concurrent = new_max
                                    ep.init_concurrency(self.RESERVED_HIGH_SLOTS)
                            changes.append(
                                f"{name}: model changed {old_model} → {new_model}"
                            )
                            logger.info(
                                "[VLLM] 🔄 Endpoint %s model hot-swapped: %s → %s",
                                name, old_model, new_model,
                            )
                    else:
                        # Server responded but no models loaded
                        if ep.enabled and ep.model:
                            ep.enabled = False
                            ep.auto_disabled = True
                            old_model = ep.model
                            ep.model = None
                            changes.append(
                                f"{name}: no models loaded — auto-disabled (was {old_model})"
                            )
                            logger.warning(
                                "[VLLM] ⚠️ Endpoint %s responded but has no models loaded — auto-disabled",
                                name,
                            )
            except Exception:
                # Endpoint unreachable
                if ep.enabled and ep.model:
                    ep.enabled = False
                    ep.auto_disabled = True
                    old_model = ep.model
                    ep.model = None
                    if old_model and old_model in self._model_endpoint_cache:
                        del self._model_endpoint_cache[old_model]
                    changes.append(
                        f"{name}: unreachable — auto-disabled (was {old_model})"
                    )
                    logger.warning(
                        "[VLLM] ⚠️ Endpoint %s at %s went offline — auto-disabled",
                        name, ep.url,
                    )

        await asyncio.gather(
            *[_probe(name, ep) for name, ep in self._endpoints.items()]
        )

        # Update active model if current one is no longer available
        if self.model:
            model_still_available = any(
                ep.model == self.model and ep.enabled
                for ep in self._endpoints.values()
            )
            if not model_still_available:
                old = self.model
                self.model = ""
                for ep in self._endpoints.values():
                    if ep.enabled and ep.model:
                        self.model = ep.model
                        break
                if self.model:
                    logger.info(
                        "[VLLM] Active model changed: %s → %s (old model no longer available)",
                        old, self.model,
                    )
                else:
                    logger.warning(
                        "[VLLM] ⚠️ No active models available after rediscovery"
                    )

        if changes:
            logger.info("[VLLM] Rediscovery changes: %s", " | ".join(changes))

    # ── Streaming ──────────────────────────────────────────────────────

    async def chat_stream(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 8000,
        enable_thinking: bool = False,
        agent_name: str = "user_chat",
        ticker: str = "",
        model_override: str | None = None,
        endpoint_override: str | None = None,
        history: list[dict] | None = None,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        bypass_prism: bool = False,
    ):
        """Real streaming: connect to vLLM's streaming API and yield tokens live.

        Uses vLLM's /v1/chat/completions with stream=true to deliver tokens
        as they are generated, giving immediate feedback in the UI.

        Think content (between <think>...</think>) is yielded as __THINK__<text>
        events so the frontend can display reasoning separately.

        images: list of base64 data URIs for vision support.

        Yields: str chunks — either raw text tokens or __THINK__<text> or __META__<json>
        """
        import json as _json

        if not self._roles_discovered:
            await self.discover_roles()

        start = time.monotonic()

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)

        # Build user message with optional vision content
        if images and len(images) > 0:
            user_content = []
            if user:
                user_content.append({"type": "text", "text": user})
            for img_uri in images:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": img_uri}}
                )
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user})

        try:
            if endpoint_override:
                target_ep = self._find_endpoint_by_name(endpoint_override)
                if not target_ep:
                    raise ValueError(f"Endpoint {endpoint_override} not found")
            elif model_override:
                target_ep = self._pick_best_endpoint(requested_model=model_override)
            else:
                target_ep = self._pick_best_endpoint()
            base_url = target_ep.url
            effective_model = target_ep.model or model_override or self.model
        except Exception as e:
            logger.error("[STREAM] Failed to resolve endpoint: %s", e)
            yield f"⚠️ LLM error: {str(e)[:200]}"
            yield f"__META__{_json.dumps({'total_tokens': 0, 'elapsed_ms': 0, 'full_text': ''})}"
            return

        prism_is_healthy = False
        if self.prism_client.enabled and not bypass_prism:
            prism_is_healthy = await self.prism_client.check_health()

        if prism_is_healthy:
            # Route through Prism proxy for user chat
            payload, target_url, headers = self.prism_client.get_stream_payload_and_url(
                model=effective_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system,
                agent_name=agent_name,
                ticker=ticker,
                enable_thinking=enable_thinking,
                tools=tools,
                is_qwen_model=_is_qwen_model(effective_model),
            )
        else:
            if self.prism_client.enabled and not bypass_prism:
                logger.warning("[STREAM] Prism Gateway unreachable. Falling back to direct vLLM connection.")
            # Route directly to vLLM
            payload: dict[str, Any] = {
                "model": effective_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if tools:
                payload["tools"] = tools
            if _is_qwen_model(effective_model):
                payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

            target_url = f"{base_url}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}

        full_text = ""
        think_text = ""
        total_tokens = 0
        in_think = False
        think_buffer = ""  # Buffer to detect <think> and </think> tags
        active_tool_calls = {}

        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                target_url,
                json=payload,
                headers=headers,
                timeout=240.0,
            ) as response:
                response.raise_for_status()
                buffer = ""
                async for raw_chunk in response.aiter_text():
                    buffer += raw_chunk
                    # Process complete SSE lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            chunk_data = _json.loads(data_str)
                        except _json.JSONDecodeError:
                            continue

                        # Handle Prism proxy format natively
                        if "type" in chunk_data:
                            ctype = chunk_data.get("type")
                            content = chunk_data.get("content", "")

                            if ctype == "thinking" and content:
                                think_text += content
                                yield f"__THINK__{content}"
                                continue
                            elif ctype == "chunk" and content:
                                full_text += content
                                yield content
                                continue
                            elif ctype == "done":
                                usage = chunk_data.get("usage", {})
                                if usage:
                                    total_tokens = usage.get(
                                        "totalTokens",
                                        usage.get(
                                            "outputTokens", usage.get("total_tokens", 0)
                                        ),
                                    )
                                continue
                            elif ctype == "error":
                                yield f"⚠️ LLM error: {content}"
                                continue
                            continue

                        # Extract vLLM token content
                        choices = chunk_data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # Handle tool calls
                        tool_calls = delta.get("tool_calls", [])
                        if tool_calls:
                            for tc in tool_calls:
                                tc_index = tc.get("index", 0)
                                if tc_index not in active_tool_calls:
                                    active_tool_calls[tc_index] = {
                                        "id": "",
                                        "function": {"name": "", "arguments": ""},
                                    }

                                if "id" in tc and tc["id"]:
                                    active_tool_calls[tc_index]["id"] = tc["id"]

                                func_delta = tc.get("function", {})
                                if "name" in func_delta and func_delta["name"]:
                                    active_tool_calls[tc_index]["function"]["name"] += (
                                        func_delta["name"]
                                    )
                                if (
                                    "arguments" in func_delta
                                    and func_delta["arguments"]
                                ):
                                    active_tool_calls[tc_index]["function"][
                                        "arguments"
                                    ] += func_delta["arguments"]

                        finish_reason = choices[0].get("finish_reason")
                        if finish_reason == "tool_calls":
                            # Stream ended with a tool call
                            for tc in active_tool_calls.values():
                                yield f"__TOOL_CALL__{_json.dumps(tc)}\n"

                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning", "")

                        if reasoning:
                            think_text += reasoning
                            yield f"__THINK__{reasoning}"

                        if not content:
                            # Check for usage in final chunk
                            usage = chunk_data.get("usage")
                            if usage:
                                total_tokens = usage.get("total_tokens", 0)
                            continue

                        # Track full text for saving to DB
                        full_text += content

                        # Handle <think> tag detection and routing
                        # Buffer content to detect tags spanning multiple chunks
                        think_buffer += content

                        while think_buffer:
                            if not in_think:
                                # Look for <think> tag
                                think_start = think_buffer.find("<think>")
                                if think_start >= 0:
                                    # Yield any text before <think> as normal token
                                    before = think_buffer[:think_start]
                                    if before:
                                        yield before
                                    in_think = True
                                    think_buffer = think_buffer[
                                        think_start + 7 :
                                    ]  # skip <think>
                                elif "<" in think_buffer and think_buffer.endswith("<"):
                                    # Might be start of <think> tag, keep buffering
                                    break
                                elif (
                                    think_buffer.endswith("<thin")
                                    or think_buffer.endswith("<thi")
                                    or think_buffer.endswith("<th")
                                    or think_buffer.endswith("<t")
                                ):
                                    # Partial tag, keep buffering
                                    break
                                else:
                                    # No tag, yield as normal token
                                    yield think_buffer
                                    think_buffer = ""
                            else:
                                # Inside <think> block - look for </think>
                                think_end = think_buffer.find("</think>")
                                if think_end >= 0:
                                    # Yield thinking content
                                    think_chunk = think_buffer[:think_end]
                                    if think_chunk:
                                        think_text += think_chunk
                                        yield f"__THINK__{think_chunk}"
                                    in_think = False
                                    think_buffer = think_buffer[
                                        think_end + 8 :
                                    ]  # skip </think>
                                elif (
                                    "</thin" in think_buffer
                                    or "</thi" in think_buffer
                                    or "</th" in think_buffer
                                    or "</" in think_buffer
                                ):
                                    # Might be end of </think> tag, keep buffering
                                    break
                                else:
                                    # Still in think block, yield as thinking
                                    think_text += think_buffer
                                    yield f"__THINK__{think_buffer}"
                                    think_buffer = ""

                # Flush any remaining buffer
                if think_buffer:
                    if in_think:
                        think_text += think_buffer
                        yield f"__THINK__{think_buffer}"
                    else:
                        yield think_buffer

        except Exception as e:
            logger.error("[STREAM] Streaming failed: %s", e)
            yield f"⚠️ LLM error: {str(e)[:200]}"

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Record to tracker
        try:
            await tracker.record(
                agent_name=agent_name,
                ticker=ticker,
                cycle_id="",
                bot_id="",
                model=effective_model,
                system_prompt=system[:500],
                user_prompt=user,
                response_text=full_text[:500],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=total_tokens,
                latency_ms=elapsed_ms,
                semaphore_active=sum(
                    ep.active_count for ep in self._endpoints.values()
                ),
                semaphore_max=sum(ep.max_concurrent for ep in self._endpoints.values()),
            )
        except Exception as e:
            logger.debug("[STREAM] Tracker record failed: %s", e)

        # Strip think tags from full_text for DB storage
        clean_text = strip_think_tags(full_text) if think_text else full_text

        yield f"__META__{_json.dumps({'total_tokens': total_tokens, 'elapsed_ms': elapsed_ms, 'full_text': clean_text})}"

    def drain_queues(self) -> int:
        """Cancel all pending futures in all endpoint queues.

        Called during stop to prevent queued requests from dispatching.
        Returns the total number of items drained.
        """
        total_drained = 0
        for ep in self._endpoints.values():
            if ep.queue is None:
                continue
            while not ep.queue.empty():
                try:
                    item = ep.queue.get_nowait()
                    if not item.future.done():
                        item.future.cancel()
                    ep.queue.task_done()
                    total_drained += 1
                except asyncio.QueueEmpty:
                    break
        if total_drained:
            logger.info("[VLLM] Drained %d queued items on stop", total_drained)
        return total_drained

    def cancel_active_requests(self) -> int:
        """Cancel all active background vLLM HTTP requests.

        Called when stopping a cycle or shutting down to prevent remote boxes
        from processing stale requests.
        """
        cancelled_count = 0
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
                cancelled_count += 1
        if cancelled_count:
            logger.info("[VLLM] Cancelled %d active background requests", cancelled_count)
        return cancelled_count

    async def close(self):
        """Shutdown the persistent HTTP client and dispatchers."""
        # Cancel and drain active/queued requests first
        self.cancel_active_requests()
        self.drain_queues()
        # Wait for active tasks to finalize/cancel
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        for ep in self._endpoints.values():
            if ep.dispatcher_task and not ep.dispatcher_task.done():
                ep.dispatcher_task.cancel()
                try:
                    await ep.dispatcher_task
                except asyncio.CancelledError:
                    pass
            if ep.metrics_task and not ep.metrics_task.done():
                ep.metrics_task.cancel()
                try:
                    await ep.metrics_task
                except asyncio.CancelledError:
                    pass
        if self._rediscovery_task and not self._rediscovery_task.done():
            self._rediscovery_task.cancel()
            try:
                await self._rediscovery_task
            except asyncio.CancelledError:
                pass
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# Singleton — import this everywhere
llm = VLLMClient()

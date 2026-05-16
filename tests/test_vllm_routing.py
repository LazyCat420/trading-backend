"""
Tests for Phase 3: Unified Prism Telemetry (vllm_client routing)

Tests cover:
  🔴🟢 TDD Unit Tests (re-implemented pure functions):
    - Priority enum ordering
    - QueueItem FIFO behavior
    - VLLMEndpoint load_score calculation
    - VLLMEndpoint circuit breaker behavior
    - _is_qwen_model detection
  🔄 Regression Tests (source code checks):
    - Jetson bypass is REMOVED from source
    - Shadow log error level is warning, not debug
"""

import sys
import os
import asyncio
import time
from enum import IntEnum
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Re-implemented data structures for isolated testing ───────────────
# These mirror the definitions in vllm_client.py but are self-contained
# so we don't need psycopg installed.

class Priority(IntEnum):
    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass(order=True)
class QueueItem:
    priority: int
    seq: int
    future: asyncio.Future = field(compare=False)
    payload: dict = field(compare=False)
    metadata: dict = field(compare=False)


MAX_QUEUE_DEPTH_MULTIPLIER = 3


@dataclass
class VLLMEndpoint:
    name: str
    url: str
    role: str
    max_concurrent: int
    purpose: str
    enabled: bool = True
    model: str | None = None
    active_count: int = 0
    cache_usage: float = 0.0
    queue: asyncio.PriorityQueue = field(default=None, repr=False, compare=False)
    slots: asyncio.Semaphore = field(default=None, repr=False, compare=False)
    pipeline_slots: asyncio.Semaphore = field(default=None, repr=False, compare=False)
    timeout_penalty_until: float = 0.0
    batch_size: int = 24
    consecutive_batch_failures: int = 0
    circuit_open_until: float = 0.0

    def init_concurrency(self, reserved_high: int = 1):
        self.queue = asyncio.PriorityQueue()
        self.slots = asyncio.Semaphore(self.max_concurrent)
        pipe_max = max(1, self.max_concurrent - reserved_high)
        self.pipeline_slots = asyncio.Semaphore(pipe_max)

    @property
    def load_score(self) -> float:
        if self.circuit_open_until > time.monotonic():
            return float('inf')
        qs = self.queue.qsize() if self.queue else 0
        score = float(self.active_count + qs)
        if self.timeout_penalty_until > time.monotonic():
            score += self.max_concurrent
        return score

    @property
    def is_overloaded(self) -> bool:
        qs = self.queue.qsize() if self.queue else 0
        return qs > self.max_concurrent * MAX_QUEUE_DEPTH_MULTIPLIER


def _is_qwen_model(model_id: str) -> bool:
    if not model_id:
        return False
    return "qwen" in model_id.lower()


VALID_ROLES = ("collector", "analyst", "trader", "training")


# ══════════════════════════════════════════════════════════════
# 💨 Smoke Tests — Pure data structures
# ══════════════════════════════════════════════════════════════


class TestPriorityEnum:
    def test_ordering(self):
        assert Priority.HIGH < Priority.NORMAL
        assert Priority.NORMAL < Priority.LOW
        assert Priority.HIGH == 0
        assert Priority.NORMAL == 1
        assert Priority.LOW == 2


class TestQueueItemOrdering:
    def test_high_beats_normal(self):
        f1 = asyncio.Future()
        f2 = asyncio.Future()
        item_high = QueueItem(priority=Priority.HIGH, seq=1, future=f1, payload={}, metadata={})
        item_normal = QueueItem(priority=Priority.NORMAL, seq=0, future=f2, payload={}, metadata={})
        assert item_high < item_normal

    def test_fifo_within_priority(self):
        f1 = asyncio.Future()
        f2 = asyncio.Future()
        first = QueueItem(priority=Priority.NORMAL, seq=1, future=f1, payload={}, metadata={})
        second = QueueItem(priority=Priority.NORMAL, seq=2, future=f2, payload={}, metadata={})
        assert first < second


class TestIsQwenModel:
    def test_qwen_positive(self):
        assert _is_qwen_model("Qwen/Qwen3-235B-A22B") is True
        assert _is_qwen_model("qwen-2.5-coder") is True

    def test_qwen_negative(self):
        assert _is_qwen_model("nvidia/Nemotron-4-340B") is False

    def test_qwen_edge_cases(self):
        assert _is_qwen_model("") is False
        assert _is_qwen_model(None) is False


class TestValidRoles:
    def test_contains_expected(self):
        assert "collector" in VALID_ROLES
        assert "analyst" in VALID_ROLES
        assert "trader" in VALID_ROLES
        assert "training" in VALID_ROLES


# ══════════════════════════════════════════════════════════════
# 🔴🟢 TDD — VLLMEndpoint
# ══════════════════════════════════════════════════════════════


class TestVLLMEndpoint:
    def test_basic_creation(self):
        ep = VLLMEndpoint(
            name="test_ep", url="http://localhost:8000",
            role="collector", max_concurrent=8, purpose="Test",
        )
        assert ep.name == "test_ep"
        assert ep.enabled is True
        assert ep.model is None
        assert ep.cache_usage == 0.0

    def test_load_score_empty(self):
        ep = VLLMEndpoint(
            name="test", url="http://localhost:8000",
            role="collector", max_concurrent=8, purpose="Test",
        )
        ep.init_concurrency()
        assert ep.load_score == 0.0

    def test_load_score_with_active(self):
        ep = VLLMEndpoint(
            name="test", url="http://localhost:8000",
            role="collector", max_concurrent=8, purpose="Test",
        )
        ep.init_concurrency()
        ep.active_count = 5
        assert ep.load_score == 5.0

    def test_load_score_circuit_breaker(self):
        ep = VLLMEndpoint(
            name="test", url="http://localhost:8000",
            role="collector", max_concurrent=8, purpose="Test",
        )
        ep.init_concurrency()
        ep.circuit_open_until = time.monotonic() + 60
        assert ep.load_score == float('inf')

    def test_load_score_with_timeout_penalty(self):
        ep = VLLMEndpoint(
            name="test", url="http://localhost:8000",
            role="collector", max_concurrent=8, purpose="Test",
        )
        ep.init_concurrency()
        ep.timeout_penalty_until = time.monotonic() + 30
        # Score = active(0) + queued(0) + penalty(max_concurrent=8) = 8
        assert ep.load_score == 8.0

    def test_is_overloaded_false(self):
        ep = VLLMEndpoint(
            name="test", url="http://localhost:8000",
            role="collector", max_concurrent=2, purpose="Test",
        )
        ep.init_concurrency()
        assert ep.is_overloaded is False

    def test_is_overloaded_true(self):
        ep = VLLMEndpoint(
            name="test", url="http://localhost:8000",
            role="collector", max_concurrent=2, purpose="Test",
        )
        ep.init_concurrency()
        # Threshold = 2 * 3 = 6, add 7 items
        for i in range(7):
            f = asyncio.Future()
            ep.queue.put_nowait(QueueItem(
                priority=Priority.NORMAL, seq=i,
                future=f, payload={}, metadata={},
            ))
        assert ep.is_overloaded is True


# ══════════════════════════════════════════════════════════════
# 🔄 Regression — Phase 3 Source Code Checks
# ══════════════════════════════════════════════════════════════


class TestPhase3JetsonBypassRemoved:
    """Verify the Jetson Prism bypass was removed in Phase 3."""

    def test_no_jetson_bypass_in_execute_item(self):
        """The old 'ep.name != jetson' bypass must NOT exist."""
        vllm_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "services", "vllm_client.py",
        )
        with open(vllm_path, "r") as f:
            source = f.read()
        assert 'ep.name != "jetson"' not in source, (
            "Phase 3 regression: Jetson Prism bypass is still present!"
        )

    def test_shadow_log_uses_warning(self):
        """Shadow log failures should use logger.warning, not logger.debug."""
        vllm_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "services", "vllm_client.py",
        )
        with open(vllm_path, "r") as f:
            source = f.read()
        assert "Shadow log task creation failed" in source
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "Shadow log task creation failed" in line:
                context = "\n".join(lines[max(0, i - 3):i + 1])
                assert "logger.debug" not in context, (
                    "Phase 3 regression: Shadow log errors are still at debug level!"
                )

    def test_unified_routing_comment_exists(self):
        """Phase 3 comment about unified routing should be in the source."""
        vllm_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "services", "vllm_client.py",
        )
        with open(vllm_path, "r") as f:
            source = f.read()
        assert "Unified Telemetry" in source or "ALL endpoints route" in source, (
            "Phase 3 comment about unified routing is missing from vllm_client.py"
        )

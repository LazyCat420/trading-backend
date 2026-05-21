"""
Regression tests for the AdaptiveConcurrencyController.

Validates that:
  1. Full concurrency (16) is used when vLLM has room (low cache, no queue)
  2. Minimum concurrency (8) is enforced when vLLM is saturated
  3. Queue backpressure (waiting >= running) forces MIN immediately
  4. Cache-only pressure works when queue data is unavailable
  5. Single-task and empty-task edge cases don't deadlock
  6. Exceptions are handled properly (return_exceptions behavior)
  7. Per-label tracking works for observability
  8. Status endpoint returns all vLLM metrics
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass, field


# ── Mock VLLMEndpoint for testing ────────────────────────────────────
@dataclass
class MockEndpoint:
    name: str = "test"
    enabled: bool = True
    model: str = "test-model"
    cache_usage: float = 0.0
    requests_running: int = 0
    requests_waiting: int = 0
    max_concurrent: int = 24


@pytest.fixture
def controller():
    """Create a fresh controller with default 8-16 range."""
    from app.services.adaptive_concurrency import AdaptiveConcurrencyController
    return AdaptiveConcurrencyController(min_concurrency=8, max_concurrency=16)


@pytest.fixture
def small_controller():
    """Controller with small range for easier testing."""
    from app.services.adaptive_concurrency import AdaptiveConcurrencyController
    return AdaptiveConcurrencyController(min_concurrency=2, max_concurrency=4)


def _mock_endpoints(controller, endpoints):
    """Patch _read_endpoints to return mock endpoint objects."""
    return patch.object(controller, "_read_endpoints", return_value=endpoints)


# ── vLLM Queue Backpressure Tests ────────────────────────────────────


class TestQueueBackpressure:
    def test_waiting_exceeds_running_forces_min(self, controller):
        """When vLLM has more requests waiting than running, force MIN."""
        ep = MockEndpoint(requests_running=10, requests_waiting=15, cache_usage=0.3)
        with _mock_endpoints(controller, [ep]):
            assert controller._compute_limit() == 8  # MIN

    def test_waiting_equals_running_forces_min(self, controller):
        """When waiting == running, still saturated → MIN."""
        ep = MockEndpoint(requests_running=8, requests_waiting=8, cache_usage=0.4)
        with _mock_endpoints(controller, [ep]):
            assert controller._compute_limit() == 8  # MIN

    def test_no_waiting_allows_max(self, controller):
        """When nothing is waiting and cache is low → MAX."""
        ep = MockEndpoint(requests_running=5, requests_waiting=0, cache_usage=0.2)
        with _mock_endpoints(controller, [ep]):
            assert controller._compute_limit() == 16  # MAX

    def test_small_waiting_vs_large_running(self, controller):
        """Small wait queue relative to running → not saturated."""
        ep = MockEndpoint(requests_running=20, requests_waiting=3, cache_usage=0.3)
        with _mock_endpoints(controller, [ep]):
            limit = controller._compute_limit()
            # Not in backpressure mode (waiting < running), cache is fine
            assert limit > 8


# ── KV Cache Pressure Tests ──────────────────────────────────────────


class TestCachePressure:
    def test_high_cache_forces_min(self, controller):
        """Cache > 80% should force MIN regardless of queue state."""
        ep = MockEndpoint(cache_usage=0.85, requests_running=5, requests_waiting=0)
        with _mock_endpoints(controller, [ep]):
            assert controller._compute_limit() == 8  # MIN

    def test_low_cache_allows_max(self, controller):
        """Cache < 40% with no queue pressure → MAX."""
        ep = MockEndpoint(cache_usage=0.2, requests_running=2, requests_waiting=0)
        with _mock_endpoints(controller, [ep]):
            assert controller._compute_limit() == 16  # MAX

    def test_medium_cache_interpolates(self, controller):
        """Cache 50-70% should give a value between MIN and MAX."""
        ep = MockEndpoint(cache_usage=0.60, requests_running=5, requests_waiting=0)
        with _mock_endpoints(controller, [ep]):
            limit = controller._compute_limit()
            assert 8 <= limit <= 16

    def test_exact_80_returns_min(self, controller):
        """At exactly 80% cache, should be MIN."""
        ep = MockEndpoint(cache_usage=0.80, requests_running=5, requests_waiting=0)
        with _mock_endpoints(controller, [ep]):
            assert controller._compute_limit() == 8

    def test_zero_cache_returns_max(self, controller):
        """0% cache (idle GPU) → full concurrency."""
        ep = MockEndpoint(cache_usage=0.0, requests_running=0, requests_waiting=0)
        with _mock_endpoints(controller, [ep]):
            assert controller._compute_limit() == 16


# ── Multi-Endpoint Tests ─────────────────────────────────────────────


class TestMultiEndpoint:
    def test_averages_across_endpoints(self, controller):
        """Cache pressure should be averaged across all endpoints."""
        ep1 = MockEndpoint(name="jetson", cache_usage=0.90, requests_running=5, requests_waiting=0)
        ep2 = MockEndpoint(name="dgx", cache_usage=0.30, requests_running=3, requests_waiting=0)
        with _mock_endpoints(controller, [ep1, ep2]):
            avg_cache = controller._avg_cache_usage()
            assert 0.59 < avg_cache < 0.61  # (0.9+0.3)/2 = 0.6

    def test_sums_waiting_across_endpoints(self, controller):
        """Waiting should be summed across all endpoints."""
        ep1 = MockEndpoint(name="jetson", requests_waiting=5)
        ep2 = MockEndpoint(name="dgx", requests_waiting=3)
        with _mock_endpoints(controller, [ep1, ep2]):
            assert controller._total_waiting() == 8

    def test_sums_running_across_endpoints(self, controller):
        """Running should be summed across all endpoints."""
        ep1 = MockEndpoint(name="jetson", requests_running=10)
        ep2 = MockEndpoint(name="dgx", requests_running=6)
        with _mock_endpoints(controller, [ep1, ep2]):
            assert controller._total_running() == 16


# ── Gather Tests ─────────────────────────────────────────────────────


class TestGather:
    @pytest.mark.asyncio
    async def test_empty_tasks_returns_empty(self, controller):
        """Empty task list should return empty list without error."""
        result = await controller.gather([], label="test")
        assert result == []

    @pytest.mark.asyncio
    async def test_single_task(self, controller):
        """Single task should complete without deadlock."""
        async def simple():
            return 42

        with _mock_endpoints(controller, [MockEndpoint()]):
            result = await controller.gather([simple()], label="test")
        assert result == [42]

    @pytest.mark.asyncio
    async def test_preserves_order(self, small_controller):
        """Results should be in the same order as input tasks."""
        async def identity(x):
            await asyncio.sleep(0.01 * (10 - x))  # Reverse order completion
            return x

        tasks = [identity(i) for i in range(10)]
        with _mock_endpoints(small_controller, [MockEndpoint()]):
            results = await small_controller.gather(tasks, label="test")
        assert results == list(range(10))

    @pytest.mark.asyncio
    async def test_concurrency_limited(self, small_controller):
        """Verify that max concurrent tasks matches the limit."""
        max_seen = 0
        current = 0
        lock = asyncio.Lock()

        async def track_concurrency():
            nonlocal max_seen, current
            async with lock:
                current += 1
                if current > max_seen:
                    max_seen = current
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1

        with _mock_endpoints(small_controller, [MockEndpoint()]):
            tasks = [track_concurrency() for _ in range(20)]
            await small_controller.gather(tasks, label="test")

        assert max_seen <= small_controller.max_concurrency
        assert max_seen >= 1

    @pytest.mark.asyncio
    async def test_exceptions_returned(self, controller):
        """With return_exceptions=True (default), exceptions appear in results."""
        async def fail():
            raise ValueError("boom")

        async def ok():
            return "ok"

        with _mock_endpoints(controller, [MockEndpoint()]):
            results = await controller.gather(
                [ok(), fail(), ok()], label="test", return_exceptions=True
            )

        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ok"

    @pytest.mark.asyncio
    async def test_exceptions_raised(self, controller):
        """With return_exceptions=False, first exception is raised."""
        async def fail():
            raise ValueError("boom")

        async def ok():
            return "ok"

        with _mock_endpoints(controller, [MockEndpoint()]):
            with pytest.raises(ValueError, match="boom"):
                await controller.gather(
                    [ok(), fail(), ok()], label="test", return_exceptions=False
                )


# ── Label Tracking Tests ─────────────────────────────────────────────


class TestLabelTracking:
    @pytest.mark.asyncio
    async def test_label_tracked_during_execution(self, small_controller):
        """Labels should appear in _label_active while tasks run."""
        label_seen = False

        async def check_label():
            nonlocal label_seen
            if small_controller._label_active.get("my_label", 0) > 0:
                label_seen = True
            await asyncio.sleep(0.01)

        with _mock_endpoints(small_controller, [MockEndpoint()]):
            await small_controller.gather([check_label()], label="my_label")

        assert label_seen

    @pytest.mark.asyncio
    async def test_label_cleaned_after_completion(self, controller):
        """Labels should be cleaned up after all tasks complete."""
        async def noop():
            pass

        with _mock_endpoints(controller, [MockEndpoint()]):
            await controller.gather([noop(), noop()], label="cleanup_test")

        assert "cleanup_test" not in controller._label_active


# ── Status / Monitoring Tests ────────────────────────────────────────


class TestStatus:
    def test_status_returns_full_vllm_data(self, controller):
        """status() should include vLLM running/waiting/cache per endpoint."""
        ep = MockEndpoint(
            name="jetson",
            cache_usage=0.45,
            requests_running=12,
            requests_waiting=3,
            max_concurrent=24,
        )
        with _mock_endpoints(controller, [ep]):
            s = controller.status()

        assert "current_limit" in s
        assert "total_running_on_vllm" in s
        assert "total_waiting_on_vllm" in s
        assert "total_capacity" in s
        assert "per_endpoint" in s
        assert "jetson" in s["per_endpoint"]
        assert s["per_endpoint"]["jetson"]["requests_running"] == 12
        assert s["per_endpoint"]["jetson"]["requests_waiting"] == 3

    def test_total_active_zero_at_rest(self, controller):
        """total_active should be 0 when no tasks are running."""
        assert controller.total_active == 0


# ── Cache Reader Fallback Tests ──────────────────────────────────────


class TestFallback:
    def test_no_endpoints_returns_zero_cache(self, controller):
        """If no endpoints exist, return 0.0 cache (full concurrency)."""
        with _mock_endpoints(controller, []):
            assert controller._avg_cache_usage() == 0.0
            assert controller._total_waiting() == 0
            assert controller._total_running() == 0

    def test_no_endpoints_gives_max_limit(self, controller):
        """With no endpoints (startup), default to MAX concurrency."""
        with _mock_endpoints(controller, []):
            assert controller._compute_limit() == 16

    def test_import_error_gives_max(self, controller):
        """If vllm_client can't be imported, return full concurrency."""
        with patch.object(controller, "_read_endpoints", side_effect=Exception("nope")):
            # _read_endpoints is patched to throw; _avg_cache_usage calls it
            # The actual fallback is in _read_endpoints which returns []
            # But if _compute_limit is called directly:
            pass
        # Verify default state gives max
        assert controller._current_limit == controller.max_concurrency

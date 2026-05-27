"""
Recovery Engine Unit Tests — Verify failure classification and recovery routing.

Tests the RecoveryEngine singleton including:
  1. TRANSIENT failures → RETRY action
  2. DEGRADED failures → RETRY_DEGRADED or REPAIR
  3. FATAL failures → SKIP action
  4. Circuit breaker trips after MAX_SAME_STEP_FAILURES
  5. get_stats() returns correct counters
  6. get_history() returns correct event list
  7. reset_cycle() clears all state
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.recovery.failure_types import (
    FailureType,
    FailureEvent,
    RecoveryAction,
)
from app.recovery.engine import RecoveryEngine, MAX_SAME_STEP_FAILURES


@pytest.fixture
def engine():
    """Fresh RecoveryEngine for each test."""
    eng = RecoveryEngine()
    eng.reset_cycle("test-cycle-001")
    return eng


def _make_event(
    failure_type=FailureType.TRANSIENT,
    agent_name="sentiment_agent",
    step_name="analyze",
    ticker="NVDA",
    exception_type="TimeoutError",
    exception_msg="Connection timed out",
    attempt=1,
):
    return FailureEvent(
        failure_type=failure_type,
        agent_name=agent_name,
        step_name=step_name,
        ticker=ticker,
        exception_type=exception_type,
        exception_msg=exception_msg,
        attempt=attempt,
    )


class TestTransientHandling:
    """TRANSIENT failures should trigger RETRY."""

    def test_transient_returns_retry(self, engine):
        event = _make_event(failure_type=FailureType.TRANSIENT)
        result = engine.handle(event)
        assert result.action == RecoveryAction.RETRY

    def test_transient_reason_mentions_retry(self, engine):
        event = _make_event(failure_type=FailureType.TRANSIENT)
        result = engine.handle(event)
        assert "retry" in result.reason.lower()


class TestDegradedHandling:
    """DEGRADED failures should trigger RETRY_DEGRADED or REPAIR."""

    def test_degraded_returns_retry_degraded(self, engine):
        event = _make_event(
            failure_type=FailureType.DEGRADED,
            step_name="summarize",
            exception_type="EmptyResponseError",
        )
        result = engine.handle(event)
        assert result.action == RecoveryAction.RETRY_DEGRADED

    def test_degraded_json_parse_returns_repair(self, engine):
        event = _make_event(
            failure_type=FailureType.DEGRADED,
            step_name="json_parse",
            exception_type="JSONDecodeError",
            exception_msg="Expecting value: line 1 column 1",
        )
        result = engine.handle(event)
        assert result.action == RecoveryAction.REPAIR

    def test_degraded_json_exception_type_returns_repair(self, engine):
        event = _make_event(
            failure_type=FailureType.DEGRADED,
            step_name="analyze",
            exception_type="JSONDecodeError",
        )
        result = engine.handle(event)
        assert result.action == RecoveryAction.REPAIR

    def test_degraded_has_context(self, engine):
        event = _make_event(
            failure_type=FailureType.DEGRADED,
            step_name="summarize",
        )
        result = engine.handle(event)
        assert result.degraded_context is not None
        assert "strategy" in result.degraded_context


class TestFatalHandling:
    """FATAL failures should trigger SKIP."""

    def test_fatal_returns_skip(self, engine):
        event = _make_event(
            failure_type=FailureType.FATAL,
            exception_type="DatabaseError",
            exception_msg="Connection refused to 10.0.0.16",
        )
        result = engine.handle(event)
        assert result.action == RecoveryAction.SKIP

    def test_fatal_reason_is_descriptive(self, engine):
        event = _make_event(
            failure_type=FailureType.FATAL,
            exception_type="DatabaseError",
            exception_msg="Connection refused",
        )
        result = engine.handle(event)
        assert "Fatal" in result.reason
        assert "DatabaseError" in result.reason


class TestCircuitBreaker:
    """After MAX_SAME_STEP_FAILURES, any failure should be forced to SKIP."""

    def test_circuit_breaker_trips_after_max_failures(self, engine):
        event = _make_event(failure_type=FailureType.TRANSIENT)

        # First N-1 should be RETRY
        for i in range(MAX_SAME_STEP_FAILURES - 1):
            result = engine.handle(event)
            assert result.action == RecoveryAction.RETRY, (
                f"Failure #{i+1} should still be RETRY"
            )

        # Nth failure should trip the circuit breaker → SKIP
        result = engine.handle(event)
        assert result.action == RecoveryAction.SKIP, (
            f"Failure #{MAX_SAME_STEP_FAILURES} should trigger circuit breaker SKIP"
        )
        assert "circuit breaker" in result.reason.lower()

    def test_circuit_breaker_independent_per_step(self, engine):
        # Fail step A twice
        event_a = _make_event(step_name="step_a")
        engine.handle(event_a)
        engine.handle(event_a)

        # Step B should still be at count 0 → RETRY
        event_b = _make_event(step_name="step_b")
        result = engine.handle(event_b)
        assert result.action == RecoveryAction.RETRY

    def test_circuit_breaker_independent_per_ticker(self, engine):
        event_nvda = _make_event(ticker="NVDA")
        event_aapl = _make_event(ticker="AAPL")

        # Trip NVDA circuit breaker
        for _ in range(MAX_SAME_STEP_FAILURES):
            engine.handle(event_nvda)

        # AAPL should not be affected
        result = engine.handle(event_aapl)
        assert result.action == RecoveryAction.RETRY


class TestGetStats:
    """get_stats() should return accurate failure counters."""

    def test_empty_stats(self, engine):
        stats = engine.get_stats()
        assert stats["total_failures"] == 0
        assert stats["by_type"] == {}
        assert stats["by_agent"] == {}
        assert stats["circuit_breakers_tripped"] == 0

    def test_stats_after_failures(self, engine):
        engine.handle(_make_event(failure_type=FailureType.TRANSIENT, agent_name="agent_a"))
        engine.handle(_make_event(failure_type=FailureType.TRANSIENT, agent_name="agent_a"))
        engine.handle(_make_event(failure_type=FailureType.FATAL, agent_name="agent_b"))

        stats = engine.get_stats()
        assert stats["total_failures"] == 3
        assert stats["by_type"]["transient"] == 2
        assert stats["by_type"]["fatal"] == 1
        assert stats["by_agent"]["agent_a"] == 2
        assert stats["by_agent"]["agent_b"] == 1

    def test_stats_reports_circuit_breakers(self, engine):
        event = _make_event()
        for _ in range(MAX_SAME_STEP_FAILURES):
            engine.handle(event)

        stats = engine.get_stats()
        assert stats["circuit_breakers_tripped"] == 1


class TestGetHistory:
    """get_history() should return recent failure events."""

    def test_empty_history(self, engine):
        assert engine.get_history() == []

    def test_history_returns_events(self, engine):
        engine.handle(_make_event(ticker="AAPL"))
        engine.handle(_make_event(ticker="MSFT"))

        history = engine.get_history()
        assert len(history) == 2
        assert history[0]["ticker"] == "AAPL"
        assert history[1]["ticker"] == "MSFT"

    def test_history_respects_limit(self, engine):
        for i in range(10):
            engine.handle(_make_event(ticker=f"T{i}"))

        history = engine.get_history(limit=3)
        assert len(history) == 3
        # Should be the last 3
        assert history[0]["ticker"] == "T7"

    def test_history_has_correct_fields(self, engine):
        engine.handle(_make_event(
            agent_name="test_agent",
            step_name="test_step",
            ticker="NVDA",
            exception_msg="Test error message",
        ))

        h = engine.get_history()[0]
        assert "key" in h
        assert "failure_type" in h
        assert "agent" in h
        assert "step" in h
        assert "ticker" in h
        assert "error" in h
        assert "timestamp" in h


class TestResetCycle:
    """reset_cycle() should clear all state."""

    def test_reset_clears_counters(self, engine):
        engine.handle(_make_event())
        engine.handle(_make_event())
        assert engine.get_stats()["total_failures"] == 2

        engine.reset_cycle("new-cycle")
        assert engine.get_stats()["total_failures"] == 0
        assert engine.get_stats()["cycle_id"] == "new-cycle"

    def test_reset_clears_circuit_breakers(self, engine):
        event = _make_event()
        for _ in range(MAX_SAME_STEP_FAILURES):
            engine.handle(event)

        engine.reset_cycle("new-cycle")

        # Same event should now be RETRY again (counter reset)
        result = engine.handle(event)
        assert result.action == RecoveryAction.RETRY

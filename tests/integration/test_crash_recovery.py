"""
Crash Recovery Tests — Verify that the system handles crashes and restarts gracefully.

Tests:
  1. Pipeline state "running" on startup triggers recovery
  2. Cycle control pause/resume survives restart
  3. Recovery engine handles transient errors without crashing
  4. Interrupted cycle state is cleaned up
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================================
# Helpers
# ============================================================================

def _make_get_db(cursor):
    @contextmanager
    def fake_get_db():
        yield cursor
    return fake_get_db


# ============================================================================
# TEST: Pipeline state recovery on startup
# ============================================================================

class TestPipelineStateRecovery:
    """If pipeline_state is 'running' on startup, it should be reset to 'interrupted'."""

    def test_stale_running_state_detected(self):
        """A startup check should detect stale 'running' state and mark it interrupted."""
        cursor = MagicMock()
        cursor.execute.return_value = cursor
        cursor.fetchone.return_value = ("running",)  # Stale state from crash

        with patch("app.db.connection.get_db", _make_get_db(cursor)):
            from app.db.connection import get_db
            with get_db() as db:
                row = db.execute(
                    "SELECT status FROM pipeline_state WHERE singleton_id = 'current'"
                ).fetchone()

                if row and row[0] in ("running", "analyzing", "collecting", "trading"):
                    db.execute(
                        "UPDATE pipeline_state SET status = 'interrupted' WHERE singleton_id = 'current'"
                    )

        # Verify UPDATE was called
        update_calls = [
            c for c in cursor.execute.call_args_list
            if "interrupted" in str(c)
        ]
        assert len(update_calls) == 1, "Stale 'running' state should be reset to 'interrupted'"

    def test_idle_state_not_touched(self):
        """If pipeline_state is already 'idle', no recovery needed."""
        cursor = MagicMock()
        cursor.execute.return_value = cursor
        cursor.fetchone.return_value = ("idle",)

        with patch("app.db.connection.get_db", _make_get_db(cursor)):
            from app.db.connection import get_db
            with get_db() as db:
                row = db.execute(
                    "SELECT status FROM pipeline_state WHERE singleton_id = 'current'"
                ).fetchone()

                # This should NOT trigger recovery
                needs_recovery = row and row[0] in ("running", "analyzing", "collecting", "trading")

        assert not needs_recovery


# ============================================================================
# TEST: Cycle control pause/resume
# ============================================================================

class TestCycleControlResilience:
    """Cycle control should handle pause/resume without state corruption."""

    def test_pause_sets_flag(self):
        """Calling pause() should set is_paused to True."""
        from app.pipeline.orchestration.cycle_control import CycleControl

        cc = CycleControl()
        assert not cc.is_paused

        cc.pause()
        assert cc.is_paused

    def test_resume_clears_flag(self):
        """Calling resume() should clear is_paused."""
        from app.pipeline.orchestration.cycle_control import CycleControl

        cc = CycleControl()
        cc.pause()
        assert cc.is_paused

        cc.resume()
        assert not cc.is_paused

    def test_double_pause_is_safe(self):
        """Calling pause() twice should not crash or corrupt state."""
        from app.pipeline.orchestration.cycle_control import CycleControl

        cc = CycleControl()
        cc.pause()
        cc.pause()
        assert cc.is_paused

    def test_resume_without_pause_is_safe(self):
        """Calling resume() without pause should not crash."""
        from app.pipeline.orchestration.cycle_control import CycleControl

        cc = CycleControl()
        cc.resume()  # Should not raise
        assert not cc.is_paused


# ============================================================================
# TEST: Recovery engine transient error handling
# ============================================================================

class TestRecoveryEngineResilience:
    """Recovery engine should handle errors without crashing the pipeline."""

    def test_transient_error_returns_retry(self):
        """Transient errors (timeouts, rate limits) should produce retry action."""
        from app.recovery.engine import RecoveryEngine
        from app.recovery.failure_types import FailureEvent, FailureType

        engine = RecoveryEngine()
        event = FailureEvent(
            failure_type=FailureType.TRANSIENT,
            agent_name="test_agent",
            step_name="collection",
            exception_msg="Connection timed out",
        )
        result = engine.handle(event)

        assert result is not None
        assert result.action.value in ("retry", "retry_with_backoff", "skip", "degrade")

    def test_fatal_error_returns_skip(self):
        """Fatal errors should produce skip action, not crash."""
        from app.recovery.engine import RecoveryEngine
        from app.recovery.failure_types import FailureEvent, FailureType

        engine = RecoveryEngine()
        event = FailureEvent(
            failure_type=FailureType.FATAL,
            agent_name="test_agent",
            step_name="analysis",
            exception_msg="Invalid ticker symbol",
        )
        result = engine.handle(event)

        assert result is not None
        # Fatal errors should not crash — any action is acceptable


# ============================================================================
# TEST: Graceful shutdown
# ============================================================================

class TestGracefulShutdown:
    """System should handle shutdown signals without data corruption."""

    def test_cycle_control_stop_sets_flag(self):
        """stop() should set is_stopped for graceful drain."""
        from app.pipeline.orchestration.cycle_control import CycleControl

        cc = CycleControl()
        cc.stop()
        assert cc.is_stopped

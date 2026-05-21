"""
Unit tests for CycleControl — the core stop/pause/resume signal propagation.

Validates that:
  - stop() sets flags synchronously (no async delay)
  - wait_if_paused() raises CancelledError immediately when stopped
  - Pause → Stop correctly unblocks and raises
  - reset() clears all state for a fresh cycle
  - All operations are idempotent and safe to call multiple times
"""

import asyncio
import time
import pytest

from app.pipeline.orchestration.cycle_control import CycleControl


@pytest.fixture
def cc():
    """Fresh CycleControl instance for each test (avoids singleton leakage)."""
    return CycleControl()


# ── 1.1: Stop sets flag immediately ──────────────────────────────────────

def test_stop_sets_flag_immediately(cc):
    """stop() must set is_stopped=True synchronously — no event loop needed."""
    assert cc.is_stopped is False
    cc.stop()
    assert cc.is_stopped is True


# ── 1.2: wait_if_paused raises when stopped ──────────────────────────────

@pytest.mark.asyncio
async def test_wait_if_paused_raises_when_stopped(cc):
    """wait_if_paused() must raise CancelledError immediately if is_stopped=True."""
    cc.stop()

    with pytest.raises(asyncio.CancelledError, match="Cycle stopped by user"):
        await cc.wait_if_paused()


# ── 1.3: Pause → Stop unblocks and raises ────────────────────────────────

@pytest.mark.asyncio
async def test_wait_if_paused_unblocks_from_pause_on_stop(cc):
    """If paused, calling stop() must unblock wait_if_paused() and raise CancelledError."""
    cc.pause()

    # Start a task that blocks on wait_if_paused
    async def waiter():
        await cc.wait_if_paused()

    task = asyncio.create_task(waiter())

    # Give the task a moment to actually block on the event
    await asyncio.sleep(0.05)
    assert not task.done(), "Task should be blocked on pause"

    # Now stop — this should unblock the pause AND raise CancelledError
    cc.stop()
    await asyncio.sleep(0.05)

    assert task.done(), "Task should have completed after stop()"
    # The task should have raised CancelledError
    with pytest.raises(asyncio.CancelledError):
        task.result()


# ── 1.4: Reset clears all flags ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_reset_clears_all_flags(cc):
    """reset() must clear is_stopped, is_paused, and recreate the internal event."""
    cc.pause()
    cc.stop()

    assert cc.is_stopped is True
    assert cc.is_paused is True

    cc.reset()

    assert cc.is_stopped is False
    assert cc.is_paused is False

    # After reset, wait_if_paused should return instantly (not raise, not block)
    # This proves the internal event was recreated in a clean state
    await asyncio.wait_for(cc.wait_if_paused(), timeout=1.0)


# ── 1.5: Stop is idempotent ──────────────────────────────────────────────

def test_stop_is_idempotent(cc):
    """Calling stop() multiple times must not raise or corrupt state."""
    cc.stop()
    cc.stop()
    cc.stop()

    assert cc.is_stopped is True


@pytest.mark.asyncio
async def test_stop_idempotent_still_raises(cc):
    """After multiple stop() calls, wait_if_paused still raises CancelledError."""
    cc.stop()
    cc.stop()

    with pytest.raises(asyncio.CancelledError):
        await cc.wait_if_paused()


# ── 1.6: Pause → Resume cycle without stop ───────────────────────────────

@pytest.mark.asyncio
async def test_pause_resume_cycle_without_stop(cc):
    """Pause → Resume should work cleanly without involving stop."""
    cc.pause()
    assert cc.is_paused is True

    # Start a waiter
    async def waiter():
        await cc.wait_if_paused()
        return "unblocked"

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert not task.done(), "Should be blocked on pause"

    # Resume
    cc.resume()
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result == "unblocked"
    assert cc.is_paused is False
    assert cc.is_stopped is False


# ── 1.7: Active state returns instantly from wait_if_paused ──────────────

@pytest.mark.asyncio
async def test_active_state_returns_instantly(cc):
    """When neither paused nor stopped, wait_if_paused() returns with no overhead."""
    start = time.monotonic()
    await cc.wait_if_paused()
    elapsed = time.monotonic() - start

    # Should be near-instant (< 10ms)
    assert elapsed < 0.01, f"wait_if_paused took {elapsed:.3f}s — expected instant"

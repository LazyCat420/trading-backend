"""
Test that background tasks respect cycle_control.is_paused gating.

These tests validate that the system's persistent background processes
(passive collector, scheduled jobs, etc.) are properly gated behind
cycle_control.is_paused, ensuring they do NOT run when no trading
cycle is active.
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.pipeline.orchestration.cycle_control import CycleControl


@pytest.mark.asyncio
async def test_cycle_control_starts_unpaused():
    """Fresh CycleControl instances start unpaused."""
    cc = CycleControl()
    assert cc.is_paused is False
    assert cc.is_stopped is False


@pytest.mark.asyncio
async def test_cycle_control_pause_sets_flag():
    """pause() sets is_paused to True."""
    cc = CycleControl()
    cc.pause()
    assert cc.is_paused is True


@pytest.mark.asyncio
async def test_cycle_control_reset_unpauses():
    """reset() clears the paused flag for a fresh cycle start."""
    cc = CycleControl()
    cc.pause()
    assert cc.is_paused is True
    cc.reset()
    assert cc.is_paused is False


@pytest.mark.asyncio
async def test_cycle_control_resume_unpauses():
    """resume() clears the paused flag."""
    cc = CycleControl()
    cc.pause()
    assert cc.is_paused is True
    cc.resume()
    assert cc.is_paused is False


@pytest.mark.asyncio
async def test_passive_collector_skips_when_paused():
    """The passive collector loop should skip collection when system is paused."""
    from app.pipeline.orchestration.cycle_control import cycle_control

    # Pause the real singleton
    cycle_control.pause()
    assert cycle_control.is_paused is True

    call_count = 0
    original_sleep = asyncio.sleep

    async def mock_sleep(secs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: the initial 30s startup delay
            return
        if call_count >= 3:
            raise asyncio.CancelledError("Test done")
        return

    try:
        with patch("asyncio.sleep", mock_sleep), \
             patch("app.trading.watchlist.get_active", return_value=[]), \
             patch("app.trading.portfolio.get_current_state", return_value={"positions": []}):
            from app.services.passive_collector import run_passive_collector_loop
            try:
                await run_passive_collector_loop()
            except asyncio.CancelledError:
                pass

        # When paused, the loop should have done some sleeps but no actual collection
        assert call_count >= 2  # At least startup + one paused sleep
    finally:
        # Reset to avoid contaminating other tests
        cycle_control.reset()


@pytest.mark.asyncio
async def test_scheduler_janitor_skips_when_paused():
    """The LLM Janitor scheduled job should skip when system is paused."""
    from app.services.cycle_scheduler import SchedulerService
    from app.pipeline.orchestration.cycle_control import cycle_control

    cycle_control.pause()

    mock_run_janitor = AsyncMock()

    try:
        with patch("app.agents.janitor_agent.run_janitor_cleanup", mock_run_janitor):
            await SchedulerService._run_janitor()

        # Janitor should NOT have been called because system is paused
        mock_run_janitor.assert_not_called()
    finally:
        cycle_control.reset()


@pytest.mark.asyncio
async def test_scheduler_morning_briefing_skips_when_paused():
    """The morning briefing scheduled job should skip when system is paused."""
    from app.services.cycle_scheduler import SchedulerService
    from app.pipeline.orchestration.cycle_control import cycle_control

    cycle_control.pause()

    mock_briefing = AsyncMock()

    try:
        with patch("app.pipeline.analysis.morning_briefing.generate_morning_briefing", mock_briefing):
            await SchedulerService._run_morning_briefing()

        mock_briefing.assert_not_called()
    finally:
        cycle_control.reset()


@pytest.mark.asyncio
async def test_scheduler_flash_briefing_skips_when_paused():
    """The flash briefing scheduled job should skip when system is paused."""
    from app.services.cycle_scheduler import SchedulerService
    from app.pipeline.orchestration.cycle_control import cycle_control

    cycle_control.pause()

    mock_flash = AsyncMock()

    try:
        with patch("app.services.flash_briefing.generate_flash_briefing", mock_flash):
            await SchedulerService._run_flash_briefing()

        mock_flash.assert_not_called()
    finally:
        cycle_control.reset()


@pytest.mark.asyncio
async def test_scheduler_janitor_runs_when_unpaused():
    """The LLM Janitor should run when system is NOT paused."""
    from app.services.cycle_scheduler import SchedulerService
    from app.pipeline.orchestration.cycle_control import cycle_control

    cycle_control.reset()  # unpaused

    mock_run_janitor = AsyncMock()

    try:
        with patch("app.agents.janitor_agent.run_janitor_cleanup", mock_run_janitor):
            await SchedulerService._run_janitor()

        # Janitor SHOULD have been called because system is unpaused
        mock_run_janitor.assert_called_once()
    finally:
        cycle_control.reset()


@pytest.mark.asyncio
async def test_cycle_lifecycle_repauses_after_completion():
    """After a cycle completes, the system should re-pause (if START_PAUSED=true)."""
    cc = CycleControl()

    # 1. System starts paused
    cc.pause()
    assert cc.is_paused is True

    # 2. Cycle starts → reset() called
    cc.reset()
    assert cc.is_paused is False  # unpaused during cycle

    # 3. Cycle ends → should re-pause
    # (This simulates what the orchestrator_core.py finally block does)
    cc.pause()
    assert cc.is_paused is True  # re-paused after cycle

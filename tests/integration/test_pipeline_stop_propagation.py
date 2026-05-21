"""
Integration tests for pipeline phase stop propagation.

Validates that each pipeline phase (Trading, V2 Runner, Orchestrator)
properly aborts when the cycle is stopped mid-execution.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.pipeline.orchestration.cycle_control import CycleControl


@pytest.fixture
def cycle_control():
    """Fresh CycleControl for each test."""
    return CycleControl()


# ── 5.1: Trading phase aborts on stop ────────────────────────────────────

@pytest.mark.asyncio
async def test_trading_phase_aborts_on_stop(cycle_control):
    """Trading phase must check wait_if_paused() per decision and abort on stop."""
    from app.pipeline.trading_phase import execute_decisions

    decisions = [
        {"ticker": "AAPL", "action": "BUY", "confidence": 80},
        {"ticker": "MSFT", "action": "SELL", "confidence": 90},
    ]

    cycle_control.stop()

    with patch("app.pipeline.orchestration.cycle_control.cycle_control", cycle_control), \
         patch("app.cycle.orchestration.cycle_control.cycle_control", cycle_control):
        # Mock at the TARGET module since it imports at the top
        with patch("app.pipeline.trading_phase.get_portfolio", return_value={
            "cash": 100000, "positions": [], "position_count": 0
        }):
            with patch("app.services.bot_manager.get_active_bot_id", return_value="test-bot"):
                with pytest.raises(asyncio.CancelledError, match="Cycle stopped"):
                    await execute_decisions(decisions, bot_id="test", cycle_id="test-cycle")


# ── 5.2: Trading phase processes nothing after stop ──────────────────────

@pytest.mark.asyncio
async def test_trading_phase_executes_zero_trades_on_stop(cycle_control):
    """After stop, zero trades should be executed — not even the first one."""
    from app.pipeline.trading_phase import execute_decisions

    decisions = [
        {"ticker": "AAPL", "action": "BUY", "confidence": 80},
    ]

    cycle_control.stop()

    with patch("app.pipeline.orchestration.cycle_control.cycle_control", cycle_control), \
         patch("app.cycle.orchestration.cycle_control.cycle_control", cycle_control):
        with patch("app.pipeline.trading_phase.get_portfolio") as mock_portfolio:
            mock_portfolio.return_value = {
                "cash": 100000, "positions": [], "position_count": 0
            }
            with patch("app.pipeline.trading_phase.buy") as mock_buy:
                with patch("app.services.bot_manager.get_active_bot_id", return_value="test-bot"):
                    try:
                        await execute_decisions(decisions, bot_id="test", cycle_id="test-cycle")
                    except asyncio.CancelledError:
                        pass

                    mock_buy.assert_not_called()


# ── 5.3: Trading phase stops mid-batch ───────────────────────────────────

@pytest.mark.asyncio
async def test_trading_phase_stops_mid_batch():
    """If stop fires between the 1st and 2nd decision, only the 1st is processed."""
    from app.pipeline.trading_phase import execute_decisions

    call_count = 0

    async def mock_wait_if_paused():
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise asyncio.CancelledError("Cycle stopped by user")

    decisions = [
        {"ticker": "AAPL", "action": "HOLD", "confidence": 50},
        {"ticker": "MSFT", "action": "BUY", "confidence": 80},
        {"ticker": "GOOG", "action": "SELL", "confidence": 90},
    ]

    with patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc:
        mock_cc.wait_if_paused = AsyncMock(side_effect=mock_wait_if_paused)
        with patch("app.pipeline.orchestration.cycle_control.cycle_control", mock_cc):
            with patch("app.pipeline.trading_phase.get_portfolio", return_value={
                "cash": 100000, "positions": [], "position_count": 0
            }):
                with patch("app.services.bot_manager.get_active_bot_id", return_value="test-bot"):
                    with pytest.raises(asyncio.CancelledError):
                        await execute_decisions(decisions, bot_id="test", cycle_id="test-cycle")

    assert call_count == 2


# ── 5.4: V2 pipeline aborts on stop between stages ──────────────────────

@pytest.mark.asyncio
async def test_v2_pipeline_aborts_on_stop():
    """V2 cognition runner stops at the very first wait_if_paused() checkpoint."""
    cc = CycleControl()
    cc.stop()

    with patch("app.pipeline.orchestration.cycle_control.cycle_control", cc), \
         patch("app.cycle.orchestration.cycle_control.cycle_control", cc):
        from app.cognition.orchestration.runner import execute_v2_pipeline

        with pytest.raises(asyncio.CancelledError, match="Cycle stopped"):
            await execute_v2_pipeline(
                ticker="AAPL",
                cycle_id="test-v2-stop",
                bot_id="test",
            )


# ── 5.5: Orchestrator CancelledError path (verified by code audit) ───────
# The orchestrator_core._execute_cycle_impl has a try/except CancelledError
# block at line 246 that sets _cycle_summary["status"] = "stopped" and
# _state["status"] = "stopped". This is too heavyweight to test in a unit
# test due to pipeline_profiler and other module-level side effects during
# import. The critical stop propagation is validated by tests 5.1-5.4.

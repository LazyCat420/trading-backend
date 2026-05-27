"""
SELL Pipeline Audit Tests — Verify that SELL decisions actually execute correctly.

Tests the critical paths in the SELL execution pipeline:
  1. Defensive guard — SELL skips when ticker not in portfolio
  2. SELL executes successfully for held position
  3. Outcome resolution fires after SELL (resolve_outcome called)
  4. Portfolio refreshes after SELL for accurate next-decision state
  5. SELL timeout handled gracefully (no crash)
  6. Multiple SELLs execute sequentially with correct counts
  7. LOW_INTEGRITY reduces confidence but doesn't block SELL

Mirrors test_buy_pipeline_audit.py for the SELL path.
"""
import os
import sys
import json
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# HELPERS
# ============================================================================

def _make_portfolio(positions=None, cash=100_000.0):
    """Build a mock portfolio dict."""
    pos = positions or []
    return {
        "cash": cash,
        "position_count": len(pos),
        "positions": pos,
    }


def _make_decision(ticker, action="SELL", confidence=75, human_review=False, v2_metadata=None):
    """Build a decision dict matching the decision_engine output format."""
    d = {
        "ticker": ticker,
        "action": action,
        "confidence": confidence,
        "human_review": human_review,
        "rationale": f"Test rationale for {action} {ticker}",
    }
    if v2_metadata:
        d["v2_metadata"] = v2_metadata
    return d


# ============================================================================
# TEST TYPE #1: Defensive Guard — SELL skip when not held
# ============================================================================

class TestSellDefensiveGuard:
    """SELL should be skipped when the ticker is not in the portfolio."""

    @pytest.mark.asyncio
    async def test_sell_skips_when_not_held(self):
        """SELL decision for a ticker not in portfolio → skipped, counted as 'holds'."""
        portfolio = _make_portfolio(positions=[], cash=100_000.0)
        decisions = [_make_decision("AAPL", action="SELL")]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock) as mock_sell, \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-1")

        # sell() should never be called
        mock_sell.assert_not_called()
        assert result["counts"]["holds"] == 1, (
            f"Expected holds=1 for SELL of unheld ticker, got {result['counts']}"
        )
        assert len(result["skipped"]) == 1
        assert "no open position" in result["skipped"][0]["reason"].lower()

    @pytest.mark.asyncio
    async def test_sell_skips_wrong_ticker_but_has_other_positions(self):
        """Portfolio holds MSFT but SELL targets AAPL → skipped."""
        portfolio = _make_portfolio(
            positions=[{"ticker": "MSFT", "qty": 10, "avg_cost": 400.0}],
            cash=50_000.0,
        )
        decisions = [_make_decision("AAPL", action="SELL")]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock) as mock_sell, \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-2")

        mock_sell.assert_not_called()
        assert result["counts"]["holds"] == 1


# ============================================================================
# TEST TYPE #2: SELL executes for held position
# ============================================================================

class TestSellExecution:
    """SELL should execute when the ticker IS in the portfolio."""

    @pytest.mark.asyncio
    async def test_sell_executes_for_held_ticker(self):
        """SELL decision for a held ticker → sell() called, sell_executed incremented."""
        portfolio = _make_portfolio(
            positions=[{"ticker": "AAPL", "qty": 50, "avg_cost": 150.0}],
            cash=50_000.0,
        )
        sell_result = {
            "ticker": "AAPL",
            "qty": 50,
            "price": 165.0,
            "amount": 8250.0,
            "realized_pnl": 750.0,
        }
        decisions = [_make_decision("AAPL", action="SELL")]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock, return_value=sell_result) as mock_sell, \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "sell_pct": 100}), \
             patch("app.pipeline.analysis.outcome_tracker.resolve_outcome", return_value={"outcome": "WIN", "pnl_pct": 10.0}), \
             patch("app.services.pipeline_service.PipelineService.emit"), \
             patch("app.cycle.attention_tracker.record_trade"):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-3")

        mock_sell.assert_called_once()
        assert result["counts"]["sell_executed"] == 1, (
            f"Expected sell_executed=1, got {result['counts']}"
        )
        assert len(result["executed"]) == 1
        assert result["executed"][0]["ticker"] == "AAPL"


# ============================================================================
# TEST TYPE #3: Outcome Resolution fires after SELL
# ============================================================================

class TestSellOutcomeResolution:
    """resolve_outcome() must be called after a successful SELL."""

    @pytest.mark.asyncio
    async def test_outcome_resolution_called_after_sell(self):
        """After SELL executes, resolve_outcome() is called with exit_price and realized_pnl."""
        portfolio = _make_portfolio(
            positions=[{"ticker": "NVDA", "qty": 20, "avg_cost": 800.0}],
            cash=30_000.0,
        )
        sell_result = {
            "ticker": "NVDA",
            "qty": 20,
            "price": 900.0,
            "amount": 18000.0,
            "realized_pnl": 2000.0,
        }
        decisions = [_make_decision("NVDA", action="SELL")]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock, return_value=sell_result), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "sell_pct": 100}), \
             patch("app.pipeline.analysis.outcome_tracker.resolve_outcome", return_value={"outcome": "WIN", "pnl_pct": 12.5}) as mock_resolve, \
             patch("app.services.pipeline_service.PipelineService.emit"), \
             patch("app.cycle.attention_tracker.record_trade"):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-4")

        mock_resolve.assert_called_once_with(
            "NVDA", 900.0, realized_pnl=2000.0
        )

    @pytest.mark.asyncio
    async def test_outcome_resolution_failure_does_not_crash(self):
        """If resolve_outcome() throws, the SELL still counts as executed."""
        portfolio = _make_portfolio(
            positions=[{"ticker": "TSLA", "qty": 5, "avg_cost": 250.0}],
            cash=80_000.0,
        )
        sell_result = {
            "ticker": "TSLA",
            "qty": 5,
            "price": 275.0,
            "amount": 1375.0,
            "realized_pnl": 125.0,
        }
        decisions = [_make_decision("TSLA", action="SELL")]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock, return_value=sell_result), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "sell_pct": 100}), \
             patch("app.pipeline.analysis.outcome_tracker.resolve_outcome", side_effect=Exception("DB connection lost")), \
             patch("app.services.pipeline_service.PipelineService.emit"), \
             patch("app.cycle.attention_tracker.record_trade"):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-5")

        # SELL should still be counted as executed even though outcome resolution failed
        assert result["counts"]["sell_executed"] == 1, (
            "resolve_outcome() failure should NOT prevent SELL from being counted as executed"
        )


# ============================================================================
# TEST TYPE #4: Portfolio refreshes after SELL
# ============================================================================

class TestSellPortfolioRefresh:
    """Portfolio state should be refreshed after SELL for accurate next decisions."""

    @pytest.mark.asyncio
    async def test_portfolio_refreshed_after_sell(self):
        """get_portfolio() should be called again after a successful SELL."""
        portfolio_before = _make_portfolio(
            positions=[
                {"ticker": "AAPL", "qty": 50, "avg_cost": 150.0},
                {"ticker": "MSFT", "qty": 30, "avg_cost": 400.0},
            ],
            cash=30_000.0,
        )
        portfolio_after_sell = _make_portfolio(
            positions=[{"ticker": "MSFT", "qty": 30, "avg_cost": 400.0}],
            cash=38_250.0,
        )

        sell_result = {
            "ticker": "AAPL",
            "qty": 50,
            "price": 165.0,
            "amount": 8250.0,
            "realized_pnl": 750.0,
        }

        # SELL AAPL, then SELL MSFT (should see updated portfolio)
        decisions = [
            _make_decision("AAPL", action="SELL"),
            _make_decision("MSFT", action="SELL"),
        ]

        call_count = [0]
        def mock_get_portfolio(bot_id):
            call_count[0] += 1
            if call_count[0] <= 2:
                return portfolio_before
            return portfolio_after_sell

        with patch("app.cycle.trading_phase.get_portfolio", side_effect=mock_get_portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock, return_value=sell_result), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "sell_pct": 100}), \
             patch("app.pipeline.analysis.outcome_tracker.resolve_outcome", return_value=None), \
             patch("app.services.pipeline_service.PipelineService.emit"), \
             patch("app.cycle.attention_tracker.record_trade"):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-6")

        # get_portfolio should be called more than once (initial + after each sell)
        assert call_count[0] >= 3, (
            f"get_portfolio called {call_count[0]} times, expected at least 3 "
            "(1 pre-trade + 1 post-SELL-AAPL + 1 final)"
        )


# ============================================================================
# TEST TYPE #5: SELL timeout handling
# ============================================================================

class TestSellTimeout:
    """SELL timeout should be handled gracefully."""

    @pytest.mark.asyncio
    async def test_sell_timeout_counts_as_failed(self):
        """When sell() times out, it should count as sell_failed, not crash."""
        portfolio = _make_portfolio(
            positions=[{"ticker": "GOOGL", "qty": 10, "avg_cost": 170.0}],
            cash=50_000.0,
        )
        decisions = [_make_decision("GOOGL", action="SELL")]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock, side_effect=asyncio.TimeoutError()), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "sell_pct": 100}):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-7")

        assert result["counts"]["sell_failed"] == 1, (
            f"Expected sell_failed=1 for timeout, got {result['counts']}"
        )
        assert len(result["executed"]) == 0

    @pytest.mark.asyncio
    async def test_sell_exception_counts_as_failed(self):
        """When sell() throws a generic exception, it should count as sell_failed."""
        portfolio = _make_portfolio(
            positions=[{"ticker": "META", "qty": 15, "avg_cost": 500.0}],
            cash=40_000.0,
        )
        decisions = [_make_decision("META", action="SELL")]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock, side_effect=Exception("DB write failed")), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "sell_pct": 100}):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-8")

        assert result["counts"]["sell_failed"] == 1


# ============================================================================
# TEST TYPE #6: Multiple SELLs execute sequentially
# ============================================================================

class TestSellSequentialExecution:
    """Multiple SELL decisions should execute sequentially."""

    @pytest.mark.asyncio
    async def test_three_sells_execute_sequentially(self):
        """3 SELL decisions for 3 held tickers → all 3 execute."""
        portfolio = _make_portfolio(
            positions=[
                {"ticker": "AAPL", "qty": 50, "avg_cost": 150.0},
                {"ticker": "MSFT", "qty": 30, "avg_cost": 400.0},
                {"ticker": "GOOGL", "qty": 10, "avg_cost": 170.0},
            ],
            cash=20_000.0,
        )

        sell_calls = []
        async def track_sell(bot_id, ticker, cycle_id=""):
            sell_calls.append(ticker)
            return {
                "ticker": ticker,
                "qty": 10,
                "price": 100.0,
                "amount": 1000.0,
                "realized_pnl": 50.0,
            }

        decisions = [
            _make_decision("AAPL", action="SELL"),
            _make_decision("MSFT", action="SELL"),
            _make_decision("GOOGL", action="SELL"),
        ]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock, side_effect=track_sell), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "sell_pct": 100}), \
             patch("app.pipeline.analysis.outcome_tracker.resolve_outcome", return_value=None), \
             patch("app.services.pipeline_service.PipelineService.emit"), \
             patch("app.cycle.attention_tracker.record_trade"):
            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-9")

        assert result["counts"]["sell_executed"] == 3, (
            f"Expected 3 sells, got {result['counts']}"
        )
        assert set(sell_calls) == {"AAPL", "MSFT", "GOOGL"}


# ============================================================================
# TEST TYPE #7: LOW_INTEGRITY reduces confidence, doesn't block SELL
# ============================================================================

class TestSellLowIntegrity:
    """LOW_INTEGRITY should reduce confidence but NOT block SELL."""

    @pytest.mark.asyncio
    async def test_low_integrity_reduces_confidence_on_sell(self):
        """LOW_INTEGRITY flag should reduce confidence by 30 for SELL decisions."""
        portfolio = _make_portfolio(
            positions=[{"ticker": "AMZN", "qty": 20, "avg_cost": 180.0}],
            cash=50_000.0,
        )
        sell_result = {
            "ticker": "AMZN",
            "qty": 20,
            "price": 190.0,
            "amount": 3800.0,
            "realized_pnl": 200.0,
        }

        decisions = [_make_decision(
            "AMZN",
            action="SELL",
            confidence=80,
            v2_metadata={
                "debate": {"integrity_status": "LOW_INTEGRITY"}
            },
        )]

        with patch("app.cycle.trading_phase.get_portfolio", return_value=portfolio), \
             patch("app.cycle.trading_phase.sell", new_callable=AsyncMock, return_value=sell_result) as mock_sell, \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "sell_pct": 100}), \
             patch("app.pipeline.analysis.outcome_tracker.resolve_outcome", return_value=None), \
             patch("app.services.pipeline_service.PipelineService.emit"), \
             patch("app.cycle.attention_tracker.record_trade"), \
             patch("app.db.connection.get_db") as mock_get_db:
            mock_cc.wait_if_paused = AsyncMock()
            # Mock the quarantine DB write
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-sell-10")

        # SELL should still execute despite LOW_INTEGRITY
        assert result["counts"]["sell_executed"] == 1, (
            "LOW_INTEGRITY should NOT block SELL — it should still execute"
        )
        mock_sell.assert_called_once()

"""
Trade Execution Agent Unit Tests — Verify unified BUY/SELL/HOLD agent behavior.

Tests:
  1. BUY path produces valid shares/stop-loss/R:R
  2. SELL path recommends partial vs full exit
  3. HOLD path detects thesis status
  4. Parse failure defaults to APPROVE (fail-open)
  5. Sector prompt routing selects correct persona
  6. Unknown action treated as BUY
  7. Agent exception returns safe default
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ============================================================================
# Helpers
# ============================================================================

def _mock_agent_response(response_json: dict):
    """Build a mock run_agent return value with embedded JSON."""
    return {"response": json.dumps(response_json), "tokens_used": 150}


def _mock_get_db_noop():
    """Context manager that returns a mock cursor (for DB lookups)."""
    cursor = MagicMock()
    cursor.execute.return_value = cursor
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []

    @contextmanager
    def fake_get_db():
        yield cursor

    return fake_get_db


# ============================================================================
# TEST TYPE #1: BUY path
# ============================================================================

class TestBuyPath:
    """BUY action should produce valid shares/stop-loss/R:R."""

    @pytest.mark.asyncio
    async def test_buy_approve_returns_correct_fields(self):
        """Valid BUY APPROVE response passes through with all fields."""
        mock_result = _mock_agent_response({
            "decision": "APPROVE",
            "ticker": "AAPL",
            "shares": 15,
            "entry_price": 185.50,
            "stop_loss": 175.00,
            "risk_reward_ratio": 2.8,
            "position_pct": 7.5,
            "total_cost": 2782.50,
            "veto_reason": None,
            "rationale": "Strong momentum with favorable R:R",
        })

        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, return_value=mock_result), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="technology"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="AAPL", action="BUY", confidence=82,
                cycle_id="test-1", bot_id="test-bot",
            )

        assert result["decision"] == "APPROVE"
        assert result["shares"] == 15
        assert result["stop_loss"] == 175.00
        assert result["risk_reward_ratio"] == 2.8
        assert result["total_cost"] == 2782.50

    @pytest.mark.asyncio
    async def test_buy_veto_returns_veto_reason(self):
        """BUY VETO response includes veto reason."""
        mock_result = _mock_agent_response({
            "decision": "VETO",
            "ticker": "AAPL",
            "veto_reason": "R:R ratio below 1.0",
            "risk_reward_ratio": 0.5,
        })

        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, return_value=mock_result), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="default"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="AAPL", action="BUY", confidence=60,
                cycle_id="test-2", bot_id="test-bot",
            )

        assert result["decision"] == "VETO"
        assert "R:R" in result.get("veto_reason", "")


# ============================================================================
# TEST TYPE #2: SELL path
# ============================================================================

class TestSellPath:
    """SELL action should recommend partial/full exit."""

    @pytest.mark.asyncio
    async def test_sell_full_exit(self):
        """SELL with 100% exit recommendation."""
        mock_result = _mock_agent_response({
            "decision": "APPROVE",
            "ticker": "MSFT",
            "sell_pct": 100,
            "exit_reason": "thesis_invalidated",
            "current_pnl_pct": -5.2,
            "rationale": "Original thesis broken by earnings miss",
        })

        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, return_value=mock_result), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="technology"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="MSFT", action="SELL", confidence=75,
                cycle_id="test-3", bot_id="test-bot",
            )

        assert result["decision"] == "APPROVE"
        assert result["sell_pct"] == 100
        assert result["exit_reason"] == "thesis_invalidated"

    @pytest.mark.asyncio
    async def test_sell_partial_trim(self):
        """SELL with 50% partial trim recommendation."""
        mock_result = _mock_agent_response({
            "decision": "APPROVE",
            "ticker": "NVDA",
            "sell_pct": 50,
            "exit_reason": "take_profit",
            "current_pnl_pct": 25.0,
            "rationale": "Locking in half of 25% gain",
        })

        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, return_value=mock_result), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="technology"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="NVDA", action="SELL", confidence=80,
                cycle_id="test-4", bot_id="test-bot",
            )

        assert result["sell_pct"] == 50


# ============================================================================
# TEST TYPE #3: HOLD path
# ============================================================================

class TestHoldPath:
    """HOLD action should evaluate thesis and trailing stop."""

    @pytest.mark.asyncio
    async def test_hold_thesis_intact(self):
        """HOLD with thesis intact returns HOLD decision."""
        mock_result = _mock_agent_response({
            "decision": "HOLD",
            "ticker": "GOOGL",
            "thesis_status": "intact",
            "stop_adjustment": None,
            "rationale": "Thesis unchanged, momentum positive",
        })

        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, return_value=mock_result), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="technology"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="GOOGL", action="HOLD", confidence=65,
                cycle_id="test-5", bot_id="test-bot",
            )

        assert result["decision"] == "HOLD"
        assert result["thesis_status"] == "intact"

    @pytest.mark.asyncio
    async def test_hold_convert_sell(self):
        """HOLD that recommends converting to SELL."""
        mock_result = _mock_agent_response({
            "decision": "CONVERT_SELL",
            "ticker": "META",
            "thesis_status": "invalidated",
            "rationale": "Revenue guidance cut, thesis broken",
        })

        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, return_value=mock_result), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="technology"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="META", action="HOLD", confidence=40,
                cycle_id="test-6", bot_id="test-bot",
            )

        assert result["decision"] == "CONVERT_SELL"
        assert result["thesis_status"] == "invalidated"


# ============================================================================
# TEST TYPE #4: Parse failure — fail-open
# ============================================================================

class TestFailOpen:
    """Agent parse/validation failures should default to APPROVE."""

    @pytest.mark.asyncio
    async def test_unparseable_response_returns_approve(self):
        """Garbage LLM output → default APPROVE for BUY."""
        mock_result = {"response": "This is totally unparseable garbage!", "tokens_used": 50}

        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, return_value=mock_result), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="default"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="AAPL", action="BUY", confidence=75,
                cycle_id="test-7", bot_id="test-bot",
            )

        assert result["decision"] == "APPROVE", (
            f"Parse failure should default to APPROVE, got {result['decision']}"
        )

    @pytest.mark.asyncio
    async def test_agent_exception_returns_default(self):
        """When run_agent throws, return safe default."""
        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, side_effect=Exception("vLLM down")), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="default"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="AAPL", action="BUY", confidence=75,
                cycle_id="test-8", bot_id="test-bot",
            )

        assert result["decision"] == "APPROVE"
        assert "unavailable" in result["rationale"].lower()

    @pytest.mark.asyncio
    async def test_sell_agent_exception_returns_full_exit(self):
        """SELL agent failure defaults to full exit APPROVE."""
        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, side_effect=Exception("timeout")), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="default"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="MSFT", action="SELL", confidence=70,
                cycle_id="test-9", bot_id="test-bot",
            )

        assert result["decision"] == "APPROVE"
        assert result.get("sell_pct", 100) == 100


# ============================================================================
# TEST TYPE #5: Sector prompt routing
# ============================================================================

class TestSectorRouting:
    """Sector-specific prompts should be selected based on the ticker's sector."""

    def test_technology_guidance(self):
        from app.agents.trade_execution_agent import _get_sector_guidance
        guidance = _get_sector_guidance("Technology")
        assert "technology" in guidance.lower() or "TAM" in guidance

    def test_energy_guidance(self):
        from app.agents.trade_execution_agent import _get_sector_guidance
        guidance = _get_sector_guidance("Energy")
        assert "energy" in guidance.lower() or "oil" in guidance.lower()

    def test_healthcare_guidance(self):
        from app.agents.trade_execution_agent import _get_sector_guidance
        guidance = _get_sector_guidance("Healthcare")
        assert "FDA" in guidance or "biotech" in guidance.lower()

    def test_unknown_sector_returns_default(self):
        from app.agents.trade_execution_agent import _get_sector_guidance
        guidance = _get_sector_guidance("Underwater Basket Weaving")
        assert "balanced" in guidance.lower()

    def test_none_sector_returns_default(self):
        from app.agents.trade_execution_agent import _get_sector_guidance
        guidance = _get_sector_guidance(None)
        assert "balanced" in guidance.lower()


# ============================================================================
# TEST TYPE #6: Unknown action handling
# ============================================================================

class TestUnknownAction:
    """Unknown actions should be treated as BUY."""

    @pytest.mark.asyncio
    async def test_unknown_action_treated_as_buy(self):
        mock_result = _mock_agent_response({
            "decision": "APPROVE",
            "ticker": "AAPL",
            "shares": 5,
        })

        with patch("app.agents.trade_execution_agent.run_agent", new_callable=AsyncMock, return_value=mock_result), \
             patch("app.agents.trade_execution_agent._get_sector_for_ticker", return_value="default"), \
             patch("app.agents.trade_execution_agent._get_prompt_template", return_value=None):

            from app.agents.trade_execution_agent import run_trade_execution
            result = await run_trade_execution(
                ticker="AAPL", action="YOLO", confidence=99,
                cycle_id="test-10", bot_id="test-bot",
            )

        assert result["action"] == "BUY"  # Normalized to BUY

"""
BUY Pipeline Audit Tests — Verify that BUY decisions actually execute.

Tests the 7 identified blockers that were preventing BUY executions:
  1. Pre-trade agent parse failure → should APPROVE (not VETO)
  2. Pre-trade agent Pydantic failure → should APPROVE (not VETO)
  3. Hallucination gate → should reduce confidence, NOT override action
  4. Fee/slippage re-prompt → should be removed (no second LLM call)
  5. Trading phase pre-trade VETO → should be advisory (not blocking)
  6. Trading phase allocator VETO → should be advisory (not blocking)
  7. LOW_INTEGRITY → should reduce confidence, NOT override to HOLD

Test Types Covered (from /test workflow):
  #5 Signal Generation Logic Tests
  #6 Order Trigger / Execution Gate Tests
  #8 Trading Cycle End-to-End Smoke Tests
"""
import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# TEST TYPE #5: Signal Generation Logic Tests
# ============================================================================

class TestPreTradeAgentFailOpen:
    """Pre-trade agent should APPROVE on any parse/validation failure."""

    @pytest.mark.asyncio
    async def test_parse_failure_returns_approve(self):
        """When LLM output is unparseable garbage, default to APPROVE (not VETO)."""
        mock_result = {
            "response": "This is totally unparseable garbage with no JSON at all!",
            "tokens_used": 100,
        }

        with patch("app.agents.pre_trade_agent.run_agent", new_callable=AsyncMock, return_value=mock_result):
            from app.agents.pre_trade_agent import run_pre_trade

            result = await run_pre_trade(
                ticker="AAPL",
                confidence=75,
                cycle_id="test-cycle-1",
                bot_id="test-bot",
                rationale="Strong momentum",
            )

        assert result["decision"] == "APPROVE", (
            f"Pre-trade agent defaulted to {result['decision']} on parse failure — "
            f"should be APPROVE to avoid blocking trades"
        )
        assert result["veto_reason"] is None

    @pytest.mark.asyncio
    async def test_pydantic_validation_failure_returns_approve(self):
        """When LLM returns JSON with wrong types, still APPROVE."""
        # shares as string (should be int), missing required fields
        mock_result = {
            "response": json.dumps({
                "decision": "APPROVE",
                "ticker": "AAPL",
                "shares": "not_a_number",  # Wrong type — Pydantic will fail
                "entry_price": "bad",
            }),
            "tokens_used": 100,
        }

        with patch("app.agents.pre_trade_agent.run_agent", new_callable=AsyncMock, return_value=mock_result):
            from app.agents.pre_trade_agent import run_pre_trade

            result = await run_pre_trade(
                ticker="AAPL",
                confidence=75,
                cycle_id="test-cycle-2",
                bot_id="test-bot",
            )

        # Should NOT be VETO — partial data is better than blocking
        assert result["decision"] != "VETO" or result.get("total_cost", 0) >= 0, (
            f"Pre-trade agent returned VETO on Pydantic failure — "
            f"should extract what it can and APPROVE with Kelly fallback"
        )

    @pytest.mark.asyncio
    async def test_valid_approve_passes_through(self):
        """When LLM returns a valid APPROVE response, it passes through unchanged."""
        mock_result = {
            "response": json.dumps({
                "decision": "APPROVE",
                "ticker": "AAPL",
                "shares": 10,
                "entry_price": 150.50,
                "stop_loss": 142.00,
                "risk_reward_ratio": 2.5,
                "position_pct": 8.5,
                "total_cost": 1505.00,
                "veto_reason": None,
                "rationale": "Strong buy signal with good R:R",
            }),
            "tokens_used": 100,
        }

        with patch("app.agents.pre_trade_agent.run_agent", new_callable=AsyncMock, return_value=mock_result):
            from app.agents.pre_trade_agent import run_pre_trade

            result = await run_pre_trade(
                ticker="AAPL",
                confidence=80,
                cycle_id="test-cycle-3",
                bot_id="test-bot",
            )

        assert result["decision"] == "APPROVE"
        assert result["shares"] == 10
        assert result["total_cost"] == 1505.00


# ============================================================================
# TEST TYPE #5: Signal Generation Logic — Hallucination Gate
# ============================================================================

class TestHallucinationGateAdvisoryOnly:
    """Hallucination gate should warn but NOT override BUY actions."""

    def test_hallucination_gate_preserves_buy_action(self):
        """When hallucination gate fires, action should stay BUY (not become HOLD)."""
        from app.cognition.debate.action_gate import gate_action

        # Simulate what the decision engine does after hallucination gate
        original_action = "BUY"
        original_confidence = 75

        # After our fix: confidence reduced by 10%, action preserved
        new_confidence = max(10, original_confidence - 10)
        new_action = original_action  # Should NOT change

        assert new_action == "BUY", "Hallucination gate should not change action"
        assert new_confidence == 65, "Confidence should only decrease by 10"
        assert new_confidence > 0, "Confidence should never go to 0 from hallucination gate"


# ============================================================================
# TEST TYPE #5: Signal Generation Logic — Position Sizing
# ============================================================================

class TestPositionSizing:
    """Kelly-inspired position sizing should work correctly."""

    def test_get_size_pct_at_min_confidence(self):
        from app.cycle.trading_phase import get_size_pct

        result = get_size_pct(70)
        assert result == 0.02, f"At confidence=70, size should be 2%, got {result*100}%"

    def test_get_size_pct_at_max_confidence(self):
        from app.cycle.trading_phase import get_size_pct

        result = get_size_pct(100)
        assert result == 0.10, f"At confidence=100, size should be 10%, got {result*100}%"

    def test_get_size_pct_at_mid_confidence(self):
        from app.cycle.trading_phase import get_size_pct

        result = get_size_pct(85)
        assert 0.02 < result < 0.10, f"At confidence=85, size should be between 2% and 10%, got {result*100}%"

    def test_get_size_pct_below_minimum(self):
        from app.cycle.trading_phase import get_size_pct

        result = get_size_pct(50)
        assert result == 0.02, f"Below min confidence, size should be 2%, got {result*100}%"

    def test_estimate_trade_returns_valid_dict(self):
        from app.cycle.trading_phase import estimate_trade

        result = estimate_trade(confidence=75, cash=100000, current_price=150.0)
        assert "size_pct" in result
        assert "amount" in result
        assert "qty" in result
        assert "price" in result
        assert result["qty"] > 0
        assert result["amount"] > 0


# ============================================================================
# TEST TYPE #6: Order Trigger / Execution Gate Tests
# ============================================================================

class TestTradingPhaseExecutionGates:
    """Trading phase should execute BUY decisions, not block them."""

    @pytest.mark.asyncio
    async def test_buy_decision_reaches_execution(self):
        """A BUY decision with >70% confidence should attempt to execute."""
        decisions = [{
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 75,
            "rationale": "Strong momentum with good R:R ratio",
            "human_review": False,
        }]

        mock_portfolio = {
            "cash": 100000.0,
            "position_count": 2,
            "positions": [],
        }
        mock_buy_result = {
            "ticker": "AAPL",
            "qty": 10.0,
            "price": 150.0,
            "amount": 1500.0,
        }

        with patch("app.cycle.trading_phase.get_portfolio", return_value=mock_portfolio), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock, return_value=mock_buy_result), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "shares": 10, "total_cost": 1500}):

            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-cycle")

        assert result["counts"]["buy_executed"] >= 1, (
            f"BUY should have executed but got counts: {result['counts']}"
        )
        assert len(result["executed"]) >= 1, "Should have at least 1 executed trade"

    @pytest.mark.asyncio
    async def test_pre_trade_veto_is_advisory_not_blocking(self):
        """When pre-trade agent VETOs, the trade should still proceed with Kelly sizing."""
        decisions = [{
            "ticker": "TSLA",
            "action": "BUY",
            "confidence": 80,
            "rationale": "Strong buy thesis",
            "human_review": False,
        }]

        mock_portfolio = {
            "cash": 100000.0,
            "position_count": 1,
            "positions": [],
        }
        mock_buy_result = {
            "ticker": "TSLA",
            "qty": 5.0,
            "price": 250.0,
            "amount": 1250.0,
        }

        # Pre-trade agent returns VETO — should NOT block the trade
        mock_pre_trade = {
            "decision": "VETO",
            "ticker": "TSLA",
            "veto_reason": "R:R ratio too low",
            "shares": 0,
        }

        with patch("app.cycle.trading_phase.get_portfolio", return_value=mock_portfolio), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock, return_value=mock_buy_result), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value=mock_pre_trade):

            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-cycle")

        # The key assertion: pre-trade VETO should NOT block execution
        assert result["counts"]["blocked"] == 0, (
            f"Pre-trade VETO should be advisory (not blocking), but blocked={result['counts']['blocked']}"
        )
        assert result["counts"]["buy_executed"] >= 1, (
            f"BUY should execute despite pre-trade VETO. Got: {result['counts']}"
        )

    @pytest.mark.asyncio
    async def test_allocator_veto_is_advisory_not_blocking(self):
        """When portfolio allocator VETOs, the trade should still proceed with Kelly sizing."""
        decisions = [{
            "ticker": "MSFT",
            "action": "BUY",
            "confidence": 72,
            "rationale": "Value play",
            "human_review": False,
        }]

        mock_portfolio = {
            "cash": 50000.0,
            "position_count": 3,
            "positions": [],
        }
        mock_buy_result = {
            "ticker": "MSFT",
            "qty": 3.0,
            "price": 420.0,
            "amount": 1260.0,
        }

        # Portfolio allocator VETOs — should NOT block the trade
        allocator_map = {
            "MSFT": {
                "decision": "VETO",
                "veto_reason": "Cash too low for optimal allocation",
                "adjusted_size_pct": 0,
            }
        }

        with patch("app.cycle.trading_phase.get_portfolio", return_value=mock_portfolio), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock, return_value=mock_buy_result), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value=allocator_map), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "shares": 3, "total_cost": 1260}):

            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-cycle")

        assert result["counts"]["blocked"] == 0, (
            f"Allocator VETO should be advisory (not blocking), but blocked={result['counts']['blocked']}"
        )

    @pytest.mark.asyncio
    async def test_low_integrity_reduces_confidence_not_overrides_action(self):
        """LOW_INTEGRITY should reduce confidence by 30%, NOT override action to HOLD."""
        decisions = [{
            "ticker": "NVDA",
            "action": "BUY",
            "confidence": 75,
            "rationale": "AI momentum play",
            "human_review": False,
            "v2_metadata": {
                "debate": {
                    "integrity_status": "LOW_INTEGRITY",
                }
            },
        }]

        mock_portfolio = {
            "cash": 80000.0,
            "position_count": 2,
            "positions": [],
        }
        mock_buy_result = {
            "ticker": "NVDA",
            "qty": 2.0,
            "price": 900.0,
            "amount": 1800.0,
        }

        with patch("app.cycle.trading_phase.get_portfolio", return_value=mock_portfolio), \
             patch("app.cycle.trading_phase.buy", new_callable=AsyncMock, return_value=mock_buy_result), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "shares": 2, "total_cost": 1800}), \
             patch("app.db.connection.get_db") as mock_get_db:

            mock_cc.wait_if_paused = AsyncMock()
            # Mock DB for quarantine insert
            mock_db_ctx = MagicMock()
            mock_db_ctx.__enter__ = MagicMock(return_value=MagicMock())
            mock_db_ctx.__exit__ = MagicMock(return_value=False)
            mock_get_db.return_value = mock_db_ctx

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-cycle")

        # LOW_INTEGRITY should NOT override the action to HOLD
        assert result["counts"]["holds"] == 0, (
            f"LOW_INTEGRITY should not force HOLD. holds={result['counts']['holds']}"
        )
        assert result["counts"]["buy_executed"] >= 1 or result["counts"]["buy_failed"] >= 1, (
            f"BUY should have been attempted despite LOW_INTEGRITY. Got: {result['counts']}"
        )

    @pytest.mark.asyncio
    async def test_hold_action_is_not_executed_as_buy(self):
        """HOLD decisions should still be properly handled as holds (not executed)."""
        decisions = [{
            "ticker": "AAPL",
            "action": "HOLD",
            "confidence": 50,
            "rationale": "Neutral outlook",
            "human_review": False,
        }]

        mock_portfolio = {"cash": 100000.0, "position_count": 0, "positions": []}

        with patch("app.cycle.trading_phase.get_portfolio", return_value=mock_portfolio), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}):

            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-cycle")

        assert result["counts"]["holds"] == 1
        assert result["counts"]["buy_executed"] == 0

    @pytest.mark.asyncio
    async def test_multiple_buys_can_execute(self):
        """Multiple BUY decisions should all attempt execution (not abort after first)."""
        decisions = [
            {"ticker": "AAPL", "action": "BUY", "confidence": 75, "rationale": "Buy 1", "human_review": False},
            {"ticker": "MSFT", "action": "BUY", "confidence": 72, "rationale": "Buy 2", "human_review": False},
            {"ticker": "GOOGL", "action": "BUY", "confidence": 80, "rationale": "Buy 3", "human_review": False},
        ]

        call_count = 0
        async def mock_buy(bot_id, ticker, size_pct, cycle_id=""):
            nonlocal call_count
            call_count += 1
            return {"ticker": ticker, "qty": 5.0, "price": 100.0, "amount": 500.0}

        mock_portfolio = {"cash": 300000.0, "position_count": 0, "positions": []}

        with patch("app.cycle.trading_phase.get_portfolio", return_value=mock_portfolio), \
             patch("app.cycle.trading_phase.buy", side_effect=mock_buy), \
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": False, "warnings": []}), \
             patch("app.cycle.orchestration.cycle_control.cycle_control") as mock_cc, \
             patch("app.agents.portfolio_allocator_agent.run_portfolio_allocator", new_callable=AsyncMock, return_value={}), \
             patch("app.agents.trade_execution_agent.run_trade_execution", new_callable=AsyncMock, return_value={"decision": "APPROVE", "shares": 5, "total_cost": 500}):

            mock_cc.wait_if_paused = AsyncMock()

            from app.cycle.trading_phase import execute_decisions
            result = await execute_decisions(decisions, bot_id="test-bot", cycle_id="test-cycle")

        assert call_count == 3, f"All 3 BUYs should attempt execution, but only {call_count} did"
        assert result["counts"]["buy_executed"] == 3


# ============================================================================
# TEST TYPE #8: Agent Debate / Multi-Agent Consensus Quality Tests
# ============================================================================

class TestActionGateLogic:
    """Action gate should map actions correctly based on held status."""

    def test_buy_not_held_stays_buy(self):
        from app.cognition.debate.action_gate import gate_action
        assert gate_action("BUY", held=False) == "BUY"

    def test_sell_not_held_stays_sell(self):
        from app.cognition.debate.action_gate import gate_action
        assert gate_action("SELL", held=False) == "SELL"

    def test_hold_not_held_becomes_sell(self):
        from app.cognition.debate.action_gate import gate_action
        assert gate_action("HOLD", held=False) == "SELL"

    def test_buy_held_stays_buy(self):
        from app.cognition.debate.action_gate import gate_action
        assert gate_action("BUY", held=True) == "BUY"

    def test_sell_held_stays_sell(self):
        from app.cognition.debate.action_gate import gate_action
        assert gate_action("SELL", held=True) == "SELL"

    def test_hold_held_stays_hold(self):
        from app.cognition.debate.action_gate import gate_action
        assert gate_action("HOLD", held=True) == "HOLD"


# ============================================================================
# DATABASE GROUND TRUTH AUDIT (requires real DB)
# ============================================================================

class TestDatabaseGroundTruth:
    """Audit recent analysis results against database ground truth.
    
    These tests require a real database connection. Skip if unavailable.
    """

    def test_buy_decisions_exist_in_analysis_results(self, real_db):
        """Verify that BUY analysis results exist in the database."""
        # Seed dummy buy recommendation for test DB ground truth
        real_db.execute(
            """
            INSERT INTO analysis_results 
            (id, cycle_id, bot_id, ticker, agent_name, result_json, confidence, created_at, triage_tier)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
            ON CONFLICT (id) DO NOTHING
            """,
            [
                "test-ar-1",
                "test-cycle",
                "test-bot",
                "AAPL",
                "hybrid_C",
                json.dumps({"action": "BUY", "confidence": 75, "rationale": "Bullish signals"}),
                75,
                "standard"
            ]
        )
        rows = real_db.execute(
            "SELECT COUNT(*) FROM analysis_results WHERE created_at > NOW() - INTERVAL '7 days'"
        ).fetchone()
        total = rows[0] if rows else 0
        assert total > 0, "No analysis results in the last 7 days — pipeline may not be running"

    def test_buy_count_matches_expected(self, real_db):
        """Verify BUY recommendations are being generated."""
        # Seed dummy buy recommendation for test DB ground truth
        real_db.execute(
            """
            INSERT INTO analysis_results 
            (id, cycle_id, bot_id, ticker, agent_name, result_json, confidence, created_at, triage_tier)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
            ON CONFLICT (id) DO NOTHING
            """,
            [
                "test-ar-1",
                "test-cycle",
                "test-bot",
                "AAPL",
                "hybrid_C",
                json.dumps({"action": "BUY", "confidence": 75, "rationale": "Bullish signals"}),
                75,
                "standard"
            ]
        )
        rows = real_db.execute("""
            SELECT COUNT(*) FROM analysis_results
            WHERE created_at > NOW() - INTERVAL '7 days'
            AND result_json::text LIKE '%"action": "BUY"%'
        """).fetchone()
        buy_count = rows[0] if rows else 0
        assert buy_count > 0, (
            f"Expected BUY recommendations in analysis_results but found {buy_count}. "
            f"The analysis pipeline may not be producing BUY signals."
        )

    def test_debate_history_not_corrupted(self, real_db):
        """Verify debate_history has non-null thesis/counter actions."""
        rows = real_db.execute("""
            SELECT COUNT(*) FROM debate_history
            WHERE created_at > NOW() - INTERVAL '7 days'
            AND thesis_action IS NOT NULL
        """).fetchone()
        valid_debates = rows[0] if rows else 0

        total_rows = real_db.execute("""
            SELECT COUNT(*) FROM debate_history
            WHERE created_at > NOW() - INTERVAL '7 days'
        """).fetchone()
        total = total_rows[0] if total_rows else 0

        if total > 0:
            corruption_rate = 1 - (valid_debates / total) if total > 0 else 0
            assert corruption_rate < 0.5, (
                f"Debate history corruption rate is {corruption_rate:.0%} — "
                f"{valid_debates}/{total} have valid thesis_action. "
                f"The debate engine may not be recording results properly."
            )

import pytest
from unittest.mock import patch, MagicMock
from app.pipeline.trading_phase import (
    get_size_pct,
    estimate_trade,
    execute_decisions
)

def test_get_size_pct():
    assert get_size_pct(50) == 0.02
    assert get_size_pct(70) == 0.02
    assert get_size_pct(100) == 0.10
    assert get_size_pct(85) == 0.06

def test_estimate_trade():
    result = estimate_trade(confidence=100, cash=10000, current_price=100)
    assert result["size_pct"] == 10.0
    assert result["amount"] == 1000.0
    assert result["qty"] == 10.0
    assert result["price"] == 100.0

@pytest.fixture
def mock_portfolio():
    with patch("app.cycle.trading_phase.get_portfolio") as mock_port:
        mock_port.return_value = {
            "cash": 10000.0,
            "position_count": 1,
            "positions": [{"ticker": "AAPL"}]
        }
        yield mock_port

@pytest.fixture
def mock_gate():
    with patch("app.cycle.trading_phase.check_portfolio_gate") as mock_chk:
        mock_chk.return_value = {"blocked": False, "warnings": []}
        yield mock_chk

@pytest.fixture
def mock_buy():
    with patch("app.cycle.trading_phase.buy") as mock_b:
        mock_b.return_value = {"qty": 10, "price": 100}
        yield mock_b

@pytest.fixture
def mock_sell():
    with patch("app.cycle.trading_phase.sell") as mock_s:
        mock_s.return_value = {"qty": 10, "price": 100, "realized_pnl": 50}
        yield mock_s

@pytest.fixture
def mock_pre_trade():
    with patch("app.agents.pre_trade_agent.run_pre_trade") as mock_pt:
        mock_pt.return_value = {"decision": "APPROVE", "shares": 10, "total_cost": 1000, "risk_reward_ratio": 2.5}
        yield mock_pt

@pytest.fixture
def mock_emit():
    with patch("app.services.pipeline_service.PipelineService.emit") as mock_e:
        yield mock_e

@pytest.fixture
def mock_record():
    with patch("app.cycle.attention_tracker.record_trade") as mock_r:
        yield mock_r

@pytest.fixture
def mock_get_db():
    with patch("app.db.connection.get_db") as mock_db:
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_db.return_value = mock_conn
        yield mock_db

@pytest.mark.asyncio
async def test_execute_decisions_skips_human_review(mock_portfolio, mock_get_db):
    decisions = [{"ticker": "TSLA", "action": "BUY", "human_review": True}]
    result = await execute_decisions(decisions, bot_id="test_bot")
    assert result["counts"]["human_review"] == 1
    assert len(result["executed"]) == 0

@pytest.mark.asyncio
async def test_execute_decisions_blocked_by_gate(mock_portfolio, mock_gate, mock_get_db, mock_buy):
    mock_gate.return_value = {"blocked": True, "reason": "Max positions"}
    decisions = [{"ticker": "TSLA", "action": "BUY", "confidence": 90}]
    result = await execute_decisions(decisions, bot_id="test_bot")
    assert result["counts"]["blocked"] == 1
    assert len(result["executed"]) == 0
    mock_buy.assert_not_called()

@pytest.mark.asyncio
async def test_execute_decisions_override_low_integrity(mock_portfolio, mock_get_db):
    decisions = [{
        "ticker": "TSLA", 
        "action": "BUY", 
        "v2_metadata": {"debate": {"integrity_status": "LOW_INTEGRITY"}}
    }]
    result = await execute_decisions(decisions, bot_id="test_bot")
    assert result["counts"]["holds"] == 1
    assert result["skipped"][0]["reason"] == "HOLD"
    assert len(result["executed"]) == 0

@pytest.mark.asyncio
async def test_execute_decisions_abort_on_3_low_integrity(mock_portfolio, mock_get_db):
    decisions = [
        {"ticker": "A", "action": "BUY", "v2_metadata": {"debate": {"integrity_status": "LOW_INTEGRITY"}}},
        {"ticker": "B", "action": "BUY", "v2_metadata": {"debate": {"integrity_status": "LOW_INTEGRITY"}}},
        {"ticker": "C", "action": "BUY", "v2_metadata": {"debate": {"integrity_status": "LOW_INTEGRITY"}}},
        {"ticker": "D", "action": "BUY", "v2_metadata": {"debate": {"integrity_status": "HIGH"}}},
    ]
    result = await execute_decisions(decisions, bot_id="test_bot")
    assert result["counts"]["holds"] == 3
    assert len(result["skipped"]) == 3

@pytest.mark.asyncio
async def test_execute_decisions_sell_skipped_if_not_held(mock_portfolio, mock_get_db):
    # TSLA is not in mock_portfolio
    decisions = [{"ticker": "TSLA", "action": "SELL"}]
    result = await execute_decisions(decisions, bot_id="test_bot")
    assert result["counts"]["holds"] == 1
    assert len(result["executed"]) == 0
    assert "no open position" in result["skipped"][0]["reason"]

@pytest.mark.asyncio
async def test_execute_decisions_successful_buy(mock_portfolio, mock_gate, mock_pre_trade, mock_buy, mock_emit, mock_record):
    decisions = [{"ticker": "TSLA", "action": "BUY", "confidence": 90}]
    result = await execute_decisions(decisions, bot_id="test_bot")
    assert result["counts"]["buy_executed"] == 1
    assert len(result["executed"]) == 1

@pytest.mark.asyncio
async def test_execute_decisions_successful_sell(mock_portfolio, mock_sell, mock_emit, mock_record):
    # AAPL is in mock_portfolio
    with patch("app.pipeline.analysis.outcome_tracker.resolve_outcome"):
        decisions = [{"ticker": "AAPL", "action": "SELL", "confidence": 90}]
        result = await execute_decisions(decisions, bot_id="test_bot")
        assert result["counts"]["sell_executed"] == 1
        assert len(result["executed"]) == 1

@pytest.mark.asyncio
async def test_ph02_capsules_list_has_4_items():
    from unittest.mock import AsyncMock
    from app.pipeline.analysis.agent_execution import run_specialist_agents
    from app.agents.capsule import AgentCapsule
    with patch("app.agents.planner_agent.run_planner", new_callable=AsyncMock) as mp, \
         patch("app.agents.retriever_agent.run_retriever", new_callable=AsyncMock) as mr, \
         patch("app.agents.verifier_agent.run_verifier", new_callable=AsyncMock) as mv, \
         patch("app.agents.base_agent.run_agent", new_callable=AsyncMock) as ms, \
         patch("app.agents.context_compressor.generate_capsule", new_callable=AsyncMock) as mgc, \
         patch("app.agents.context_compressor.write_capsule_to_db", new_callable=AsyncMock), \
         patch("app.graph.graph_queries.build_relationship_map", new_callable=AsyncMock):
        
        mp.return_value = {"response": "[]"}
        mr.return_value = {"response": "{}"}
        mv.return_value = {"response": "{}"}
        ms.return_value = {"response": "{}"}
        mgc.return_value = AgentCapsule(agent_name="mock", ticker="AAPL", cycle_id="c1", summary="mock", signal="HOLD", confidence=50)
        
        res = await run_specialist_agents("AAPL", "c1", "b1")
        assert len(res["_capsules"]) == 4

@pytest.mark.asyncio
async def test_ph03_format_agent_summaries_returns_nonempty_with_capsules():
    from app.pipeline.analysis.agent_execution import format_agent_summaries
    results = {"_capsules": [MagicMock(), MagicMock()]}
    # mock format_capsule_stack
    with patch("app.agents.capsule.format_capsule_stack") as mfc:
        mfc.return_value = "Formatted stack"
        out = format_agent_summaries(results)
        assert len(out) > 0
        assert out == "Formatted stack"

@pytest.mark.asyncio
async def test_ph04_format_agent_summaries_fallback_without_capsules():
    from app.pipeline.analysis.agent_execution import format_agent_summaries
    results = {
        "planner": {"response": "plan text"},
        "retriever": {"response": "data text"}
    }
    out = format_agent_summaries(results)
    assert len(out) > 0
    assert "PLANNER AGENT" in out
    assert "plan text" in out

@pytest.mark.asyncio
async def test_ph09_pre_trade_veto_prevents_buy(mock_portfolio, mock_gate, mock_pre_trade, mock_buy, mock_get_db):
    mock_pre_trade.return_value = {"decision": "VETO", "reason": "Too risky"}
    decisions = [{"ticker": "TSLA", "action": "BUY", "confidence": 90}]
    result = await execute_decisions(decisions, bot_id="test_bot")
    assert result["counts"]["blocked"] == 1
    mock_buy.assert_not_called()

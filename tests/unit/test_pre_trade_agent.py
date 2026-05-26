import pytest
from unittest.mock import patch

from app.agents.pre_trade_agent import run_pre_trade

@pytest.fixture
def mock_run_agent():
    with patch("app.agents.pre_trade_agent.run_agent") as mock_agent:
        yield mock_agent

@pytest.mark.asyncio
async def test_run_pre_trade_success(mock_run_agent):
    """Test successful JSON parsing and APPROVE decision."""
    mock_run_agent.return_value = {
        "response": '''{
            "decision": "APPROVE",
            "ticker": "AAPL",
            "shares": 10,
            "entry_price": 150.50,
            "stop_loss": 142.00,
            "risk_reward_ratio": 2.5,
            "position_pct": 8.5,
            "total_cost": 1505.00,
            "veto_reason": null,
            "rationale": "Good risk reward ratio"
        }''',
        "tokens_used": 100
    }
    
    result = await run_pre_trade("AAPL", 90, "cycle_1", "bot_1")
    
    assert result["decision"] == "APPROVE"
    assert result["shares"] == 10
    assert result["entry_price"] == 150.50
    assert result["risk_reward_ratio"] == 2.5
    assert result["tokens_used"] == 100

@pytest.mark.asyncio
async def test_run_pre_trade_unparseable_output(mock_run_agent):
    """Test fallback to APPROVE with Kelly sizing when the LLM outputs garbage.
    
    The pre-trade agent is advisory-only: unparseable output should NOT
    block the trade. Instead it falls back to APPROVE with Kelly sizing.
    """
    mock_run_agent.return_value = {
        "response": "This is not JSON.",
        "tokens_used": 50
    }
    
    result = await run_pre_trade("AAPL", 90, "cycle_1", "bot_1")
    
    # Advisory-only: unparseable output approves with Kelly fallback
    assert result["decision"] == "APPROVE"

@pytest.mark.asyncio
async def test_run_pre_trade_veto_decision(mock_run_agent):
    """Test VETO decision is correctly returned."""
    mock_run_agent.return_value = {
        "response": '''{
            "decision": "VETO",
            "ticker": "AAPL",
            "shares": 0,
            "entry_price": 0,
            "stop_loss": 0,
            "risk_reward_ratio": 0.5,
            "position_pct": 0,
            "total_cost": 0,
            "veto_reason": "Risk reward ratio < 1.5",
            "rationale": "Too risky"
        }''',
        "tokens_used": 100
    }
    
    result = await run_pre_trade("AAPL", 90, "cycle_1", "bot_1")
    
    assert result["decision"] == "VETO"
    assert result["risk_reward_ratio"] == 0.5
    assert result["veto_reason"] == "Risk reward ratio < 1.5"

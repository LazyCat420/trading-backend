import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from app.cycle.trading_phase import execute_decisions
from app.cycle.orchestration.cycle_control import cycle_control

@pytest.fixture(autouse=True)
def clean_cycle_control():
    cycle_control.reset()
    yield
    cycle_control.reset()

@pytest.fixture
def mock_trading_deps():
    with patch("app.cycle.trading_phase.get_portfolio") as mock_get_pf, \
         patch("app.cycle.trading_phase.buy", new_callable=AsyncMock) as mock_buy, \
         patch("app.cycle.trading_phase.sell", new_callable=AsyncMock) as mock_sell, \
         patch("app.cycle.trading_phase.check_portfolio_gate") as mock_gate, \
         patch("app.cycle.trading_phase._get_current_price") as mock_price, \
         patch("app.cycle.trading_phase.run_portfolio_allocator", new_callable=AsyncMock) as mock_alloc, \
         patch("app.cycle.trading_phase.run_trade_execution", new_callable=AsyncMock) as mock_exec:
        
        # Default mock returns
        mock_get_pf.return_value = {
            "cash": 10000.0,
            "positions": [],
            "position_count": 0
        }
        mock_gate.return_value = {"blocked": False, "warnings": []}
        mock_price.return_value = (150.0, None)
        mock_alloc.return_value = {}
        mock_exec.return_value = {"decision": "APPROVE"}
        mock_buy.return_value = {"qty": 10, "price": 150.0, "amount": 1500.0}
        mock_sell.return_value = {"qty": 10, "price": 150.0, "amount": 1500.0, "realized_pnl": 0.0}
        
        yield {
            "get_portfolio": mock_get_pf,
            "buy": mock_buy,
            "sell": mock_sell,
            "gate": mock_gate,
            "price": mock_price,
            "allocator": mock_alloc,
            "executor": mock_exec
        }

@pytest.mark.anyio
async def test_portfolio_sizing_veto_honored(mock_trading_deps):
    deps = mock_trading_deps
    deps["allocator"].return_value = {
        "AAPL": {
            "decision": "VETO",
            "veto_reason": "Extreme leverage sector cap reached"
        }
    }
    
    decisions = [
        {"ticker": "AAPL", "action": "BUY", "confidence": 85, "rationale": "Bullish breakout"}
    ]
    
    result = await execute_decisions(decisions, bot_id="test_bot")
    
    # Assert buy was NOT called
    deps["buy"].assert_not_called()
    assert result["counts"]["blocked"] == 1
    assert any("VETO (Portfolio Sizing)" in s["reason"] for s in result["skipped"])

@pytest.mark.anyio
async def test_trade_execution_veto_honored(mock_trading_deps):
    deps = mock_trading_deps
    deps["executor"].return_value = {
        "decision": "VETO",
        "veto_reason": "High risk of reversion"
    }
    
    decisions = [
        {"ticker": "AAPL", "action": "BUY", "confidence": 85, "rationale": "Bullish breakout"}
    ]
    
    result = await execute_decisions(decisions, bot_id="test_bot")
    
    # Assert buy was NOT called
    deps["buy"].assert_not_called()
    assert result["counts"]["blocked"] == 1
    assert any("VETO (Trade Execution)" in s["reason"] for s in result["skipped"])

@pytest.mark.anyio
async def test_convert_sell_hold_path(mock_trading_deps):
    deps = mock_trading_deps
    # AAPL is held
    deps["get_portfolio"].return_value = {
        "cash": 5000.0,
        "positions": [{"ticker": "AAPL", "qty": 100, "avg_entry_price": 150.0}],
        "position_count": 1
    }
    # HOLD advisory suggests CONVERT_SELL
    deps["executor"].return_value = {
        "decision": "CONVERT_SELL",
        "sell_pct": 50
    }
    
    decisions = [
        {"ticker": "AAPL", "action": "HOLD", "confidence": 75, "rationale": "Strong resistance"}
    ]
    
    await execute_decisions(decisions, bot_id="test_bot")
    
    # Assert sell was called with qty_pct=0.5
    deps["sell"].assert_called_once_with("test_bot", "AAPL", cycle_id="", qty_pct=0.5)

@pytest.mark.anyio
async def test_sell_pct_advisory_sizing(mock_trading_deps):
    deps = mock_trading_deps
    # AAPL is held
    deps["get_portfolio"].return_value = {
        "cash": 5000.0,
        "positions": [{"ticker": "AAPL", "qty": 100, "avg_entry_price": 150.0}],
        "position_count": 1
    }
    # SELL advisory suggests 40% exit
    deps["executor"].return_value = {
        "decision": "APPROVE",
        "sell_pct": 40
    }
    
    decisions = [
        {"ticker": "AAPL", "action": "SELL", "confidence": 90, "rationale": "Harvest profits"}
    ]
    
    await execute_decisions(decisions, bot_id="test_bot")
    
    # Assert sell was called with qty_pct=0.4
    deps["sell"].assert_called_once_with("test_bot", "AAPL", cycle_id="", qty_pct=0.4)

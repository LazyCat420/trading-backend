import pytest
from unittest.mock import patch, MagicMock

from app.trading.portfolio import (
    get_current_state,
    take_snapshot,
    get_equity_curve,
    get_performance_summary
)
from app.config import settings

@pytest.fixture
def mock_db():
    return MagicMock()

@patch("app.trading.portfolio._get_default_bot_id", return_value="bot1")
@patch("app.trading.portfolio.get_db")
def test_get_current_state_empty(mock_get_db, mock_default_bot, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    # Empty bot, empty snapshots, empty positions
    def execute_mock(query, *args):
        cur = MagicMock()
        if "FROM bots" in query:
            cur.fetchone.return_value = None
        elif "FROM portfolio_snapshots" in query:
            cur.fetchone.return_value = None
        elif "FROM positions" in query:
            cur.fetchall.return_value = []
        return cur
    mock_db.execute.side_effect = execute_mock
    
    state = get_current_state()
    
    assert state["bot_id"] == "bot1"
    assert state["cash"] == settings.STARTING_CASH
    assert state["total_value"] == settings.STARTING_CASH
    assert state["position_count"] == 0

@patch("app.trading.portfolio._get_default_bot_id", return_value="bot1")
@patch("app.trading.portfolio.get_db")
def test_get_current_state_with_positions(mock_get_db, mock_default_bot, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    def execute_mock(query, *args):
        cur = MagicMock()
        if "FROM bots" in query:
            cur.fetchone.return_value = (5000.0, 100.0, 5) # cash, pnl, trades
        elif "FROM portfolio_snapshots" in query:
            cur.fetchone.return_value = ("2023-10-01T00:00:00Z",)
        elif "FROM positions" in query:
            # AAPL: 10 shares @ $150
            cur.fetchall.return_value = [("AAPL", 10.0, 150.0, 0.05)]
        elif "FROM price_history" in query:
            cur.fetchone.return_value = (160.0,) # current price
        elif "FROM ticker_metadata" in query:
            cur.fetchone.return_value = ("Tech", "Mega", 2000000.0)
        elif "FROM fundamentals" in query:
            cur.fetchone.return_value = (15.0, 0.1)
        elif "FROM technicals" in query:
            cur.fetchone.return_value = (50.0,)
        return cur
    mock_db.execute.side_effect = execute_mock
    
    state = get_current_state()
    
    assert state["cash"] == 5000.0
    assert state["total_value"] == 5000.0 + (10.0 * 160.0) # 5000 + 1600 = 6600
    assert state["position_count"] == 1
    assert state["positions"][0]["ticker"] == "AAPL"
    assert state["positions"][0]["current_price"] == 160.0

@patch("app.trading.portfolio._get_default_bot_id", return_value="bot1")
@patch("app.trading.portfolio.get_db")
def test_get_current_state_price_sanity(mock_get_db, mock_default_bot, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    def execute_mock(query, *args):
        cur = MagicMock()
        if "FROM bots" in query:
            cur.fetchone.return_value = (5000.0, 0, 0)
        elif "FROM portfolio_snapshots" in query:
            cur.fetchone.return_value = None
        elif "FROM positions" in query:
            # AAPL: 10 shares @ $150
            cur.fetchall.return_value = [("AAPL", 10.0, 150.0, 0.05)]
        elif "FROM price_history" in query:
            # Phantom price! 2000 vs 150 > 10x
            cur.fetchone.return_value = (2000.0,)
        elif "FROM ticker_metadata" in query:
            cur.fetchone.return_value = None
        elif "FROM fundamentals" in query:
            cur.fetchone.return_value = None
        elif "FROM technicals" in query:
            cur.fetchone.return_value = None
        return cur
    mock_db.execute.side_effect = execute_mock
    
    state = get_current_state()
    
    # Should fallback to entry price of 150
    assert state["positions"][0]["current_price"] == 150.0
    assert state["total_value"] == 5000.0 + 1500.0

@patch("app.trading.portfolio.get_current_state")
@patch("app.trading.portfolio.get_db")
def test_take_snapshot(mock_get_db, mock_get_current_state, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    mock_get_current_state.return_value = {"cash": 1000.0, "total_value": 2000.0}
    
    res = take_snapshot("bot1")
    
    assert res["total_value"] == 2000.0
    mock_db.execute.assert_called_once()

@patch("app.trading.portfolio.get_db")
def test_get_equity_curve(mock_get_db, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    now = "2023-10-01T00:00:00Z"
    mock_db.execute.return_value.fetchall.return_value = [
        (1000.0, 500.0, now)
    ]
    
    curve = get_equity_curve("bot1")
    
    assert len(curve) == 1
    assert curve[0]["total_value"] == 1000.0

@patch("app.trading.portfolio.get_current_state")
@patch("app.services.bot_manager.get_bot_starting_cash", return_value=10000.0)
@patch("app.trading.portfolio.get_db")
def test_get_performance_summary(mock_get_db, mock_starting_cash, mock_get_current_state, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    mock_get_current_state.return_value = {"cash": 5000.0, "total_value": 15000.0, "position_count": 2}
    
    # Mock bot table
    mock_db.execute.return_value.fetchone.return_value = (10, 5000.0, 0.6) # trades, realized, win rate
    
    summary = get_performance_summary("bot1")
    
    assert summary["pnl"] == 5000.0 # 15000 - 10000
    assert summary["pnl_pct"] == 50.0
    assert summary["total_trades"] == 10
    assert summary["win_rate"] == 0.6

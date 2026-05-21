import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from app.trading.order_triggers import (
    create_trigger,
    check_triggers,
    cancel_trigger,
    list_triggers
)

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.mark.asyncio
@patch("app.trading.order_triggers.get_db")
async def test_create_trigger_invalid(mock_get_db):
    res1 = await create_trigger("bot1", "AAPL", "invalid_type", 100.0)
    assert "error" in res1
    assert "Invalid trigger_type" in res1["error"]

    res2 = await create_trigger("bot1", "AAPL", "stop_loss", -10.0)
    assert "error" in res2
    assert "trigger_price must be positive" in res2["error"]

    res3 = await create_trigger("bot1", "AAPL", "trailing_stop", 100.0, trailing_pct=-0.1)
    assert "error" in res3
    assert "trailing_stop requires a positive trailing_pct" in res3["error"]

@pytest.mark.asyncio
@patch("app.trading.order_triggers._get_current_price")
@patch("app.trading.order_triggers.get_db")
async def test_create_trigger_success(mock_get_db, mock_get_current_price, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    mock_get_current_price.return_value = (150.0, None)
    
    res = await create_trigger("bot1", "AAPL", "stop_loss", 100.0)
    
    assert "id" in res
    assert res["ticker"] == "AAPL"
    mock_db.execute.assert_called_once()

@pytest.mark.asyncio
@patch("app.trading.order_triggers._get_current_price")
@patch("app.trading.order_triggers.get_db")
async def test_create_trigger_trailing_stop(mock_get_db, mock_get_current_price, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    mock_get_current_price.return_value = (200.0, None)
    
    res = await create_trigger("bot1", "AAPL", "trailing_stop", 100.0, trailing_pct=0.1)
    
    assert "id" in res
    assert res["trigger_type"] == "trailing_stop"
    # Execute args check
    args = mock_db.execute.call_args[0][1]
    # args: [trigger_id, bot_id, ticker, trigger_type, trigger_price, action, qty_pct, trailing_pct, highest_price, ...]
    assert args[8] == 200.0  # highest_price

@pytest.mark.asyncio
@patch("app.services.pipeline_service.PipelineService.start_cycle")
@patch("app.trading.order_triggers._get_current_price")
@patch("app.trading.order_triggers.get_db")
async def test_check_triggers_stop_loss_fired(mock_get_db, mock_get_current_price, mock_start_cycle, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    # Mock active triggers in DB
    mock_db.execute.return_value.fetchall.return_value = [
        ("trg1", "AAPL", "stop_loss", 100.0, "SELL", 1.0, None, None, "reason")
    ]
    
    # Current price is 95, so stop loss should fire
    mock_get_current_price.return_value = (95.0, None)
    
    mock_start_cycle.return_value = {"cycle_id": "test_cycle"}
    
    results = await check_triggers("bot1")
    
    assert len(results) == 1
    assert results[0]["status"] == "cycle_started"
    assert results[0]["trigger_id"] == "trg1"
    
    mock_start_cycle.assert_called_once_with(
        tickers=["AAPL"],
        collect=True,
        analyze=True,
        trade=True,
        trigger_type="edge_case_stop_loss"
    )

@pytest.mark.asyncio
@patch("app.services.pipeline_service.PipelineService.start_cycle")
@patch("app.trading.order_triggers._get_current_price")
@patch("app.trading.order_triggers.get_db")
async def test_check_triggers_trailing_stop(mock_get_db, mock_get_current_price, mock_start_cycle, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    # Mock active triggers in DB (highest price 200, trail 10%)
    mock_db.execute.return_value.fetchall.return_value = [
        ("trg1", "AAPL", "trailing_stop", 0.0, "SELL", 1.0, 0.1, 200.0, "reason")
    ]
    
    # Trigger fires at 200 * 0.9 = 180. Current price = 175
    mock_get_current_price.return_value = (175.0, None)
    mock_start_cycle.return_value = {"cycle_id": "test_cycle"}
    
    results = await check_triggers("bot1")
    
    assert len(results) == 1
    assert results[0]["trigger_id"] == "trg1"

@pytest.mark.asyncio
@patch("app.trading.order_triggers.get_db")
async def test_cancel_trigger(mock_get_db, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    mock_db.execute.return_value.fetchone.return_value = ("trg1", "AAPL", "stop_loss")
    
    res = await cancel_trigger("trg1")
    
    assert res["status"] == "cancelled"
    assert res["id"] == "trg1"

def test_list_triggers():
    with patch("app.trading.order_triggers.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_get_db.return_value.__enter__.return_value = mock_db
        
        now = datetime.now(timezone.utc)
        mock_db.execute.return_value.fetchall.return_value = [
            ("trg1", "AAPL", "stop_loss", 100.0, "SELL", 1.0, None, None, "reason", True, None, now, "bot")
        ]
        
        res = list_triggers("bot1")
        
        assert len(res) == 1
        assert res[0]["id"] == "trg1"

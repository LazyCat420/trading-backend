import pytest
from unittest.mock import MagicMock, patch
from app.trading.paper_trader import sell

@pytest.mark.integration
@pytest.mark.asyncio
async def test_lot_closures_created_on_sell(real_db):
    """Verify that a full SELL operation correctly creates lot_closures records in the real DB."""
    from contextlib import contextmanager
    import datetime
    
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    @contextmanager
    def fake_get_db():
        yield real_db

    # Seed the test DB with a mock position and lot
    real_db.execute(
        "INSERT INTO positions (id, bot_id, ticker, qty, avg_entry_price, stop_loss_pct, opened_at) "
        "VALUES ('pos-123', 'test-bot', 'AAPL', 10.0, 150.0, 0.08, %s)",
        [now_iso]
    )
    real_db.execute(
        "INSERT INTO position_lots (lot_id, bot_id, ticker, fill_id, opened_at, original_qty, remaining_qty, entry_price, status) "
        "VALUES ('lot-abc', 'test-bot', 'AAPL', 'fill-123', %s, 10.0, 10.0, 150.0, 'open')",
        [now_iso]
    )
    real_db.execute(
        "INSERT INTO price_history (ticker, close, date, volume, source) VALUES ('AAPL', 160.0, %s, 1000, 'mock')",
        [now_iso]
    )
    
    with patch("app.trading.paper_trader.get_db", new=fake_get_db):
        result = await sell("test-bot", "AAPL", qty_pct=1.0)
        
    assert result.get("error") is None, f"Sell failed: {result.get('error')}"
    assert result["action"] == "SELL"
    assert result["qty"] == 10.0
    
    # Verify lot_closures insert actually succeeded in DB
    closures = real_db.execute("SELECT bot_id, ticker, closed_qty, entry_price, exit_price, realized_pnl FROM lot_closures").fetchall()
    
    assert len(closures) == 1, "Expected one insert into lot_closures"
    
    c = closures[0]
    assert c[0] == "test-bot" # bot_id
    assert c[1] == "AAPL" # ticker
    assert c[2] == 10.0 # closed_qty
    assert c[3] == 150.0 # entry_price
    assert c[4] == 160.0 # exit_price
    assert c[5] == 100.0 # realized_pnl

@pytest.mark.integration
@pytest.mark.asyncio
async def test_lot_closures_partial_sell(real_db):
    """Verify that a partial SELL creates a lot_closures record for only the sold amount in the real DB."""
    from contextlib import contextmanager
    import datetime
    
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    @contextmanager
    def fake_get_db():
        yield real_db

    real_db.execute(
        "INSERT INTO positions (id, bot_id, ticker, qty, avg_entry_price, stop_loss_pct, opened_at) "
        "VALUES ('pos-123', 'test-bot', 'AAPL', 10.0, 150.0, 0.08, %s)",
        [now_iso]
    )
    real_db.execute(
        "INSERT INTO position_lots (lot_id, bot_id, ticker, fill_id, opened_at, original_qty, remaining_qty, entry_price, status) "
        "VALUES ('lot-abc', 'test-bot', 'AAPL', 'fill-123', %s, 10.0, 10.0, 150.0, 'open')",
        [now_iso]
    )
    real_db.execute(
        "INSERT INTO price_history (ticker, close, date, volume, source) VALUES ('AAPL', 160.0, %s, 1000, 'mock')",
        [now_iso]
    )

    with patch("app.trading.paper_trader.get_db", new=fake_get_db):
        # Sell 50% of the position
        result = await sell("test-bot", "AAPL", qty_pct=0.5)
        
    assert result.get("error") is None
    assert result["qty"] == 5.0
    
    closures = real_db.execute("SELECT closed_qty, realized_pnl FROM lot_closures").fetchall()
    
    assert len(closures) == 1
    c = closures[0]
    assert c[0] == 5.0 # closed_qty is only 5.0
    assert c[1] == 50.0 # realized_pnl is only 50.0


import pytest
from unittest.mock import MagicMock, patch
from app.services.alert_service import record_fund_alert
from app.trading.paper_trader import check_stop_losses

@pytest.mark.asyncio
async def test_fund_alerts_creation(real_db):
    """Verify that a fund alert is created successfully in the real database."""
    from contextlib import contextmanager

    @contextmanager
    def fake_get_db():
        yield real_db

    with patch("app.services.alert_service.get_db", fake_get_db), patch("app.trading.paper_trader.get_db", fake_get_db):
        # 1. Test direct creation
        result = record_fund_alert(
            alert_type="anomaly",
            entity_name="test-bot",
            detail="Testing alert service directly",
            severity="low"
        )
        
        assert "error" not in result
        assert result["alert_type"] == "anomaly"
        
        # Verify insert in real DB
        alerts = real_db.execute("SELECT alert_type, entity_name, severity FROM fund_alerts WHERE alert_type = 'anomaly'").fetchall()
        assert len(alerts) == 1
        assert alerts[0] == ("anomaly", "test-bot", "low")
        
        # 2. Test stop-loss triggering an alert
        import datetime
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        
        # Seed position
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
            "INSERT INTO price_history (ticker, close, date, volume, source) VALUES ('AAPL', 130.0, %s, 1000, 'mock')",
            [now_iso]
        )
        
        # Mock _ensure_bot from trying to insert
        with patch("app.trading.paper_trader._ensure_bot", lambda x: None):
            triggered = await check_stop_losses("test-bot", cycle_id="test-cycle")
            
        assert len(triggered) == 1
        
        sl_alerts = real_db.execute("SELECT ticker, entity_name, severity FROM fund_alerts WHERE alert_type = 'stop_loss'").fetchall()
        assert len(sl_alerts) == 1
        assert sl_alerts[0] == ("AAPL", "test-bot", "high")

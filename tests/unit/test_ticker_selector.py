from unittest.mock import MagicMock, patch
from contextlib import contextmanager
import pytest
from app.pipeline.ticker_selector import TickerSelector, TickerSelectionResult


def test_select_tickers_for_cycle_v2(monkeypatch, mock_db):
    """Verify ticker selection merges positions, watchlist, and discoveries."""

    def side_effect_execute(query, params=None):
        mock_cursor = MagicMock()
        if "information_schema" in query:
            mock_cursor.fetchone.return_value = (1,)
        elif "position_lots" in query:
            mock_cursor.fetchall.return_value = [("AAPL",), ("MSFT",)]
        elif "watchlist" in query:
            mock_cursor.fetchall.return_value = [("GOOG",)]
        elif "decision_outcomes WHERE created_at" in query:
            mock_cursor.fetchall.return_value = []
        elif "analysis_results WHERE created_at" in query:
            mock_cursor.fetchall.return_value = []
        elif "discovered_tickers" in query:
            assert "?" not in query, "SQLite placeholder '?' found! Must use '%s' for Postgres."
            mock_cursor.fetchall.return_value = [("TSLA", 100, "large", True, "2000-01-01")]
        else:
            mock_cursor.fetchall.return_value = []
        return mock_cursor

    mock_db.execute.side_effect = side_effect_execute

    @contextmanager
    def fake_get_db():
        yield mock_db

    # Patch at the root level so all modules get the fake DB
    monkeypatch.setattr("app.db.connection.get_db", fake_get_db)
    monkeypatch.setattr("app.pipeline.ticker_selector.get_db", fake_get_db)
    
    # Mock bot manager to avoid nested db calls
    monkeypatch.setattr("app.services.bot_manager.get_active_bot_id", lambda: "test-bot")

    res = TickerSelector.select_tickers_for_cycle_v2(["NVDA"], cap=50)

    assert "AAPL" in res.position_tickers
    assert "MSFT" in res.position_tickers
    assert "GOOG" in res.non_position_tickers
    assert "NVDA" in res.non_position_tickers
    assert "TSLA" in res.non_position_tickers

    assert len(res.position_tickers) == 2
    assert len(res.non_position_tickers) == 3

    assert len(res.all_tickers) == 5

@patch("app.pipeline.ticker_selector.get_db")
@patch("app.services.bot_manager.get_active_bot_id")
def test_selector_quarantine_filtering(mock_get_active_bot_id, mock_get_db):
    """Verify that the discovery query explicitly filters out quarantined tickers."""
    mock_get_active_bot_id.return_value = "test-bot"
    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    executed_queries = []
    
    def side_effect_execute(query, params=None):
        executed_queries.append(query)
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        return mock_cursor

    mock_db.execute.side_effect = side_effect_execute
    
    # We pass empty requested to force discovery logic to run
    TickerSelector.select_tickers_for_cycle_v2([], cap=10)
    
    discovery_query = next((q for q in executed_queries if "discovered_tickers" in q), None)
    assert discovery_query is not None
    assert "d.validation_status != 'quarantine'" in discovery_query

@patch("app.pipeline.ticker_selector.get_db")
@patch("app.services.bot_manager.get_active_bot_id")
def test_select_tickers_with_cooldown(mock_get_active_bot_id, mock_get_db):
    mock_get_active_bot_id.return_value = "test-bot"
    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    def side_effect_execute(query, params=None):
        mock_cursor = MagicMock()
        if "information_schema" in query:
            mock_cursor.fetchone.return_value = (1,)
        elif "position_lots" in query:
            mock_cursor.fetchall.return_value = [("AAPL",)]
        elif "decision_outcomes WHERE created_at" in query:
            # AAPL is a position, should still be included despite cooldown
            mock_cursor.fetchall.return_value = [("AAPL",)] 
        elif "analysis_results WHERE created_at" in query:
            # NVDA requested, GOOG watchlist, both should be skipped
            mock_cursor.fetchall.return_value = [("NVDA",), ("GOOG",)] 
        elif "watchlist" in query:
            mock_cursor.fetchall.return_value = [("GOOG",), ("MSFT",)]
        elif "discovered_tickers" in query:
            mock_cursor.fetchall.return_value = [("TSLA", 100, "large", True, "2000-01-01")]
        else:
            mock_cursor.fetchall.return_value = []
        return mock_cursor

    mock_db.execute.side_effect = side_effect_execute
    
    res = TickerSelector.select_tickers_for_cycle_v2(["NVDA", "AMZN"], cap=50)
    
    assert "AAPL" in res.position_tickers # Position included regardless of cooldown
    assert "NVDA" not in res.non_position_tickers # Requested, skipped due to cooldown
    assert "GOOG" not in res.non_position_tickers # Watchlist, skipped due to cooldown
    assert "AMZN" in res.non_position_tickers # Requested, not in cooldown
    assert "MSFT" in res.non_position_tickers # Watchlist, not in cooldown
    assert "TSLA" in res.non_position_tickers # Discovered, not in cooldown

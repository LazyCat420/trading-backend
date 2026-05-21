import pytest
from unittest.mock import MagicMock, patch

from app.trading.watchlist import (
    add_ticker,
    remove_ticker,
    auto_purge_ticker,
    pause_ticker,
    resume_ticker,
    ban_ticker,
    is_banned,
    check_ban_patterns
)

@pytest.fixture
def mock_db():
    db = MagicMock()
    return db

@patch("app.trading.watchlist.get_db")
@patch("app.trading.watchlist.is_banned", return_value=False)
def test_add_ticker_success(mock_is_banned, mock_get_db, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    mock_db.execute.return_value.fetchone.return_value = None
    
    assert add_ticker("AAPL", source="test") is True
    
    from unittest.mock import ANY
    mock_db.execute.assert_called_with(
        "INSERT INTO watchlist (ticker, source, notes, added_at, status) VALUES (%s, %s, %s, %s, 'active')",
        ["AAPL", "test", "", ANY]
    )

@patch("app.trading.watchlist.is_banned", return_value=True)
def test_add_ticker_banned(mock_is_banned):
    assert add_ticker("AAPL") is False

@patch("app.trading.watchlist.get_db")
def test_remove_ticker(mock_get_db, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    # Exists
    mock_db.execute.return_value.fetchone.return_value = ("AAPL",)
    assert remove_ticker("AAPL") is True
    
    # Doesn't exist
    mock_db.execute.return_value.fetchone.return_value = None
    assert remove_ticker("INVALID") is False

@patch("app.trading.watchlist.get_db")
def test_auto_purge_ticker(mock_get_db, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    mock_db.execute.return_value.fetchone.return_value = ("AAPL",)
    assert auto_purge_ticker("AAPL", "Low confidence") is True

@patch("app.trading.watchlist.get_db")
def test_pause_resume_ticker(mock_get_db, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    # Pause
    mock_db.execute.return_value.fetchone.return_value = ("AAPL",)
    assert pause_ticker("AAPL") is True
    
    # Resume
    assert resume_ticker("AAPL") is True

@patch("app.trading.watchlist.get_db")
@patch("app.trading.watchlist._snapshot_market_data", return_value=(None, 0.5, None))
def test_check_ban_patterns(mock_snapshot, mock_get_db, mock_db):
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    # Mock pattern: price < 1.0
    mock_db.execute.return_value.fetchall.return_value = [
        ("penny_stock", '{"price_lt": 1.0}')
    ]
    
    assert check_ban_patterns("PENN") == "penny_stock"

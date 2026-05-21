import pytest
from unittest.mock import MagicMock
import pandas as pd

from app.data.sector_aggregator import backfill_sector_performance


@pytest.mark.asyncio
async def test_backfill_sector_performance_empty_db(monkeypatch, mock_db):
    """Ensure backfill handles an empty price_history gracefully."""
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    monkeypatch.setattr("app.data.sector_aggregator.get_db", mock_get_db)
    
    # Mock row count check to return 0 (no history)
    mock_db.execute.return_value.fetchone.return_value = (0,)
    
    # Mock price_history query to return empty
    mock_db.execute.return_value.fetchall.return_value = []
    
    await backfill_sector_performance()
    
    # executemany should not be called if there's no data
    assert mock_db.executemany.call_count == 0


@pytest.mark.asyncio
async def test_backfill_sector_performance_populates_history(monkeypatch, mock_db):
    """Ensure backfill calculates and inserts historical data."""
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    monkeypatch.setattr("app.data.sector_aggregator.get_db", mock_get_db)
    
    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "COUNT(DISTINCT date)" in query:
            cursor.fetchone.return_value = (0,)
        elif "SELECT p.ticker, p.date" in query:
            # Provide some mock price history: 2 days of data for 2 tickers in same sector
            cursor.description = [("ticker",), ("date",), ("close",), ("sector",)]
            cursor.fetchall.return_value = [
                ("AAPL", "2023-01-01", 100.0, "Technology"),
                ("AAPL", "2023-01-02", 105.0, "Technology"),
                ("MSFT", "2023-01-01", 200.0, "Technology"),
                ("MSFT", "2023-01-02", 210.0, "Technology"),
            ]
        return cursor
        
    mock_db.execute.side_effect = mock_execute
    
    await backfill_sector_performance()
    
    # Ensure executemany was called to insert data
    assert mock_db.executemany.call_count == 1
    
    # Verify the inserted data
    insert_query = mock_db.executemany.call_args[0][0]
    insert_data = mock_db.executemany.call_args[0][1]
    
    assert "INSERT INTO sector_performance" in insert_query
    
    # We expect 1 row inserted since there's only 1 day of returns (the second day)
    # Day 1 has no return since pct_change requires a previous row
    assert len(insert_data) == 1
    row = insert_data[0]
    assert row[0] == "Technology"
    assert row[1] == "2023-01-02"
    # AAPL went 100 -> 105 (5%), MSFT went 200 -> 210 (5%), average is 5% (0.05)
    assert row[2] == pytest.approx(0.05)

@pytest.mark.asyncio
async def test_backfill_skips_if_history_exists(monkeypatch, mock_db):
    """Ensure backfill skips if history is already present."""
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    monkeypatch.setattr("app.data.sector_aggregator.get_db", mock_get_db)
    
    # Mock row count check to return 2 (history exists)
    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "COUNT(DISTINCT date)" in query:
            cursor.fetchone.return_value = (2,)
        return cursor
        
    mock_db.execute.side_effect = mock_execute
    
    await backfill_sector_performance()
    
    # price history should not be queried and executemany should not be called
    assert mock_db.executemany.call_count == 0

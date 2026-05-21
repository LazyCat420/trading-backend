import pytest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from app.data.sector_aggregator import backfill_sector_performance
from app.data.sector_correlation_engine import compute_all_correlations


@pytest.mark.asyncio
async def test_correlation_computes_with_backfilled_data(monkeypatch, mock_db):
    """
    Simulate startup:
    1. backfill_sector_performance runs on 20 days of data
    2. compute_all_correlations runs
    Assert that the correlations actually compute instead of skipping.
    """
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    monkeypatch.setattr("app.data.sector_aggregator.get_db", mock_get_db)
    monkeypatch.setattr("app.data.sector_correlation_engine.get_db", mock_get_db)
    
    # 1. Setup mock data for backfill_sector_performance
    base_date = datetime(2023, 1, 1)
    mock_dates = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(20)]
    
    backfill_data = []
    # Tech sector
    for i, date_str in enumerate(mock_dates):
        backfill_data.append(("AAPL", date_str, 100.0 + i, "Technology"))
    # Finance sector
    for i, date_str in enumerate(mock_dates):
        backfill_data.append(("JPM", date_str, 200.0 - i, "Finance"))

    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "COUNT(DISTINCT date)" in query:
            cursor.fetchone.return_value = (0,)
        elif "SELECT p.ticker, p.date, p.close, t.sector" in query:
            cursor.description = [("ticker",), ("date",), ("close",), ("sector",)]
            cursor.fetchall.return_value = backfill_data
        elif "SELECT sector, date, avg_return_1d" in query:
            cursor.description = [("sector",), ("date",), ("avg_return_1d",)]
            # Mock the data that would have been inserted by backfill
            sector_data = []
            for i, date_str in enumerate(mock_dates[1:]): # pct change drops first day
                # Add variance so correlation isn't NaN
                val = 0.01 if i % 2 == 0 else -0.01
                sector_data.append(("Technology", date_str, val))
                sector_data.append(("Finance", date_str, -val))
            cursor.fetchall.return_value = sector_data
        elif "SELECT p.ticker, p.date, p.close as stock_price, t.sector" in query:
            cursor.description = [("ticker",), ("date",), ("stock_price",), ("sector",)]
            cursor.fetchall.return_value = []
        elif "SELECT symbol as commodity, date, close as comm_price" in query:
            cursor.description = [("commodity",), ("date",), ("comm_price",)]
            cursor.fetchall.return_value = []
        else:
            cursor.fetchall.return_value = []
        return cursor
        
    mock_db.execute.side_effect = mock_execute
    
    # Run backfill
    await backfill_sector_performance()
    
    # Verify backfill ran and inserted 19 days (20 - 1) * 2 sectors = 38 rows
    assert mock_db.executemany.call_count == 1
    insert_data = mock_db.executemany.call_args[0][1]
    assert len(insert_data) == 38
    
    # Run correlations (which uses 30d period so it expects > 15 points)
    # 19 days is > 15 days, so it should not skip
    result = await compute_all_correlations()
    
    # Ensure executemany was called a second time for sector_correlations
    assert mock_db.executemany.call_count == 2
    
    # Verify sector_correlations insert
    corr_insert_query = mock_db.executemany.call_args[0][0]
    corr_insert_data = mock_db.executemany.call_args[0][1]
    
    assert "INSERT INTO sector_correlations" in corr_insert_query
    
    # We expect 2 periods (30d and 90d) for the 1 pair (Technology, Finance)
    # But wait, 19 days is > 15 (for 30d) but NOT > 45 (for 90d).
    # So it should only insert the 30d correlation.
    assert len(corr_insert_data) == 1
    
    row = corr_insert_data[0]
    assert (row[0] == "Finance" and row[1] == "Technology") or (row[0] == "Technology" and row[1] == "Finance")
    assert row[4] == "30d"
    assert "Computed 1 sector & 0 comm correlations" in result

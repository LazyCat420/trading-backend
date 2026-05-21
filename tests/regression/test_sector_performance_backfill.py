import pytest
from unittest.mock import MagicMock
from app.data.sector_correlation_engine import compute_all_correlations

@pytest.mark.asyncio
async def test_regression_0_sector_correlations_computed(monkeypatch, mock_db):
    """
    REGRESSION LOCK
    Original Bug: "Computed 0 sector correlations and 1788 commodity correlations"
    Cause: The `sector_performance` table was only populated with the latest date,
    meaning it lacked the minimum 15-day history required by `compute_all_correlations`.
    As a result, `df_sector` was pivoted to a single row, and the function
    silently skipped correlation computations without throwing an error.
    
    Fix: Added `backfill_sector_performance` to pre-populate the historical
    returns in `sector_performance` before `compute_all_correlations` runs.
    
    This test verifies that `compute_all_correlations` skips correctly when data < 15 days,
    and succeeds when data is >= 15 days, mimicking the exact conditions of the bug.
    """
    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    monkeypatch.setattr("app.data.sector_correlation_engine.get_db", mock_get_db)

    # Sub-test 1: Test the bug condition (too few days in history)
    def mock_execute_too_few(query, *args, **kwargs):
        cursor = MagicMock()
        if "SELECT sector, date, avg_return_1d" in query:
            cursor.description = [("sector",), ("date",), ("avg_return_1d",)]
            # Only 1 day of data - this was the bug state
            cursor.fetchall.return_value = [
                ("Technology", "2023-01-01", 0.05),
                ("Finance", "2023-01-01", -0.05),
            ]
        elif "SELECT p.ticker, p.date" in query or "SELECT symbol as commodity" in query:
            cursor.description = [("a",), ("b",), ("c",), ("d",)]
            cursor.fetchall.return_value = []
        else:
            cursor.fetchall.return_value = []
        return cursor
        
    mock_db.execute.side_effect = mock_execute_too_few
    
    # Run correlations with bug state
    result = await compute_all_correlations()
    
    # Assert 0 correlations computed because it skipped due to lack of history
    assert "Computed 0 sector" in result
    
    # Sub-test 2: Test the fixed condition (15+ days of variance history)
    def mock_execute_fixed(query, *args, **kwargs):
        cursor = MagicMock()
        if "SELECT sector, date, avg_return_1d" in query:
            cursor.description = [("sector",), ("date",), ("avg_return_1d",)]
            # 20 days of data - this is the fixed state after backfill
            sector_data = []
            from datetime import datetime, timedelta
            base_date = datetime(2023, 1, 1)
            for i in range(20):
                date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
                val = 0.01 if i % 2 == 0 else -0.01
                sector_data.append(("Technology", date_str, val))
                sector_data.append(("Finance", date_str, -val))
            cursor.fetchall.return_value = sector_data
        elif "SELECT p.ticker, p.date" in query or "SELECT symbol as commodity" in query:
            cursor.description = [("a",), ("b",), ("c",), ("d",)]
            cursor.fetchall.return_value = []
        else:
            cursor.fetchall.return_value = []
        return cursor
        
    mock_db.execute.side_effect = mock_execute_fixed
    
    # Reset mock call count
    mock_db.executemany.reset_mock()
    
    # Run correlations with fixed state
    result = await compute_all_correlations()
    
    # Assert > 0 correlations computed because the backfill fixed the missing history
    assert "Computed 1 sector" in result
    assert mock_db.executemany.call_count == 1

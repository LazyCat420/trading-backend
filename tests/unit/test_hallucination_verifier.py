"""
Test: Hallucination Verifier Tool Coverage.

Unit tests for check_hallucination to ensure it returns complete
ground-truth data including technical indicators and fundamentals.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
import asyncio


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_db(price_row=None, tech_row=None, fund_row=None):
    """Create a mock DB context manager with configurable query responses."""
    db = MagicMock()

    call_count = {"n": 0}
    # Track SQL queries and return appropriate mock data
    queries_and_results = []

    def _execute_side_effect(sql, *args, **kwargs):
        cursor = MagicMock()
        sql_lower = sql.lower() if isinstance(sql, str) else ""
        queries_and_results.append(sql_lower)

        if "price_history" in sql_lower:
            cursor.fetchone.return_value = price_row
        elif "technicals" in sql_lower:
            cursor.fetchone.return_value = tech_row
        elif "fundamentals" in sql_lower:
            cursor.fetchone.return_value = fund_row
        else:
            cursor.fetchone.return_value = None
            cursor.fetchall.return_value = []
        return cursor

    db.execute = MagicMock(side_effect=_execute_side_effect)
    db.__enter__ = MagicMock(return_value=db)
    db.__exit__ = MagicMock(return_value=False)
    db._queries = queries_and_results
    return db


class TestCheckHallucination:

    def test_returns_technical_indicators(self):
        """check_hallucination must return technical indicators in ground_truth."""
        price_row = (198.43, 20000000)
        tech_row = (53.2, -1.08, -0.5, -0.58, 199.13, 187.40, 184.08,
                    5.72, 19.5, 12.5, 16.9, 210.0, 180.0)
        fund_row = (40.51, 4824056201216.0, 17.66, 0.63, 30.67, 0.56, 0.73, 7.25, 2.24)

        db = _make_mock_db(price_row, tech_row, fund_row)

        with patch("app.db.connection.get_db", return_value=db):
            from app.tools.pipeline_tools import check_hallucination
            result = json.loads(_run(check_hallucination("NVDA", "RSI is 53.2")))

        assert result["status"] == "success"
        gt = result["ground_truth"]
        assert "indicators" in gt, "ground_truth must include 'indicators'"
        assert "rsi_14" in gt["indicators"]
        assert gt["indicators"]["rsi_14"] == 53.2

    def test_returns_fundamentals(self):
        """check_hallucination must return fundamental data in ground_truth."""
        price_row = (198.43, 20000000)
        fund_row = (40.51, 4824056201216.0, 17.66, 0.63, 30.67, 0.56, 0.73, 7.25, 2.24)

        db = _make_mock_db(price_row, None, fund_row)

        with patch("app.db.connection.get_db", return_value=db):
            from app.tools.pipeline_tools import check_hallucination
            result = json.loads(_run(check_hallucination("AAPL", "PE is 40")))

        gt = result["ground_truth"]
        assert "pe_ratio" in gt
        assert gt["pe_ratio"] == 40.51
        assert "market_cap" in gt
        assert "forward_pe" in gt
        # Must NOT have dividend_yield (regression)
        assert "dividend_yield" not in gt

    def test_no_data_returns_inconclusive(self):
        """When no ground-truth data exists, return inconclusive."""
        db = _make_mock_db(None, None, None)

        with patch("app.db.connection.get_db", return_value=db):
            from app.tools.pipeline_tools import check_hallucination
            result = json.loads(_run(check_hallucination("FAKE", "price is 100")))

        assert result["status"] == "inconclusive"

    def test_queries_correct_table_name(self):
        """REGRESSION: Must query 'technicals', NOT 'technical_indicators'."""
        db = _make_mock_db((100.0, 1000), None, None)

        with patch("app.db.connection.get_db", return_value=db):
            from app.tools.pipeline_tools import check_hallucination
            _run(check_hallucination("TEST", "RSI is 50"))

        # Check all SQL queries executed
        all_sql = " ".join(db._queries)
        assert "technical_indicators" not in all_sql, (
            "REGRESSION: Still querying non-existent 'technical_indicators' table"
        )
        # Verify 'technicals' IS queried
        assert "technicals" in all_sql, (
            "Must query 'technicals' table for technical indicators"
        )

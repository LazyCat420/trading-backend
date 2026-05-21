"""Tests for app.pipeline.analysis.thesis_store."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.analysis.thesis_store import (
    ThesisRecord,
    get_thesis,
    is_thesis_stale,
    mark_unchanged,
    thesis_age_hours,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _mock_db_with_rows(rows):
    """Create a mock DB context manager that returns `rows` from fetchone."""
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = mock_cursor
    if rows is None:
        mock_cursor.fetchone.return_value = None
    else:
        mock_cursor.fetchone.return_value = rows
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    return mock_cursor


# ── get_thesis ─────────────────────────────────────────────────────────

@patch("app.pipeline.analysis.thesis_store.get_db")
def test_get_thesis_returns_none_for_unknown_ticker(mock_get_db):
    mock_db = _mock_db_with_rows(None)
    mock_get_db.return_value = mock_db
    assert get_thesis("FAKE999") is None


@patch("app.pipeline.analysis.thesis_store.get_db")
def test_get_thesis_returns_record_for_known_ticker(mock_get_db):
    now = datetime.now(timezone.utc)
    mock_db = _mock_db_with_rows(
        ("NVDA", "SELL", 82, "Overvalued due to PE ratio", now, False)
    )
    mock_get_db.return_value = mock_db

    thesis = get_thesis("NVDA")
    assert thesis is not None
    assert thesis.ticker == "NVDA"
    assert thesis.verdict == "SELL"
    assert thesis.confidence == 82
    assert thesis.summary == "Overvalued due to PE ratio"
    assert thesis.updated_at == now
    assert thesis.unchanged is False


# ── thesis_age_hours ───────────────────────────────────────────────────

@patch("app.pipeline.analysis.thesis_store.get_db")
def test_thesis_age_in_hours(mock_get_db):
    old_time = datetime.now(timezone.utc) - timedelta(hours=25)
    mock_db = _mock_db_with_rows(
        ("AAPL", "BUY", 75, "Undervalued", old_time, False)
    )
    mock_get_db.return_value = mock_db

    age = thesis_age_hours("AAPL")
    assert age > 24
    assert age < 26


@patch("app.pipeline.analysis.thesis_store.get_db")
def test_thesis_age_returns_inf_for_unknown(mock_get_db):
    mock_db = _mock_db_with_rows(None)
    mock_get_db.return_value = mock_db
    assert thesis_age_hours("FAKE") == float("inf")


# ── is_thesis_stale ───────────────────────────────────────────────────

@patch("app.pipeline.analysis.thesis_store.get_db")
def test_thesis_is_stale_after_72h(mock_get_db):
    old_time = datetime.now(timezone.utc) - timedelta(hours=73)
    mock_db = _mock_db_with_rows(
        ("AAPL", "HOLD", 50, "Neutral", old_time, False)
    )
    mock_get_db.return_value = mock_db
    assert is_thesis_stale("AAPL", hours=72) is True


@patch("app.pipeline.analysis.thesis_store.get_db")
def test_thesis_is_not_stale_within_window(mock_get_db):
    recent_time = datetime.now(timezone.utc) - timedelta(hours=10)
    mock_db = _mock_db_with_rows(
        ("AAPL", "BUY", 80, "Strong momentum", recent_time, False)
    )
    mock_get_db.return_value = mock_db
    assert is_thesis_stale("AAPL", hours=72) is False


@patch("app.pipeline.analysis.thesis_store.get_db")
def test_thesis_is_stale_when_none_exists(mock_get_db):
    mock_db = _mock_db_with_rows(None)
    mock_get_db.return_value = mock_db
    assert is_thesis_stale("FAKE", hours=72) is True


# ── mark_unchanged ─────────────────────────────────────────────────────

@patch("app.pipeline.analysis.thesis_store.get_db")
def test_mark_unchanged_calls_update(mock_get_db):
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_db
    mock_db.__enter__ = MagicMock(return_value=mock_db)
    mock_db.__exit__ = MagicMock(return_value=False)
    mock_get_db.return_value = mock_db

    mark_unchanged("NVDA")

    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args
    assert "thesis_unchanged = TRUE" in call_args[0][0]
    assert call_args[0][1] == ["NVDA"]

"""Tests for app.pipeline.data.watermark_store."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.data.watermark_store import (
    get_watermark,
    set_watermark,
    get_all_watermarks,
    _EPOCH,
)


def _mock_db_ctx(return_value=None):
    """Create a mock DB context manager."""
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_db
    mock_db.fetchone.return_value = return_value
    mock_db.fetchall.return_value = return_value if isinstance(return_value, list) else []
    mock_db.__enter__ = MagicMock(return_value=mock_db)
    mock_db.__exit__ = MagicMock(return_value=False)
    return mock_db


# ── get_watermark ──────────────────────────────────────────────────────

@patch("app.pipeline.data.watermark_store.get_db")
def test_get_watermark_returns_epoch_for_unknown(mock_get_db):
    mock_db = _mock_db_ctx(return_value=None)
    mock_get_db.return_value = mock_db
    assert get_watermark("FAKE", "news") == _EPOCH


@patch("app.pipeline.data.watermark_store.get_db")
def test_get_watermark_returns_stored_timestamp(mock_get_db):
    now = datetime.now(timezone.utc)
    mock_db = _mock_db_ctx(return_value=(now,))
    mock_get_db.return_value = mock_db
    assert get_watermark("AAPL", "news") == now


# ── set_watermark ──────────────────────────────────────────────────────

@patch("app.pipeline.data.watermark_store.get_db")
def test_set_watermark_calls_upsert(mock_get_db):
    mock_db = MagicMock()
    mock_db.execute.return_value = mock_db
    mock_db.__enter__ = MagicMock(return_value=mock_db)
    mock_db.__exit__ = MagicMock(return_value=False)
    mock_get_db.return_value = mock_db

    now = datetime.now(timezone.utc)
    set_watermark("AAPL", "news", now)

    mock_db.execute.assert_called_once()
    call_args = mock_db.execute.call_args
    assert "ON CONFLICT" in call_args[0][0]
    assert call_args[0][1] == ["AAPL", "news", now]


# ── get_all_watermarks ────────────────────────────────────────────────

@patch("app.pipeline.data.watermark_store.get_db")
def test_get_all_watermarks_returns_dict(mock_get_db):
    now = datetime.now(timezone.utc)
    mock_db = _mock_db_ctx(return_value=[("news", now), ("reddit", now)])
    mock_get_db.return_value = mock_db

    result = get_all_watermarks("AAPL")
    assert result == {"news": now, "reddit": now}


@patch("app.pipeline.data.watermark_store.get_db")
def test_get_all_watermarks_returns_empty_for_unknown(mock_get_db):
    mock_db = _mock_db_ctx(return_value=[])
    mock_get_db.return_value = mock_db
    assert get_all_watermarks("FAKE") == {}

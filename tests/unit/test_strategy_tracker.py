"""
Strategy Tracker Unit Tests — Verify per-prompt P&L tracking and benching.

Tests record_strategy, evaluate_pnl, compute_rankings, get_confidence_bonus,
and bench_underperformers using mocked DB.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def mock_db_ctx():
    """Provide a mock get_db context manager."""
    cursor = MagicMock()
    cursor.execute.return_value = cursor
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []

    @contextmanager
    def fake_get_db():
        yield cursor

    return fake_get_db, cursor


class TestRecordStrategy:
    """record_strategy should only record BUY/SELL signals."""

    def test_hold_signal_returns_none(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        with patch("app.trading.strategy_tracker.get_db", fake_get_db):
            from app.trading.strategy_tracker import record_strategy
            result = record_strategy("cand-1", "outcome-1", "hash123", "AAPL", "HOLD", 150.0)
        assert result is None

    def test_buy_signal_inserts_record(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        with patch("app.trading.strategy_tracker.get_db", fake_get_db):
            from app.trading.strategy_tracker import record_strategy
            result = record_strategy("cand-1", "outcome-1", "hash123", "AAPL", "BUY", 150.0)
        assert result is not None
        cursor.execute.assert_called_once()
        # Check that the INSERT was called with correct values
        call_args = cursor.execute.call_args
        assert "INSERT INTO strategy_performance" in call_args[0][0]

    def test_sell_signal_inserts_record(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        with patch("app.trading.strategy_tracker.get_db", fake_get_db):
            from app.trading.strategy_tracker import record_strategy
            result = record_strategy("cand-1", "outcome-1", "hash456", "MSFT", "SELL", 400.0)
        assert result is not None

    def test_db_failure_returns_none(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        cursor.execute.side_effect = Exception("DB connection lost")
        with patch("app.trading.strategy_tracker.get_db", fake_get_db):
            from app.trading.strategy_tracker import record_strategy
            result = record_strategy("cand-1", "outcome-1", "hash789", "NVDA", "BUY", 800.0)
        assert result is None


class TestEvaluatePnl:
    """evaluate_pnl should resolve open BUY entries for a closed trade."""

    def test_no_open_entries_returns_empty(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        cursor.fetchall.return_value = []
        with patch("app.trading.strategy_tracker.get_db", fake_get_db):
            from app.trading.strategy_tracker import evaluate_pnl
            result = evaluate_pnl("AAPL", exit_price=165.0)
        assert result == []

    def test_win_trade_resolved_correctly(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        # Return one open BUY entry
        cursor.fetchall.return_value = [
            ("perf-001", 150.0, "BUY", "hash123"),
        ]
        # fetchone for created_at lookup
        from datetime import datetime, timezone
        cursor.fetchone.return_value = (datetime.now(timezone.utc).isoformat(),)

        with patch("app.trading.strategy_tracker.get_db", fake_get_db):
            from app.trading.strategy_tracker import evaluate_pnl
            result = evaluate_pnl("AAPL", exit_price=165.0)

        assert len(result) == 1
        assert result[0]["win"] is True
        assert result[0]["return_pct"] == pytest.approx(10.0, abs=0.1)

    def test_loss_trade_resolved_correctly(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        cursor.fetchall.return_value = [
            ("perf-002", 200.0, "BUY", "hash456"),
        ]
        from datetime import datetime, timezone
        cursor.fetchone.return_value = (datetime.now(timezone.utc).isoformat(),)

        with patch("app.trading.strategy_tracker.get_db", fake_get_db):
            from app.trading.strategy_tracker import evaluate_pnl
            result = evaluate_pnl("MSFT", exit_price=180.0)

        assert len(result) == 1
        assert result[0]["win"] is False
        assert result[0]["return_pct"] < 0


class TestGetConfidenceBonus:
    """get_confidence_bonus should return +5 for winning prompts."""

    def test_no_trades_returns_zero(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        cursor.fetchone.return_value = (0, 0.0)

        with patch("app.trading.strategy_tracker.get_db", fake_get_db), \
             patch("app.trading.strategy_tracker.settings") as mock_settings:
            mock_settings.MIN_TRADES_BEFORE_BENCH = 10
            mock_settings.WIN_RATE_BONUS_THRESHOLD = 0.55
            from app.trading.strategy_tracker import get_confidence_bonus
            result = get_confidence_bonus("hash123")

        assert result == 0

    def test_high_win_rate_returns_bonus(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        cursor.fetchone.return_value = (15, 0.60)  # 60% win rate, 15 trades

        with patch("app.trading.strategy_tracker.get_db", fake_get_db), \
             patch("app.trading.strategy_tracker.settings") as mock_settings:
            mock_settings.MIN_TRADES_BEFORE_BENCH = 10
            mock_settings.WIN_RATE_BONUS_THRESHOLD = 0.55
            from app.trading.strategy_tracker import get_confidence_bonus
            result = get_confidence_bonus("hash123")

        assert result == 5

    def test_low_win_rate_returns_zero(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        cursor.fetchone.return_value = (20, 0.35)  # 35% win rate

        with patch("app.trading.strategy_tracker.get_db", fake_get_db), \
             patch("app.trading.strategy_tracker.settings") as mock_settings:
            mock_settings.MIN_TRADES_BEFORE_BENCH = 10
            mock_settings.WIN_RATE_BONUS_THRESHOLD = 0.55
            from app.trading.strategy_tracker import get_confidence_bonus
            result = get_confidence_bonus("hash123")

        assert result == 0


class TestBenchUnderperformers:
    """bench_underperformers should deactivate low win rate prompts."""

    def test_no_underperformers_returns_empty(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        cursor.fetchall.return_value = []

        with patch("app.trading.strategy_tracker.get_db", fake_get_db), \
             patch("app.trading.strategy_tracker.settings") as mock_settings:
            mock_settings.MIN_TRADES_BEFORE_BENCH = 10
            mock_settings.WIN_RATE_BENCH_THRESHOLD = 0.40
            from app.trading.strategy_tracker import bench_underperformers
            result = bench_underperformers()

        assert result == []

    def test_underperformer_gets_benched(self, mock_db_ctx):
        fake_get_db, cursor = mock_db_ctx
        cursor.fetchall.return_value = [
            ("hash_bad", "Bad Strategy", 15, 0.25),  # 25% win rate
        ]

        with patch("app.trading.strategy_tracker.get_db", fake_get_db), \
             patch("app.trading.strategy_tracker.settings") as mock_settings:
            mock_settings.MIN_TRADES_BEFORE_BENCH = 10
            mock_settings.WIN_RATE_BENCH_THRESHOLD = 0.40
            from app.trading.strategy_tracker import bench_underperformers
            result = bench_underperformers()

        assert "hash_bad" in result
        # Verify UPDATE was called to set active = FALSE
        update_calls = [
            c for c in cursor.execute.call_args_list
            if "UPDATE generated_agent_prompts" in str(c)
        ]
        assert len(update_calls) >= 1

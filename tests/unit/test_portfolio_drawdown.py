"""
Portfolio Drawdown Unit Tests — Verify max drawdown calculations.

Tests compute_portfolio_drawdown with various P&L sequences:
  1. No trades → returns None
  2. All winning trades → drawdown is 0
  3. All losing trades → drawdown equals total loss
  4. Mixed trades → correct peak-to-trough drawdown
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_mock_db(pnl_rows):
    """Create a mock DB cursor that returns lot_closures P&L rows."""
    cursor = MagicMock()
    cursor.execute.return_value = cursor
    cursor.fetchall.return_value = [(pnl,) for pnl in pnl_rows]
    return cursor


class TestComputePortfolioDrawdown:
    """compute_portfolio_drawdown should correctly calculate max drawdown."""

    def test_no_trades_returns_none(self):
        db = _make_mock_db([])
        with patch("app.config.settings") as mock_settings:
            mock_settings.BOT_ID = "test-bot"
            from app.trading.portfolio_drawdown import compute_portfolio_drawdown
            result = compute_portfolio_drawdown(db, initial_cash=100_000.0)
        assert result is None

    def test_all_winning_trades_zero_drawdown(self):
        # Equity only goes up → drawdown should be 0
        pnls = [500.0, 300.0, 700.0, 200.0]
        db = _make_mock_db(pnls)
        with patch("app.config.settings") as mock_settings:
            mock_settings.BOT_ID = "test-bot"
            from app.trading.portfolio_drawdown import compute_portfolio_drawdown
            result = compute_portfolio_drawdown(db, initial_cash=100_000.0)
        assert result == 0.0

    def test_all_losing_trades(self):
        # Equity only goes down
        pnls = [-1000.0, -2000.0, -500.0]
        db = _make_mock_db(pnls)
        with patch("app.config.settings") as mock_settings:
            mock_settings.BOT_ID = "test-bot"
            from app.trading.portfolio_drawdown import compute_portfolio_drawdown
            result = compute_portfolio_drawdown(db, initial_cash=100_000.0)

        # Total loss = -3500, peak was 100,000
        # Equity at end = 96,500. DD = (96500 - 100000) / 100000 = -0.035
        assert result < 0
        assert abs(result - (-0.035)) < 0.001

    def test_mixed_trades_correct_drawdown(self):
        # Win, win, big loss, win → max drawdown during the loss
        # Start: 100k → 101k → 103k → 93k → 95k
        # Peak at 103k, trough at 93k → DD = (93k-103k)/103k ≈ -9.7%
        pnls = [1000.0, 2000.0, -10000.0, 2000.0]
        db = _make_mock_db(pnls)
        with patch("app.config.settings") as mock_settings:
            mock_settings.BOT_ID = "test-bot"
            from app.trading.portfolio_drawdown import compute_portfolio_drawdown
            result = compute_portfolio_drawdown(db, initial_cash=100_000.0)

        expected_dd = (93_000.0 - 103_000.0) / 103_000.0  # ≈ -0.0971
        assert result < 0
        assert abs(result - expected_dd) < 0.001

    def test_recovery_after_drawdown(self):
        # Big loss then full recovery → max drawdown still recorded
        # 100k → 80k → 120k
        pnls = [-20000.0, 40000.0]
        db = _make_mock_db(pnls)
        with patch("app.config.settings") as mock_settings:
            mock_settings.BOT_ID = "test-bot"
            from app.trading.portfolio_drawdown import compute_portfolio_drawdown
            result = compute_portfolio_drawdown(db, initial_cash=100_000.0)

        # Peak was 100k, trough was 80k → DD = -20%
        expected_dd = (80_000.0 - 100_000.0) / 100_000.0
        assert abs(result - expected_dd) < 0.001

    def test_single_trade_loss(self):
        pnls = [-5000.0]
        db = _make_mock_db(pnls)
        with patch("app.config.settings") as mock_settings:
            mock_settings.BOT_ID = "test-bot"
            from app.trading.portfolio_drawdown import compute_portfolio_drawdown
            result = compute_portfolio_drawdown(db, initial_cash=100_000.0)

        assert result == pytest.approx(-0.05, abs=0.001)

    def test_single_trade_win(self):
        pnls = [5000.0]
        db = _make_mock_db(pnls)
        with patch("app.config.settings") as mock_settings:
            mock_settings.BOT_ID = "test-bot"
            from app.trading.portfolio_drawdown import compute_portfolio_drawdown
            result = compute_portfolio_drawdown(db, initial_cash=100_000.0)

        assert result == 0.0

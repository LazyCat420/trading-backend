"""
Unit tests for the Quant Edge Verifier.
Verifies backtesting logic on mock DataFrame price inputs.
"""

import pandas as pd
import numpy as np
import pytest
from app.trading.quant_edge_verifier import (
    backtest_zscore_strategy,
    backtest_rsi_macd_strategy,
    backtest_stop_loss_comparison,
    backtest_spec_strategy,
)

@pytest.fixture
def dummy_price_data():
    """Generates a dummy DataFrame of prices, Z-scores, RSI, and MACD."""
    dates = pd.date_range(start="2026-01-01", periods=10, freq="D")
    data = {
        "close": [100.0, 95.0, 90.0, 85.0, 92.0, 98.0, 102.0, 105.0, 103.0, 101.0],
        "z_score": [0.0, -1.0, -1.8, -2.5, -0.5, 0.2, 1.1, 2.0, 0.8, -0.1],
        "rsi_14": [50.0, 40.0, 35.0, 25.0, 45.0, 65.0, 75.0, 80.0, 55.0, 48.0],
        "macd_hist": [0.1, -0.2, -0.5, 0.3, 0.8, 1.2, 0.4, -0.2, -0.8, -0.1],
        "atr_14": [2.0, 2.1, 2.2, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.8],
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_backtest_zscore_strategy(dummy_price_data):
    """
    On day 4 (index 3), z_score is -2.5 (<= -2.0) -> Buy at 85.0.
    On day 6 (index 5), z_score is 0.2 (>= 0.0) -> Sell at 98.0.
    Expected return = (98 - 85)/85 * 100 = 15.29%.
    """
    res = backtest_zscore_strategy(dummy_price_data, entry_z=-2.0, exit_z=0.0)
    
    assert res["total_trades"] == 1
    assert len(res["trades"]) == 1
    trade = res["trades"][0]
    
    assert trade["entry_price"] == 85.0
    assert trade["exit_price"] == 98.0
    assert abs(trade["return_pct"] - 15.29) < 0.1
    assert res["win_rate_pct"] == 100.0
    assert res["cumulative_return_pct"] > 0.0

def test_backtest_rsi_macd_strategy(dummy_price_data):
    """
    On day 4 (index 3), RSI is 25.0 (< 30) and macd_hist is 0.3 (> 0) -> Buy at 85.0.
    On day 7 (index 6), RSI is 75.0 (> 70) -> Sell at 102.0.
    Expected return = (102 - 85)/85 * 100 = 20.0%.
    """
    res = backtest_rsi_macd_strategy(dummy_price_data, buy_rsi=30.0, sell_rsi=70.0)
    
    assert res["total_trades"] == 1
    trade = res["trades"][0]
    assert trade["entry_price"] == 85.0
    assert trade["exit_price"] == 102.0
    assert abs(trade["return_pct"] - 20.0) < 0.1
    assert res["win_rate_pct"] == 100.0

def test_backtest_stop_loss_comparison_fixed(dummy_price_data):
    """
    RSI drops < 30 on day 4 (index 3) -> Buy at 85.0.
    Exit 1: fixed 5% stop loss (stop price 80.75) vs 10% target (93.5).
    On day 6 (index 5), close is 98.0 which is >= profit target of 93.5 -> Exit.
    Expected return = (98 - 85)/85 * 100 = 15.29%.
    """
    res = backtest_stop_loss_comparison(dummy_price_data, use_atr=False, fixed_stop_pct=0.05)
    
    assert res["total_trades"] == 1
    trade = res["trades"][0]
    assert trade["entry_price"] == 85.0
    assert trade["exit_price"] == 98.0
    assert abs(trade["return_pct"] - 15.29) < 0.1

def test_backtest_spec_strategy(dummy_price_data):
    """
    On day 4 (index 3), z_score_norm is -2.5 mapped -> ~0.08, ev_norm = 0.6, rr_norm = 0.17, kelly_norm = 0.4.
    Formula evaluates to base score.
    Verifies that the spec strategy runs and returns trade dictionary.
    """
    res = backtest_spec_strategy(dummy_price_data, entry_threshold=5.0, exit_threshold=4.5)
    assert "total_trades" in res
    assert isinstance(res["trades"], list)

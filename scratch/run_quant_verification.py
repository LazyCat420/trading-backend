"""
Run Quant Verification on actual historical data.
Fetches tickers from the DB and runs verifications.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app.trading.quant_edge_verifier import (
    load_historical_data,
    backtest_zscore_strategy,
    backtest_rsi_macd_strategy,
    backtest_stop_loss_comparison,
)
from app.db.connection import get_db

def get_available_tickers():
    """Fetch tickers that have data in both price_history and technicals."""
    with get_db() as db:
        rows = db.execute(
            """
            SELECT DISTINCT p.ticker 
            FROM price_history p
            JOIN technicals t ON p.ticker = t.ticker
            LIMIT 5
            """
        ).fetchall()
        return [r[0] for r in rows]

def run_verification():
    tickers = get_available_tickers()
    if not tickers:
        print("No tickers with both price and technical history found in DB.")
        # Fallback to whatever tickers are in price_history
        with get_db() as db:
            rows = db.execute("SELECT DISTINCT ticker FROM price_history LIMIT 5").fetchall()
            tickers = [r[0] for r in rows]
            
    if not tickers:
        print("No price history found in DB.")
        return

    print(f"Found tickers to verify: {tickers}")
    print("=" * 90)
    print(f"{'Ticker':<8} | {'Strategy':<20} | {'Trades':<6} | {'Win Rate':<10} | {'Cum Return':<12} | {'Max DD':<8}")
    print("-" * 90)
    
    for ticker in tickers:
        df = load_historical_data(ticker)
        if df.empty or len(df) < 20:
            continue
            
        # 1. Z-Score Mean Reversion
        zscore_res = backtest_zscore_strategy(df)
        if "error" not in zscore_res:
            print(f"{ticker:<8} | {'Z-Score Reversion':<20} | {zscore_res['total_trades']:<6} | {zscore_res['win_rate_pct']:.1f}%     | {zscore_res['cumulative_return_pct']:.2f}%      | {zscore_res['max_drawdown_pct']:.2f}%")
            
        # 2. RSI + MACD
        rsi_macd_res = backtest_rsi_macd_strategy(df)
        if "error" not in rsi_macd_res:
            print(f"{ticker:<8} | {'RSI + MACD':<20} | {rsi_macd_res['total_trades']:<6} | {rsi_macd_res['win_rate_pct']:.1f}%     | {rsi_macd_res['cumulative_return_pct']:.2f}%      | {rsi_macd_res['max_drawdown_pct']:.2f}%")
            
        # 3. Fixed Stop Loss vs ATR Stop Loss
        fixed_sl_res = backtest_stop_loss_comparison(df, use_atr=False)
        atr_sl_res = backtest_stop_loss_comparison(df, use_atr=True)
        
        if "error" not in fixed_sl_res:
            print(f"{ticker:<8} | {'Fixed 5% Stop':<20} | {fixed_sl_res['total_trades']:<6} | {fixed_sl_res['win_rate_pct']:.1f}%     | {fixed_sl_res['cumulative_return_pct']:.2f}%      | {fixed_sl_res['max_drawdown_pct']:.2f}%")
        if "error" not in atr_sl_res:
            print(f"{ticker:<8} | {'ATR Trailing Stop':<20} | {atr_sl_res['total_trades']:<6} | {atr_sl_res['win_rate_pct']:.1f}%     | {atr_sl_res['cumulative_return_pct']:.2f}%      | {atr_sl_res['max_drawdown_pct']:.2f}%")
            
        print("-" * 90)

if __name__ == "__main__":
    run_verification()

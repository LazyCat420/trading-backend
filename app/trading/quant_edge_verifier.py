"""
Quant Edge Verifier.
Provides a testing framework to back-test and verify mathematical formulas
and indicators (e.g., Z-score, RSI, MACD, Sharpe, ATR-based stops) to
measure if they provide a statistical trading edge on historical data.
"""

import logging
import numpy as np
import pandas as pd
from app.db.connection import get_db

logger = logging.getLogger(__name__)

def load_historical_data(ticker: str) -> pd.DataFrame:
    """
    Fetch price history and technical indicators merged on date.
    """
    with get_db() as db:
        prices_df = _cursor_to_df(
            db.execute(
                """
                SELECT date, open, high, low, close, volume 
                FROM price_history 
                WHERE ticker = %s 
                ORDER BY date ASC
                """,
                [ticker]
            )
        )
        if prices_df.empty:
            return pd.DataFrame()
            
        tech_df = _cursor_to_df(
            db.execute(
                """
                SELECT date, rsi_14, macd, macd_signal, macd_hist, atr_14, support, resistance 
                FROM technicals 
                WHERE ticker = %s 
                ORDER BY date ASC
                """,
                [ticker]
            )
        )
        
        if tech_df.empty:
            # If no technicals table rows, compute rolling Z-score only
            prices_df["date"] = pd.to_datetime(prices_df["date"])
            return prices_df.set_index("date")
            
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        tech_df["date"] = pd.to_datetime(tech_df["date"])
        
        merged = pd.merge(prices_df, tech_df, on="date", how="left")
        merged = merged.sort_values("date").set_index("date")
        
        # Backfill rolling window variables if missing
        if "close" in merged:
            rolling_mean = merged["close"].rolling(window=60).mean()
            rolling_std = merged["close"].rolling(window=60).std()
            merged["z_score"] = (merged["close"] - rolling_mean) / rolling_std
            
        return merged

def _cursor_to_df(cursor) -> pd.DataFrame:
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=cols)

def backtest_zscore_strategy(df: pd.DataFrame, entry_z: float = -2.0, exit_z: float = 0.0) -> dict:
    """
    Backtests a simple Z-score mean reversion strategy:
    - BUY when Z-score <= entry_z
    - SELL/EXIT when Z-score >= exit_z
    """
    if df.empty or "z_score" not in df:
        return {"error": "Insufficient data or missing Z-score"}
        
    position = 0  # 0 = cash, 1 = long
    entry_price = 0.0
    trades = []
    
    for date, row in df.iterrows():
        z = row["z_score"]
        close = row["close"]
        
        if pd.isna(z) or pd.isna(close):
            continue
            
        if position == 0 and z <= entry_z:
            position = 1
            entry_price = close
            entry_date = date
        elif position == 1 and z >= exit_z:
            position = 0
            pnl = (close - entry_price) / entry_price
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "entry_price": entry_price,
                "exit_price": close,
                "return_pct": pnl * 100
            })
            
    # Handle open position at end
    if position == 1:
        close = df["close"].iloc[-1]
        pnl = (close - entry_price) / entry_price
        trades.append({
            "entry_date": entry_date,
            "exit_date": df.index[-1],
            "entry_price": entry_price,
            "exit_price": close,
            "return_pct": pnl * 100,
            "open": True
        })
        
    return _summarize_trades(trades)

def backtest_rsi_macd_strategy(df: pd.DataFrame, buy_rsi: float = 30.0, sell_rsi: float = 70.0) -> dict:
    """
    Backtests an RSI and MACD confirmation strategy:
    - BUY when RSI < buy_rsi AND MACD histogram > 0
    - SELL/EXIT when RSI > sell_rsi
    """
    if df.empty or "rsi_14" not in df or "macd_hist" not in df:
        return {"error": "Missing RSI or MACD indicators"}
        
    position = 0
    entry_price = 0.0
    trades = []
    
    for date, row in df.iterrows():
        rsi = row["rsi_14"]
        macd_h = row["macd_hist"]
        close = row["close"]
        
        if pd.isna(rsi) or pd.isna(macd_h) or pd.isna(close):
            continue
            
        if position == 0 and rsi < buy_rsi and macd_h > 0:
            position = 1
            entry_price = close
            entry_date = date
        elif position == 1 and rsi > sell_rsi:
            position = 0
            pnl = (close - entry_price) / entry_price
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "entry_price": entry_price,
                "exit_price": close,
                "return_pct": pnl * 100
            })
            
    # Handle open position at end
    if position == 1:
        close = df["close"].iloc[-1]
        pnl = (close - entry_price) / entry_price
        trades.append({
            "entry_date": entry_date,
            "exit_date": df.index[-1],
            "entry_price": entry_price,
            "exit_price": close,
            "return_pct": pnl * 100,
            "open": True
        })
        
    return _summarize_trades(trades)

def backtest_stop_loss_comparison(df: pd.DataFrame, use_atr: bool = False, fixed_stop_pct: float = 0.05, atr_multiplier: float = 2.0) -> dict:
    """
    Enters a trade when RSI < 30 (simple entry) and compares exits:
    - Exit 1: 5% Fixed stop loss vs 10% profit target
    - Exit 2 (ATR): Trailing stop at atr_multiplier * ATR from peak price since entry vs 10% profit target
    """
    if df.empty or "rsi_14" not in df or (use_atr and "atr_14" not in df):
        return {"error": "Missing required indicators for Stop Loss comparison"}
        
    position = 0
    entry_price = 0.0
    peak_price = 0.0
    trades = []
    
    for date, row in df.iterrows():
        rsi = row["rsi_14"]
        close = row["close"]
        atr = row.get("atr_14", 0.0)
        
        if pd.isna(rsi) or pd.isna(close):
            continue
            
        if position == 0 and rsi < 30.0:
            position = 1
            entry_price = close
            peak_price = close
            entry_date = date
        elif position == 1:
            peak_price = max(peak_price, close)
            
            # Compute stop price
            if use_atr:
                stop_price = peak_price - (atr_multiplier * atr)
            else:
                stop_price = entry_price * (1 - fixed_stop_pct)
                
            profit_target = entry_price * 1.10
            
            # Check exit conditions
            if close <= stop_price or close >= profit_target:
                position = 0
                pnl = (close - entry_price) / entry_price
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": close,
                    "return_pct": pnl * 100
                })
                
    if position == 1:
        close = df["close"].iloc[-1]
        pnl = (close - entry_price) / entry_price
        trades.append({
            "entry_date": entry_date,
            "exit_date": df.index[-1],
            "entry_price": entry_price,
            "exit_price": close,
            "return_pct": pnl * 100,
            "open": True
        })
        
    return _summarize_trades(trades)

def _summarize_trades(trades: list) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "average_return_pct": 0.0,
            "cumulative_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": []
        }
        
    df_trades = pd.DataFrame(trades)
    win_rate = (df_trades["return_pct"] > 0).mean() * 100
    avg_return = df_trades["return_pct"].mean()
    cum_return = ( (1 + df_trades["return_pct"] / 100).prod() - 1 ) * 100
    
    # Calculate Drawdown of the trade equity curve
    equity = (1 + df_trades["return_pct"] / 100).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100
    
    return {
        "total_trades": len(trades),
        "win_rate_pct": round(float(win_rate), 2),
        "average_return_pct": round(float(avg_return), 3),
        "cumulative_return_pct": round(float(cum_return), 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "trades": trades
    }

def backtest_spec_strategy(df: pd.DataFrame, entry_threshold: float = 7.0, exit_threshold: float = 4.0) -> dict:
    """
    Backtests the dynamic spec-based Edge Score strategy.
    - Computes Edge Score dynamically for each row using the active spec.
    - BUY when edge_score >= entry_threshold.
    - SELL/EXIT when edge_score < exit_threshold.
    """
    from app.trading.scoring_engine import calculate_pillar_score
    
    if df.empty:
        return {"error": "DataFrame is empty"}
        
    position = 0
    entry_price = 0.0
    trades = []
    
    # Pre-calculate normalized variables to match scoring_engine format
    df = df.copy()
    if "rsi_14" in df:
        df["rsi_norm"] = df["rsi_14"] / 100.0
    else:
        df["rsi_norm"] = 0.5
        
    if "macd_hist" in df:
        # Simple normalization
        df["macd_norm"] = 1.0 / (1.0 + np.exp(-df["macd_hist"]))
    else:
        df["macd_norm"] = 0.5
        
    if "z_score" in df:
        df["z_score_norm"] = np.clip((df["z_score"] + 3.0) / 6.0, 0.0, 1.0)
    else:
        df["z_score_norm"] = 0.5
        
    df["ev_norm"] = 0.6
    df["rr_norm"] = 0.5
    df["kelly_norm"] = 0.4
    df["vol_norm"] = 0.3
    df["dd_norm"] = 0.2
    df["beta_norm"] = 0.5
    
    # Calculate score series
    scores = []
    for date, row in df.iterrows():
        variables = {k: float(row[k]) for k in ["ev_norm", "rr_norm", "kelly_norm", "vol_norm", "dd_norm", "beta_norm", "z_score_norm", "rsi_norm"] if k in row}
        for k in ["ev_norm", "rr_norm", "kelly_norm", "vol_norm", "dd_norm", "beta_norm", "z_score_norm", "rsi_norm"]:
            if k not in variables:
                variables[k] = float(row.get(k, 0.5))
                
        score = calculate_pillar_score("edge_score", variables)
        scores.append(score)
        
    df["edge_score"] = scores
    
    for date, row in df.iterrows():
        score = row["edge_score"]
        close = row["close"]
        
        if pd.isna(score) or pd.isna(close):
            continue
            
        if position == 0 and score >= entry_threshold:
            position = 1
            entry_price = close
            entry_date = date
        elif position == 1 and score < exit_threshold:
            position = 0
            pnl = (close - entry_price) / entry_price
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "entry_price": entry_price,
                "exit_price": close,
                "return_pct": pnl * 100
            })
            
    if position == 1:
        close = df["close"].iloc[-1]
        pnl = (close - entry_price) / entry_price
        trades.append({
            "entry_date": entry_date,
            "exit_date": df.index[-1],
            "entry_price": entry_price,
            "exit_price": close,
            "return_pct": pnl * 100,
            "open": True
        })
        
    return _summarize_trades(trades)

"""
Technical Processor — compute indicators from price_history → technicals table.

Pure Python + ta library. No LLM calls. No hallucinations.
"""

import pandas as pd
import ta
from app.db.connection import get_db


import logging

logger = logging.getLogger(__name__)


def compute_technicals(ticker: str, period: int = 500) -> int:
    """
    Compute all technical indicators for a ticker and write to technicals table.
    Needs at least 5 rows (more = better indicators).
    Returns number of rows written.
    """
    with get_db() as db:
        # Fetch price history
        rows = db.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE ticker = %s
            ORDER BY date ASC
            LIMIT %s
        """,
            [ticker, period],
        ).fetchall()

        if not rows or len(rows) < 5:
            logger.debug(
                "[tech] %s: not enough price data (%d rows, need >=5)",
                ticker,
                len(rows) if rows else 0,
            )
            return 0

        df = pd.DataFrame(
            rows, columns=["date", "open", "high", "low", "close", "volume"]
        )

        # ── Trend indicators ──
        df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
        df["sma_200"] = ta.trend.sma_indicator(df["close"], window=200)
        df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
        df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)

        # MACD
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # ADX
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx_14"] = adx.adx()

        # ── Momentum indicators ──
        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # ── Volatility indicators ──
        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()

        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        )
        df["atr_14"] = atr.average_true_range()

        # ── Volume indicators ──
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"].astype(float))
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        # ── Support / Resistance (simple: recent swing low/high) ──
        lookback = min(20, len(df))
        df["support"] = df["low"].rolling(lookback).min()
        df["resistance"] = df["high"].rolling(lookback).max()

        # ── Write to DB (only rows with valid RSI, i.e. skip first 13) ──
        valid = df.dropna(subset=["rsi_14"])

        rows_to_insert = []
        for _, row in valid.iterrows():
            rows_to_insert.append(
                [
                    ticker,
                    row["date"],
                    _f(row["rsi_14"]),
                    _f(row["macd"]),
                    _f(row["macd_signal"]),
                    _f(row["macd_hist"]),
                    _f(row["sma_20"]),
                    _f(row["sma_50"]),
                    _f(row["sma_200"]),
                    _f(row["ema_12"]),
                    _f(row["ema_26"]),
                    _f(row["bb_upper"]),
                    _f(row["bb_mid"]),
                    _f(row["bb_lower"]),
                    _f(row["atr_14"]),
                    _f(row["adx_14"]),
                    _f(row["stoch_k"]),
                    _f(row["stoch_d"]),
                    _f(row["obv"]),
                    _f(row["vwap"]),
                    _f(row["support"]),
                    _f(row["resistance"]),
                ]
            )

        if rows_to_insert:
            for r in rows_to_insert:
                db.execute(
                    """
                    INSERT INTO technicals
                    (ticker, date, rsi_14, macd, macd_signal, macd_hist,
                     sma_20, sma_50, sma_200, ema_12, ema_26,
                     bb_upper, bb_mid, bb_lower, atr_14, adx_14,
                     stoch_k, stoch_d, obv, vwap, support, resistance)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date) DO NOTHING
                """,
                    r,
                )

        count = len(rows_to_insert)

        logger.debug("[tech] %s: %d technical rows written", ticker, count)
        return count


def get_signals(ticker: str) -> str:
    """
    Get the latest technical signals as pre-formatted text for the LLM.
    Returns a human-readable summary string the agent can analyze.
    """
    with get_db() as db:
        row = db.execute(
            """
            SELECT * FROM technicals
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """,
            [ticker],
        ).fetchone()

        if not row:
            return f"No technical data available for {ticker}."

        cols = [
            d[0] for d in db.execute("SELECT * FROM technicals LIMIT 0").description
        ]
        data = dict(zip(cols, row))

        # Build labeled signal text
        lines = [f"=== TECHNICAL ANALYSIS: {ticker} (as of {data['date']}) ==="]

        # RSI
        rsi = data.get("rsi_14")
        if rsi is not None:
            label = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
            lines.append(f"RSI-14: {rsi:.1f} ({label})")

        # MACD
        macd_h = data.get("macd_hist")
        if macd_h is not None:
            signal = "BULLISH" if macd_h > 0 else "BEARISH"
            lines.append(f"MACD histogram: {macd_h:.4f} ({signal})")

        # Moving averages
        close = db.execute(
            "SELECT close FROM price_history WHERE ticker = %s ORDER BY date DESC LIMIT 1",
            [ticker],
        ).fetchone()
        if close:
            price = close[0]
            lines.append(f"Price: ${price:.2f}")
            for ma_name in ["sma_20", "sma_50", "sma_200"]:
                val = data.get(ma_name)
                if val:
                    pos = "ABOVE" if price > val else "BELOW"
                    lines.append(f"  {ma_name.upper()}: ${val:.2f} (price {pos})")

        # Bollinger Bands
        bb_u, bb_l = data.get("bb_upper"), data.get("bb_lower")
        if bb_u and bb_l and close:
            pct = (price - bb_l) / (bb_u - bb_l) * 100 if bb_u != bb_l else 50
            band_pos = (
                "UPPER BAND" if pct > 80 else "LOWER BAND" if pct < 20 else "MID RANGE"
            )
            lines.append(f"Bollinger: {band_pos} ({pct:.0f}% width)")

        # Stochastic
        k, d = data.get("stoch_k"), data.get("stoch_d")
        if k is not None:
            label = "OVERBOUGHT" if k > 80 else "OVERSOLD" if k < 20 else "NEUTRAL"
            lines.append(f"Stochastic K/D: {k:.1f}/{d:.1f} ({label})")

        # ATR
        atr = data.get("atr_14")
        if atr is not None:
            lines.append(f"ATR-14: ${atr:.2f} (daily volatility)")

        # ADX
        adx = data.get("adx_14")
        if adx is not None:
            strength = "STRONG TREND" if adx > 25 else "WEAK/NO TREND"
            lines.append(f"ADX-14: {adx:.1f} ({strength})")

        # Support/Resistance
        sup, res = data.get("support"), data.get("resistance")
        if sup and res:
            lines.append(f"Support: ${sup:.2f} | Resistance: ${res:.2f}")

        return "\n".join(lines)


def _f(val) -> float | None:
    """Convert to float, handling NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return float(val)

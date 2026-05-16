"""
Backtest Data Provider — prepares fixed OOS backtest windows for the evolution loop.

Pulls OHLCV data from the existing PostgreSQL price_history table and exports
it as a Parquet file for the sandbox executor.
"""

import logging
import os
import tempfile

import pandas as pd

from app.db.connection import get_db

logger = logging.getLogger(__name__)


def get_backtest_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_path: str | None = None,
    allow_synthetic: bool = False,
) -> str:
    """Extract OHLCV data for the backtest window and save as Parquet.

    Args:
        tickers: List of tickers to include (e.g. ["SPY", "QQQ"]).
        start_date: Start of OOS window (YYYY-MM-DD).
        end_date: End of OOS window (YYYY-MM-DD).
        output_path: Where to save the Parquet file. If None, uses a temp file.
        allow_synthetic: If True, fall back to synthetic data when no real data
            is found. Should only be True in unit tests.

    Returns:
        Path to the Parquet file.
    """
    placeholders = ", ".join(["%s" for _ in tickers])
    with get_db() as db:
        rows = db.execute(
            f"SELECT ticker, date, open, high, low, close, volume "
            f"FROM price_history "
            f"WHERE ticker IN ({placeholders}) "
            f"AND date >= %s AND date <= %s "
            f"ORDER BY date",
            [*tickers, start_date, end_date],
        ).fetchall()

        if not rows:
            # Fallback: try asset_prices table
            rows = db.execute(
                f"SELECT symbol as ticker, date, open, high, low, close, volume "
                f"FROM asset_prices "
                f"WHERE symbol IN ({placeholders}) "
                f"AND date >= %s AND date <= %s "
                f"ORDER BY date",
                [*tickers, start_date, end_date],
            ).fetchall()

    if not rows:
        if not allow_synthetic:
            raise ValueError(
                f"No price data for {tickers} between {start_date}\u2013{end_date}. "
                "Pass allow_synthetic=True only for unit tests."
            )
        logger.warning(
            "SYNTHETIC DATA IN USE \u2014 evolution scores will be meaningless"
        )
        return _generate_synthetic_data(tickers, start_date, end_date, output_path)

    df = pd.DataFrame(
        rows, columns=["ticker", "date", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".parquet", prefix="backtest_")
        os.close(fd)

    df.to_parquet(output_path)
    logger.info("Backtest data exported: %d rows → %s", len(df), output_path)
    return output_path


def _generate_synthetic_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_path: str | None = None,
) -> str:
    """Generate synthetic OHLCV data when no real data is available."""
    import numpy as np

    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    if n == 0:
        dates = pd.bdate_range(start="2023-01-01", end="2024-01-01")
        n = len(dates)

    if not tickers:
        tickers = ["SYNTH"]

    frames = []
    for i, symbol in enumerate(tickers):
        np.random.seed(42 + i)
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8, n)
        volume = np.random.randint(1_000_000, 50_000_000, n)

        df = pd.DataFrame(
            {
                "ticker": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        df.index.name = "date"
        frames.append(df)

    combined = pd.concat(frames)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".parquet", prefix="backtest_synth_")
        os.close(fd)

    combined.to_parquet(output_path)
    return output_path

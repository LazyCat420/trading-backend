"""
Market Regime Collector — Fetches indexes, futures, VIX, yields, sector ETFs, dollar.

Stores into `asset_prices` table with asset_class distinguishing them.
All data fetched via yfinance batch download for efficiency.
"""

import logging
import yfinance as yf
import asyncio
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ─── Ticker Universe ───────────────────────────────────────────
MARKET_TICKERS: dict[str, dict] = {
    # Indexes
    "^GSPC": {"name": "S&P 500", "class": "index"},
    "^IXIC": {"name": "NASDAQ Composite", "class": "index"},
    "^RUT": {"name": "Russell 2000", "class": "index"},
    "^DJI": {"name": "Dow Jones", "class": "index"},
    # Futures
    "ES=F": {"name": "S&P 500 Futures", "class": "futures"},
    "NQ=F": {"name": "NASDAQ Futures", "class": "futures"},
    "YM=F": {"name": "Dow Futures", "class": "futures"},
    "RTY=F": {"name": "Russell Futures", "class": "futures"},
    # Volatility
    "^VIX": {"name": "VIX", "class": "volatility"},
    "^VIX3M": {"name": "VIX 3-Month", "class": "volatility"},
    # Treasury Yields (yfinance reports as price = yield %)
    "^IRX": {"name": "13-Week T-Bill", "class": "yield"},
    "^FVX": {"name": "5Y Treasury", "class": "yield"},
    "^TNX": {"name": "10Y Treasury", "class": "yield"},
    "^TYX": {"name": "30Y Treasury", "class": "yield"},
    # Sector ETFs (clean sector benchmarks)
    "XLE": {"name": "Energy ETF", "class": "sector_etf"},
    "XLF": {"name": "Financials ETF", "class": "sector_etf"},
    "XLK": {"name": "Technology ETF", "class": "sector_etf"},
    "XLV": {"name": "Healthcare ETF", "class": "sector_etf"},
    "XLY": {"name": "Consumer Disc. ETF", "class": "sector_etf"},
    "XLP": {"name": "Consumer Staples ETF", "class": "sector_etf"},
    "XLI": {"name": "Industrials ETF", "class": "sector_etf"},
    "XLB": {"name": "Materials ETF", "class": "sector_etf"},
    "XLU": {"name": "Utilities ETF", "class": "sector_etf"},
    "XLRE": {"name": "Real Estate ETF", "class": "sector_etf"},
    "XLC": {"name": "Comm. Services ETF", "class": "sector_etf"},
    # US Dollar
    "DX-Y.NYB": {"name": "US Dollar Index", "class": "currency"},
    # Commodity Futures
    "GC=F": {"name": "Gold", "class": "commodity"},
    "CL=F": {"name": "Crude Oil", "class": "commodity"},
    "HG=F": {"name": "Copper", "class": "commodity"},
    "NG=F": {"name": "Natural Gas", "class": "commodity"},
    "SI=F": {"name": "Silver", "class": "commodity"},
    "ZW=F": {"name": "Wheat", "class": "commodity"},
}

# Mapping from ETF symbol → GICS sector name (for cross-referencing)
ETF_TO_SECTOR = {
    "XLE": "Energy",
    "XLF": "Financials",
    "XLK": "Technology",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# yfinance symbol → DB display symbol overrides
_SYMBOL_OVERRIDES = {
    "GC=F": "GOLD",
    "CL=F": "OIL",
    "HG=F": "COPPER",
    "NG=F": "NATGAS",
    "SI=F": "SILVER",
    "ZW=F": "WHEAT",
}


# Friendly display names (strip yfinance suffixes)
def _display_symbol(yf_symbol: str) -> str:
    """Convert yfinance symbol to a display-friendly name."""
    if yf_symbol in _SYMBOL_OVERRIDES:
        return _SYMBOL_OVERRIDES[yf_symbol]
    return yf_symbol.replace("^", "").replace("=F", "_FUT").replace("-Y.NYB", "")


async def collect_market_data(period: str = "6mo") -> dict:
    """
    Batch-download all market regime tickers and store in asset_prices.

    Returns dict with counts per asset class.
    """
    symbols = list(MARKET_TICKERS.keys())
    logger.info(f"[market_regime] Downloading {len(symbols)} tickers, period={period}")

    try:
        # Offload synchronous network IO to background thread
        df = await asyncio.to_thread(
            yf.download, symbols, period=period, auto_adjust=True, threads=True
        )
    except Exception as e:
        logger.error(f"[market_regime] yfinance batch download failed: {e}")
        return {"error": str(e), "total": 0}

    if df.empty:
        logger.warning("[market_regime] Empty dataframe from yfinance")
        return {"total": 0}

    with get_db() as db:
        counts: dict[str, int] = {}
        total = 0
        all_rows = []

        for yf_sym, meta in MARKET_TICKERS.items():
            asset_class = meta["class"]
            display_sym = _display_symbol(yf_sym)

            try:
                # Multi-ticker download returns MultiIndex columns: (field, ticker)
                if len(symbols) > 1:
                    ticker_df = (
                        df.xs(yf_sym, level=1, axis=1)
                        if isinstance(df.columns, type(df.columns))
                        else df
                    )
                    # yfinance multi-download: columns are (Close, sym), (Open, sym), etc.
                    close_series = (
                        df["Close"][yf_sym] if yf_sym in df["Close"].columns else None
                    )
                    open_series = (
                        df["Open"][yf_sym] if yf_sym in df["Open"].columns else None
                    )
                    high_series = (
                        df["High"][yf_sym] if yf_sym in df["High"].columns else None
                    )
                    low_series = (
                        df["Low"][yf_sym] if yf_sym in df["Low"].columns else None
                    )
                    vol_series = (
                        df["Volume"][yf_sym] if yf_sym in df["Volume"].columns else None
                    )
                else:
                    close_series = df["Close"]
                    open_series = df["Open"]
                    high_series = df["High"]
                    low_series = df["Low"]
                    vol_series = df["Volume"]

                if close_series is None or close_series.dropna().empty:
                    logger.debug(f"[market_regime] No data for {yf_sym}")
                    continue

                import math

                symbol_rows = 0
                for date_idx in close_series.dropna().index:
                    d = date_idx.date() if hasattr(date_idx, "date") else date_idx
                    c = float(close_series[date_idx])
                    o = (
                        float(open_series[date_idx])
                        if open_series is not None
                        else None
                    )
                    h = (
                        float(high_series[date_idx])
                        if high_series is not None
                        else None
                    )
                    l_ = float(low_series[date_idx]) if low_series is not None else None
                    v = float(vol_series[date_idx]) if vol_series is not None else 0

                    # Handle NaN
                    if math.isnan(c):
                        continue
                    o = None if o is not None and math.isnan(o) else o
                    h = None if h is not None and math.isnan(h) else h
                    l_ = None if l_ is not None and math.isnan(l_) else l_
                    v = 0 if v is not None and math.isnan(v) else v

                    all_rows.append(
                        (display_sym, asset_class, d, o, h, l_, c, v, "yfinance")
                    )
                    symbol_rows += 1

                counts[display_sym] = symbol_rows
                total += symbol_rows
                if symbol_rows > 0:
                    logger.info(
                        f"[market_regime] {display_sym} ({asset_class}): {symbol_rows} rows"
                    )

            except Exception as e:
                logger.warning(f"[market_regime] Failed for {yf_sym}: {e}")
                continue

        if all_rows:
            for row in all_rows:
                db.execute(
                    """
                    INSERT INTO asset_prices
                    (symbol, asset_class, date, open, high, low, close, volume, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, asset_class, date) DO NOTHING
                """,
                    list(row),
                )

        logger.info(f"[market_regime] Total: {total} rows across {len(counts)} symbols")
        return {"total": total, "per_symbol": counts}


def get_latest_market_snapshot() -> dict:
    """
    Get the latest values for key market instruments from asset_prices.
    Returns a dict with VIX, yields, indexes, dollar.
    """
    with get_db() as db:
        result = {}

        key_symbols = {
            "VIX": "volatility",
            "VIX3M": "volatility",
            "GSPC": "index",
            "IXIC": "index",
            "RUT": "index",
            "DJI": "index",
            "TNX": "yield",
            "FVX": "yield",
            "IRX": "yield",
            "TYX": "yield",
            "DX": "currency",
        }

        for sym, aclass in key_symbols.items():
            try:
                row = db.execute(
                    """
                    SELECT close, date FROM asset_prices
                    WHERE symbol = %s AND asset_class = %s
                    ORDER BY date DESC LIMIT 1
                """,
                    [sym, aclass],
                ).fetchone()
                if row:
                    result[sym] = {"close": row[0], "date": str(row[1])}
            except Exception:
                pass

        # Get all sector ETFs
        etf_syms = list(ETF_TO_SECTOR.keys())
        for sym in etf_syms:
            try:
                row = db.execute(
                    """
                    SELECT close, date FROM asset_prices
                    WHERE symbol = %s AND asset_class = 'sector_etf'
                    ORDER BY date DESC LIMIT 1
                """,
                    [sym],
                ).fetchone()
                if row:
                    result[sym] = {"close": row[0], "date": str(row[1])}
            except Exception:
                pass

        return result


def get_asset_history(symbol: str, asset_class: str, days: int = 90) -> list[dict]:
    """Get price history for a specific asset from asset_prices."""
    with get_db() as db:
        rows = db.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM asset_prices
            WHERE symbol = %s AND asset_class = %s
            ORDER BY date DESC
            LIMIT %s
        """,
            [symbol, asset_class, days],
        ).fetchall()

        return [
            {
                "date": str(r[0]),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": r[5],
            }
            for r in rows
        ]

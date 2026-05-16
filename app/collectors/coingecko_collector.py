"""
CoinGecko Collector — Free crypto price data.

Pure data collector. No LLM calls.
Writes to: asset_prices (asset_class='crypto')
No API key needed (public endpoints, 10-30 calls/min).

Fallback: yfinance BTC-USD etc. if CoinGecko 429s.
"""

import logging

logger = logging.getLogger(__name__)


import asyncio
import datetime
import httpx
import yfinance as yf
from app.db.connection import get_db

BASE_URL = "https://api.coingecko.com/api/v3"

# Top coins to track — (coingecko_id, symbol, yfinance_ticker)
TRACKED_COINS = [
    ("bitcoin", "BTC", "BTC-USD"),
    ("ethereum", "ETH", "ETH-USD"),
    ("solana", "SOL", "SOL-USD"),
    ("binancecoin", "BNB", "BNB-USD"),
    ("ripple", "XRP", "XRP-USD"),
]


async def collect_crypto_prices(days: int = 30) -> int:
    """Fetch historical daily prices for tracked crypto.

    Primary: CoinGecko (free, no key).
    Fallback: yfinance (BTC-USD etc.) if CoinGecko rate-limits.
    Returns total rows inserted.
    """
    with get_db() as db:
        total = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            for coin_id, symbol, yf_ticker in TRACKED_COINS:
                try:
                    await asyncio.sleep(2)  # Rate limit: 10-30 calls/min
                    r = await client.get(
                        f"{BASE_URL}/coins/{coin_id}/market_chart",
                        params={
                            "vs_currency": "usd",
                            "days": days,
                            "interval": "daily",
                        },
                    )
                    r.raise_for_status()
                    data = r.json()

                    prices = data.get("prices", [])
                    volumes = data.get("total_volumes", [])

                    count = 0
                    for i, (ts, price) in enumerate(prices):
                        date = datetime.datetime.fromtimestamp(
                            ts / 1000, tz=datetime.UTC
                        ).date()
                        vol = volumes[i][1] if i < len(volumes) else 0

                        db.execute(
                            """
                            INSERT INTO asset_prices
                            (symbol, asset_class, date, open, high, low, close,
                             volume, currency, source)
                            VALUES (%s, 'crypto', %s, %s, %s, %s, %s, %s, 'USD',
                                    'coingecko')
                ON CONFLICT (symbol, asset_class, date) DO NOTHING
                        """,
                            [
                                symbol,
                                date,
                                price,
                                price,
                                price,
                                price,
                                vol,
                            ],
                        )
                        count += 1

                    logger.info(f"[coingecko] {symbol}: {count} price points written")
                    total += count

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        logger.info(
                            f"[coingecko] {symbol}: 429 rate limited, "
                            f"falling back to yfinance"
                        )
                        count = _yfinance_fallback(symbol, yf_ticker, days, db)
                        total += count
                    else:
                        logger.info(
                            f"[coingecko] {symbol}: HTTP {e.response.status_code}"
                        )
                except Exception as e:
                    logger.info(
                        f"[coingecko] {symbol}: error — {e}, trying yfinance fallback"
                    )
                    count = _yfinance_fallback(symbol, yf_ticker, days, db)
                    total += count

        return total


def _yfinance_fallback(symbol: str, yf_ticker: str, days: int, db) -> int:
    """Fallback: fetch crypto prices via yfinance."""
    try:
        stock = yf.Ticker(yf_ticker)
        data = stock.history(period=f"{days}d", auto_adjust=True)
        if data.empty:
            logger.info(f"[yfinance-crypto] {symbol}: no data")
            return 0

        count = 0
        for date, row in data.iterrows():
            db.execute(
                """
                INSERT INTO asset_prices
                (symbol, asset_class, date, open, high, low, close,
                 volume, currency, source)
                VALUES (%s, 'crypto', %s, %s, %s, %s, %s, %s, 'USD', 'yfinance')
            ON CONFLICT (symbol, asset_class, date) DO NOTHING
            """,
                [
                    symbol,
                    date.date(),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    float(row.get("Volume", 0)),
                ],
            )
            count += 1

        logger.info(f"[yfinance-crypto] {symbol}: {count} price points (fallback)")
        return count
    except Exception as e:
        logger.info(f"[yfinance-crypto] {symbol}: fallback failed — {e}")
        return 0


async def collect_current_prices() -> dict:
    """Fetch current prices for all tracked coins (single API call)."""
    ids = ",".join(c[0] for c in TRACKED_COINS)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{BASE_URL}/simple/price",
            params={
                "ids": ids,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            },
        )
        r.raise_for_status()
        data = r.json()

    result = {}
    for coin_id, symbol, _ in TRACKED_COINS:
        if coin_id in data:
            info = data[coin_id]
            result[symbol] = {
                "price": info.get("usd"),
                "market_cap": info.get("usd_market_cap"),
                "volume_24h": info.get("usd_24h_vol"),
                "change_24h": info.get("usd_24h_change"),
            }
            logger.info(f"[coingecko] {symbol}: ${info.get('usd'):,.2f}")

    return result


async def collect_all() -> dict:
    """Run all CoinGecko collectors."""
    rows = await collect_crypto_prices(days=7)
    await asyncio.sleep(3)  # Rate limit before next call
    try:
        current = await collect_current_prices()
    except httpx.HTTPStatusError:
        logger.info("[coingecko] current prices rate-limited, skipping")
        current = {}

    return {
        "historical_rows": rows,
        "current_prices": current,
    }

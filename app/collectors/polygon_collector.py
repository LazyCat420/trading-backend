"""
Polygon.io Collector — Fallback for OHLCV prices.

Pure data collector. No LLM calls.
Writes to: price_history
Requires: POLYGON_API_KEY in .env (free tier = 5 calls/min)
"""

import logging
import datetime
from app.config import settings
from app.db.connection import get_db
from app.services.request_utils import SmartClient

logger = logging.getLogger(__name__)


def _get_key() -> str:
    key = settings.POLYGON_API_KEY
    if not key:
        raise ValueError("POLYGON_API_KEY not set in .env")
    return key


async def collect_price_history(ticker: str, days_back: int = 365) -> int:
    """Fetch OHLCV history and upsert into price_history table."""
    try:
        api_key = _get_key()
    except ValueError:
        return 0

    to_date = datetime.date.today()
    from_date = to_date - datetime.timedelta(days=days_back)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"

    async with SmartClient(
        base_delay=12.0, max_retries=3
    ) as client:  # 5 calls/min = 12s delay ideally
        resp = await client.get(
            url,
            params={
                "apiKey": api_key,
                "adjusted": "true",
                "sort": "desc",
                "limit": 50000,
            },
        )
        if resp.status_code != 200:
            logger.info(
                f"[polygon] Error fetching price history for {ticker}: HTTP {resp.status_code}"
            )
            return 0
        data = resp.json()

    results = data.get("results", [])
    if not results:
        logger.info(f"[polygon] No price data for {ticker}")
        return 0

    with get_db() as db:
        count = 0
        for day in results:
            try:
                # Polygon timestamps are in milliseconds
                date_obj = datetime.datetime.fromtimestamp(
                    day["t"] / 1000.0, tz=datetime.UTC
                ).date()

                db.execute(
                    """
                    INSERT INTO price_history (ticker, date, open, high, low, close, volume, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'polygon')
                    ON CONFLICT (ticker, date, source) DO NOTHING
                    """,
                    [
                        ticker,
                        date_obj,
                        float(day.get("o", 0)),
                        float(day.get("h", 0)),
                        float(day.get("l", 0)),
                        float(day.get("c", 0)),
                        int(day.get("v", 0)),
                    ],
                )
                count += 1
            except Exception as e:
                continue

        logger.info(f"[polygon] {ticker}: {count} price rows written")
        return count


async def collect_all(ticker: str) -> dict:
    """Run all Polygon collectors."""
    prices = await collect_price_history(ticker)
    return {"ticker": ticker, "price_rows": prices, "source": "polygon"}

"""
FRED Collector — Federal Reserve Economic Data.

Pure data collector. No LLM calls.
Writes to: macro_indicators
Requires: FRED_API_KEY in .env (free at fred.stlouisfed.org)
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# FRED gets max 3 concurrent DB connections to avoid starving the pool
# when pipeline agents are also running.
_FRED_DB_SEMAPHORE = asyncio.Semaphore(3)


import datetime
from fredapi import Fred
from app.config import settings
from app.db.connection import get_db

# Key FRED series IDs
SERIES = {
    "CPI": "CPIAUCSL",  # Consumer Price Index
    "GDP": "GDP",  # Gross Domestic Product
    "UNEMPLOYMENT": "UNRATE",  # Unemployment Rate
    "FED_FUNDS": "FEDFUNDS",  # Federal Funds Rate
    "TREASURY_10Y": "DGS10",  # 10-Year Treasury Yield
    "TREASURY_2Y": "DGS2",  # 2-Year Treasury Yield
    "TREASURY_3MO": "DGS3MO",  # 3-Month Treasury Yield
    "INFLATION_EXPECT": "T5YIE",  # 5-Year Breakeven Inflation
    "REAL_GDP_GROWTH": "A191RL1Q225SBEA",  # Real GDP Growth Rate
    "INITIAL_CLAIMS": "ICSA",  # Initial Jobless Claims
}


def _get_client() -> Fred:
    key = settings.FRED_API_KEY
    if not key:
        raise ValueError(
            "FRED_API_KEY not set in .env — get free key at fred.stlouisfed.org"
        )
    return Fred(api_key=key)


async def collect_macro_indicator(
    indicator_name: str,
    series_id: str,
    lookback_years: int = 2,
) -> int:
    """
    Fetch a single FRED series and upsert into macro_indicators.
    Returns number of rows inserted.
    """
    client = _get_client()
    start = datetime.date.today() - datetime.timedelta(days=lookback_years * 365)

    try:
        import asyncio

        data = await asyncio.to_thread(
            client.get_series, series_id, observation_start=start
        )
    except Exception as e:
        logger.info(f"[fred] {indicator_name} ({series_id}): error — {e}")
        return 0

    if data is None or data.empty:
        logger.info(f"[fred] {indicator_name}: no data")
        return 0

    rows = []
    for date, value in data.items():
        if str(value) == "nan" or value is None:
            continue
        rows.append((indicator_name, date.date(), float(value), "US", "fred"))

    def _insert_rows(rows_to_insert):
        with get_db() as db:
            db.executemany(
                """
                INSERT INTO macro_indicators
                (indicator, date, value, country, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (indicator, date, country) DO NOTHING
            """,
                rows_to_insert,
            )

    if rows:
        # Gate DB writes behind semaphore to avoid pool starvation
        async with _FRED_DB_SEMAPHORE:
            await asyncio.to_thread(_insert_rows, rows)

    count = len(rows)

    logger.info(f"[fred] {indicator_name}: {count} data points written")
    return count


async def collect_yield_curve() -> dict:
    """Fetch Treasury yields for yield curve construction."""
    results = {}
    for name in ["TREASURY_3MO", "TREASURY_2Y", "TREASURY_10Y"]:
        count = await collect_macro_indicator(name, SERIES[name], lookback_years=1)
        results[name] = count
    return results


async def collect_all() -> dict:
    """Fetch all tracked FRED series concurrently."""
    import asyncio

    results = {}

    async def fetch_and_record(name, series_id):
        count = await collect_macro_indicator(name, series_id, lookback_years=30)
        return name, count

    tasks = [fetch_and_record(name, series_id) for name, series_id in SERIES.items()]
    completed = await asyncio.gather(*tasks)

    for name, count in completed:
        results[name] = count

    return results

def sync_collect_fred(is_shutting_down) -> int:
    """Fetch all FRED series (runs in a thread).

    Uses batch executemany instead of row-by-row inserts to
    reduce CPU load by ~10x. Sleeps between series to yield
    CPU so the event loop can serve API requests.
    """
    import datetime
    import time
    from fredapi import Fred
    from app.config import settings

    key = settings.FRED_API_KEY
    if not key:
        logger.warning("[startup] FRED_API_KEY not set — skipping FRED refresh")
        return 0

    client = Fred(api_key=key)
    start = datetime.date.today() - datetime.timedelta(days=30 * 365)
    total = 0

    for name, series_id in SERIES.items():
        if is_shutting_down():
            logger.info("[startup] FRED refresh aborted due to shutdown.")
            break
        try:
            data = client.get_series(series_id, observation_start=start)
            if data is None or data.empty:
                continue
            # Batch collect rows then executemany (much faster than row-by-row)
            rows = []
            for date, value in data.items():
                if str(value) == "nan" or value is None:
                    continue
                rows.append((name, date.date(), float(value), "US", "fred"))
            if rows:
                with get_db() as db:
                    db.executemany(
                        "INSERT INTO macro_indicators "
                        "(indicator, date, value, country, source) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (indicator, date, country) DO NOTHING",
                        rows,
                    )
                total += len(rows)
            logger.info("[startup] FRED %s: %d rows", name, len(rows))
        except Exception as e:
            logger.warning("[startup] FRED %s failed: %s", name, e)
        # Yield CPU between series so API requests aren't starved
        time.sleep(0.1)

    return total

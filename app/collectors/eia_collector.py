"""
EIA Collector — U.S. Energy Information Administration.

Pure data collector. No LLM calls.
Writes to: global.energy_reports
Requires: EIA_API_KEY in .env (free at eia.gov/opendata/register.php)

Key data collected:
  - Weekly US crude oil inventories (the #1 market-moving energy report)
  - Weekly gasoline/distillate inventories
  - Weekly crude production
  - Weekly crude imports
"""

import hashlib
import logging
from datetime import datetime, timezone

import httpx

from app.config import settings
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── EIA API v2 Base ──────────────────────────────────────────────
BASE_URL = "https://api.eia.gov/v2"

# Key petroleum series to track (Weekly Petroleum Status Report)
# These are the series that move crude oil prices every Wednesday at 10:30 AM ET
EIA_SERIES = {
    "crude_inventory": {
        "route": "/petroleum/stoc/wstk/data/",
        "params": {
            "frequency": "weekly",
            "data[]": "value",
            "facets[product][]": "EPC0",
            "facets[process][]": "SAE",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 104,
        },  # ~2 years weekly
        "unit": "thousand_barrels",
    },
    "gasoline_inventory": {
        "route": "/petroleum/stoc/wstk/data/",
        "params": {
            "frequency": "weekly",
            "data[]": "value",
            "facets[product][]": "EPM0",
            "facets[process][]": "SAE",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 104,
        },
        "unit": "thousand_barrels",
    },
    "distillate_inventory": {
        "route": "/petroleum/stoc/wstk/data/",
        "params": {
            "frequency": "weekly",
            "data[]": "value",
            "facets[product][]": "EPD0",
            "facets[process][]": "SAE",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 104,
        },
        "unit": "thousand_barrels",
    },
    "crude_production": {
        "route": "/petroleum/sum/sndw/data/",
        "params": {
            "frequency": "weekly",
            "data[]": "value",
            "facets[process][]": "FPF",
            "facets[product][]": "EPC0",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 104,
        },
        "unit": "thousand_bpd",
    },
    "crude_imports": {
        "route": "/petroleum/move/wkly/data/",
        "params": {
            "frequency": "weekly",
            "data[]": "value",
            "facets[product][]": "EPC0",
            "facets[process][]": "NET",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 104,
        },
        "unit": "thousand_bpd",
    },
}


def _make_id(indicator: str, date: str) -> str:
    """Deterministic ID for dedup."""
    return hashlib.md5(f"eia:{indicator}:{date}".encode()).hexdigest()


async def collect_eia_series(
    indicator: str,
    config: dict,
) -> int:
    """Fetch one EIA series and upsert into global.energy_reports."""
    key = settings.EIA_API_KEY
    if not key:
        logger.warning("[eia] EIA_API_KEY not set — skipping %s", indicator)
        return 0

    url = BASE_URL + config["route"]
    params = {**config["params"], "api_key": key}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error("[eia] %s: HTTP error — %s", indicator, e)
        return 0

    # EIA v2 returns data in response.data
    rows = data.get("response", {}).get("data", [])
    if not rows:
        logger.warning("[eia] %s: no data returned", indicator)
        return 0

    with get_db() as db:
        count = 0
        for row in rows:
            period = row.get("period")
            value = row.get("value")
            if not period or value is None:
                continue
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            row_id = _make_id(indicator, period)
            series_id = row.get("series", indicator)
            now = datetime.now(timezone.utc)

            db.execute(
                """
                INSERT INTO global.energy_reports
                (id, series_id, indicator, date, value, unit, source, collected_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'eia', %s)
                ON CONFLICT (id) DO UPDATE SET
                    value = EXCLUDED.value,
                    collected_at = EXCLUDED.collected_at
            """,
                [row_id, series_id, indicator, period, value, config["unit"], now],
            )
            count += 1

        logger.info("[eia] %s: %d data points written", indicator, count)
        logger.info(f"  [eia] {indicator}: {count} data points written")
        return count


async def collect_all() -> dict:
    """Fetch all tracked EIA series. Returns {indicator: count}."""
    if not settings.EIA_API_KEY:
        logger.info("  [eia] EIA_API_KEY not set — skipping all EIA collection")
        return {"error": "EIA_API_KEY not set"}

    results = {}
    total = 0
    for indicator, config in EIA_SERIES.items():
        count = await collect_eia_series(indicator, config)
        results[indicator] = count
        total += count

    logger.info(f"  [eia] Total: {total} data points across {len(EIA_SERIES)} series")
    return results

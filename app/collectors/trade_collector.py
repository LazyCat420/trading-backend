"""
Trade Flow Collector — Multi-source global trade data.

Uses 3 free sources in rotation to avoid rate limits:
  1. World Bank WITS API (primary — same data as COMTRADE, no key needed)
  2. U.S. Census Bureau International Trade API (US-specific, no key needed)
  3. Open Trade Statistics / tradestatistics.io (no key needed)

Writes to: global.trade_flows
No API keys required.

Key data collected:
  - Commodity-level import/export flows between major economies
  - Tracked commodities defined in global.tracked_commodities table
  - Monthly frequency (trade data is inherently monthly/quarterly)
"""

import hashlib
import logging
from datetime import datetime, timezone

import httpx

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── API Endpoints ────────────────────────────────────────────────
WITS_BASE = (
    "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/tradestats-trade/reporter"
)
OTS_BASE = "https://api.tradestatistics.io"
CENSUS_BASE = "https://api.census.gov/data/timeseries/intltrade"

# Key trade pairs to monitor (reporter → partner for specific commodities)
# These are the flows that most impact global markets
TRADE_PAIRS = [
    # Energy flows
    {
        "reporter": "USA",
        "partner": "SAU",
        "hs_codes": ["2709"],
        "desc": "US-Saudi crude oil",
    },
    {
        "reporter": "USA",
        "partner": "CAN",
        "hs_codes": ["2709"],
        "desc": "US-Canada crude oil",
    },
    {
        "reporter": "CHN",
        "partner": "SAU",
        "hs_codes": ["2709"],
        "desc": "China-Saudi crude oil",
    },
    {
        "reporter": "USA",
        "partner": "WLD",
        "hs_codes": ["2711"],
        "desc": "US LNG exports (world)",
    },
    # Tech/semiconductor flows
    {
        "reporter": "USA",
        "partner": "CHN",
        "hs_codes": ["8542"],
        "desc": "US-China semiconductors",
    },
    {
        "reporter": "USA",
        "partner": "TWN",
        "hs_codes": ["8542"],
        "desc": "US-Taiwan semiconductors",
    },
    {
        "reporter": "CHN",
        "partner": "TWN",
        "hs_codes": ["8542"],
        "desc": "China-Taiwan semiconductors",
    },
    # Agricultural flows
    {
        "reporter": "USA",
        "partner": "CHN",
        "hs_codes": ["1201"],
        "desc": "US-China soybeans",
    },
    {
        "reporter": "USA",
        "partner": "CHN",
        "hs_codes": ["1001"],
        "desc": "US-China wheat",
    },
    # Metals
    {
        "reporter": "CHN",
        "partner": "WLD",
        "hs_codes": ["7403"],
        "desc": "China copper imports (world)",
    },
    {
        "reporter": "CHN",
        "partner": "AUS",
        "hs_codes": ["2601"],
        "desc": "China-Australia iron ore",
    },
    # Precious metals
    {
        "reporter": "CHN",
        "partner": "WLD",
        "hs_codes": ["7108"],
        "desc": "China gold imports (world)",
    },
]

# ISO3 country codes for Census API (uses different format)
CENSUS_COUNTRY_MAP = {
    "CHN": "5700",
    "SAU": "5170",
    "CAN": "1220",
    "TWN": "5830",
    "AUS": "6021",
    "WLD": "-",
}


def _make_id(
    reporter: str, partner: str, hs: str, period: str, flow: str, source: str
) -> str:
    """Deterministic ID for dedup."""
    raw = f"{source}:{reporter}:{partner}:{hs}:{period}:{flow}"
    return hashlib.md5(raw.encode()).hexdigest()


async def _fetch_wits(
    reporter: str,
    partner: str,
    hs_code: str,
    year: int,
) -> list[dict]:
    """Fetch trade data from World Bank WITS API."""
    # WITS REST URL format:
    # /reporter/{iso3}/year/{year}/partner/{iso3}/product/{hs2}/indicator/MPRT-TRD-VL
    results = []
    for indicator, flow in [("MPRT-TRD-VL", "Import"), ("XPRT-TRD-VL", "Export")]:
        url = (
            f"https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/"
            f"DF_WITS_TradeStats_Trade/.{reporter}.{partner}.{hs_code}.{indicator}"
            f"?startPeriod={year}&endPeriod={year}&format=json"
        )
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, headers={"Accept": "application/json"})
                if resp.status_code == 200:
                    data = resp.json()
                    # Parse SDMX JSON response
                    observations = data.get("dataSets", [{}])[0].get("series", {})
                    for series_key, series_data in observations.items():
                        for obs_key, obs_vals in series_data.get(
                            "observations", {}
                        ).items():
                            value = obs_vals[0] if obs_vals else None
                            if value is not None:
                                results.append(
                                    {
                                        "reporter": reporter,
                                        "partner": partner,
                                        "commodity_code": hs_code,
                                        "trade_flow": flow,
                                        "value_usd": float(value),
                                        "period": str(year),
                                        "source": "wits",
                                    }
                                )
                elif resp.status_code == 404:
                    pass  # No data for this combo — normal
                else:
                    logger.debug(
                        "[wits] %d for %s-%s HS%s",
                        resp.status_code,
                        reporter,
                        partner,
                        hs_code,
                    )
        except Exception as e:
            logger.debug("[wits] Error fetching %s-%s: %s", reporter, partner, e)

    return results


async def _fetch_ots(
    reporter: str,
    partner: str,
    hs_code: str,
    year: int,
) -> list[dict]:
    """Fetch from Open Trade Statistics API (tradestatistics.io)."""
    results = []
    # OTS uses HS 2-digit chapter codes
    hs2 = hs_code[:2]
    url = f"{OTS_BASE}/yrpc?y={year}&r={reporter.lower()}&p={partner.lower()}&c={hs2}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for row in data:
                        export_val = row.get("export_value_usd", 0)
                        import_val = row.get("import_value_usd", 0)
                        if export_val:
                            results.append(
                                {
                                    "reporter": reporter,
                                    "partner": partner,
                                    "commodity_code": hs_code,
                                    "trade_flow": "Export",
                                    "value_usd": float(export_val),
                                    "period": str(year),
                                    "source": "ots",
                                }
                            )
                        if import_val:
                            results.append(
                                {
                                    "reporter": reporter,
                                    "partner": partner,
                                    "commodity_code": hs_code,
                                    "trade_flow": "Import",
                                    "value_usd": float(import_val),
                                    "period": str(year),
                                    "source": "ots",
                                }
                            )
    except Exception as e:
        logger.debug("[ots] Error: %s", e)

    return results


async def _fetch_census(
    partner_code: str,
    hs_code: str,
    year: int,
) -> list[dict]:
    """Fetch US trade data from Census Bureau API (US reporter only, no key needed)."""
    results = []
    # Census Bureau Imports endpoint
    for endpoint, flow in [("imports/hs", "Import"), ("exports/hs", "Export")]:
        url = f"{CENSUS_BASE}/{endpoint}"
        params = {
            "get": "GEN_VAL_MO,I_COMMODITY",
            "YEAR": str(year),
            "I_COMMODITY": hs_code,
        }
        if partner_code != "-":
            params["CTY_CODE"] = partner_code

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    # Census returns header row + data rows
                    if len(data) > 1:
                        total_value = sum(
                            float(row[0])
                            for row in data[1:]
                            if row[0] and row[0] != "null"
                        )
                        if total_value > 0:
                            results.append(
                                {
                                    "reporter": "USA",
                                    "partner": partner_code,
                                    "commodity_code": hs_code,
                                    "trade_flow": flow,
                                    "value_usd": total_value,
                                    "period": str(year),
                                    "source": "census",
                                }
                            )
        except Exception as e:
            logger.debug("[census] Error: %s", e)

    return results


async def collect_trade_pair(pair: dict, year: int) -> int:
    """
    Fetch trade data for one pair using sources in rotation.
    Tries WITS first, falls back to OTS, then Census (US only).
    """
    reporter = pair["reporter"]
    partner = pair["partner"]
    hs_codes = pair["hs_codes"]

    all_results = []

    for hs_code in hs_codes:
        # Source 1: Try WITS (World Bank)
        results = await _fetch_wits(reporter, partner, hs_code, year)
        if results:
            all_results.extend(results)
            continue

        # Source 2: Try OTS
        results = await _fetch_ots(reporter, partner, hs_code, year)
        if results:
            all_results.extend(results)
            continue

        # Source 3: Census Bureau (US reporter only)
        if reporter == "USA" and partner in CENSUS_COUNTRY_MAP:
            census_partner = CENSUS_COUNTRY_MAP[partner]
            results = await _fetch_census(census_partner, hs_code, year)
            if results:
                all_results.extend(results)

    # Write results to DB
    with get_db() as db:
        count = 0
        for r in all_results:
            row_id = _make_id(
                r["reporter"],
                r["partner"],
                r["commodity_code"],
                r["period"],
                r["trade_flow"],
                r["source"],
            )

            # Look up commodity description from tracked_commodities
            desc_row = db.execute(
                "SELECT name FROM global.tracked_commodities WHERE hs_code = %s",
                [r["commodity_code"]],
            ).fetchone()
            commodity_desc = desc_row[0] if desc_row else r["commodity_code"]
            now = datetime.now(timezone.utc)
            db.execute(
                """
                INSERT INTO global.trade_flows
                (id, reporter_code, reporter, partner_code, partner,
                 commodity_code, commodity_desc, trade_flow,
                 value_usd, net_weight_kg, period, source, collected_at)
                VALUES (%s, NULL, %s, NULL, %s, %s, %s, %s, %s, NULL, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    value_usd = EXCLUDED.value_usd,
                    collected_at = EXCLUDED.collected_at
            """,
                [
                    row_id,
                    r["reporter"],
                    r["partner"],
                    r["commodity_code"],
                    commodity_desc,
                    r["trade_flow"],
                    r["value_usd"],
                    r["period"],
                    r["source"],
                    now,
                ],
            )
            count += 1

        if count:
            logger.info(
                f"    {pair['desc']}: {count} records ({all_results[0]['source']})"
            )

        return count


async def collect_all() -> dict:
    """
    Fetch all tracked trade pairs for the last 2 years.
    Uses 3 sources in rotation to avoid rate limits.
    """
    current_year = datetime.now(timezone.utc).year
    years = [current_year - 1, current_year]  # Last year + current year

    total = 0
    results = {}
    errors = 0

    logger.info(f"  [trade] Collecting {len(TRADE_PAIRS)} trade pairs for {years}...")

    for pair in TRADE_PAIRS:
        pair_total = 0
        for year in years:
            try:
                count = await collect_trade_pair(pair, year)
                pair_total += count
            except Exception as e:
                logger.error("[trade] Error for %s year %d: %s", pair["desc"], year, e)
                errors += 1

        results[pair["desc"]] = pair_total
        total += pair_total

    logger.info(f"  [trade] Total: {total} trade flow records ({errors} errors)")
    return results

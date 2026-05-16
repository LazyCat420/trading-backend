"""
GDELT Collector — War/Oil geo-tagged news from GDELT Event CSV exports.

Pure data collector. No LLM calls.
Writes to: global.war_news_feed
No API key required — uses GDELT's free raw CSV exports.

Downloads the latest GDELT Event CSV (updated every 15 minutes),
extracts events related to oil/energy/maritime intersected with
conflict contexts, and writes them as geo-tagged news items.

Data source: http://data.gdeltproject.org/gdeltv2/lastupdate.txt
"""

import hashlib
import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── GDELT Raw Data URLs ────────────────────────────────────────
LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# Energy/maritime/critical infrastructure CAMEO event root codes
# We want broader event types for news (not just conflict)
# 01-04 = Verbal cooperation/conflict, 05-09 = Material cooperation
# 10-20 = All event types including peace and conflict
# For war news, we focus on QuadClass 3 (Verbal Conflict) and 4 (Material Conflict)
QUAD_CONFLICT = {"3", "4"}

# Oil/energy/maritime countries of interest (GDELT country codes)
ENERGY_COUNTRIES = {
    "IR",
    "IZ",
    "SA",
    "KU",
    "QA",
    "AE",
    "BA",
    "MU",  # Middle East oil
    "YM",
    "LE",
    "SY",  # Conflict zones near chokepoints
    "RS",
    "UP",  # Russia/Ukraine
    "VE",
    "NG",
    "LY",  # Major oil producers
}

# Column positions (same as acled_collector.py)
COL_GLOBAL_EVENT_ID = 0
COL_DAY = 1
COL_ACTOR1_NAME = 6
COL_ACTOR2_NAME = 16
COL_EVENT_ROOT_CODE = 28
COL_QUAD_CLASS = 29
COL_GOLDSTEIN_SCALE = 30
COL_NUM_MENTIONS = 31
COL_AVG_TONE = 34
COL_ACTION_GEO_FULLNAME = 52
COL_ACTION_GEO_COUNTRY = 53
COL_ACTION_GEO_LAT = 56
COL_ACTION_GEO_LONG = 57
COL_SOURCE_URL = 60

# Minimum thresholds for market-relevant news
MIN_MENTIONS = 3  # Filter low-signal single-mention events
MAX_GOLDSTEIN = -4.0  # Only keep events with Goldstein <= -4 (material conflict)


def _make_id(event_id: str) -> str:
    """Deterministic ID from GDELT event ID."""
    return hashlib.md5(f"gdelt_news:{event_id}".encode()).hexdigest()


def _safe_float(val: str) -> float | None:
    """Safely convert string to float."""
    try:
        return float(val) if val.strip() else None
    except (ValueError, TypeError):
        return None


async def collect_gdelt_news() -> int:
    """
    Download the latest GDELT Event CSV export, filter for conflict events
    from energy/strategic countries, and write as war news.
    Returns total number of rows written.
    """
    with get_db() as db:
        total_count = 0

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                # Step 1: Get the lastupdate.txt to find the export URL
                r1 = await client.get(LASTUPDATE_URL)
                r1.raise_for_status()

                export_url = None
                for line in r1.text.strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 3 and ".export." in parts[2].lower():
                        export_url = parts[2]
                        break

                if not export_url:
                    logger.warning("[gdelt] No export file found in lastupdate.txt")
                    logger.info("  [gdelt] No export file found")
                    return 0

                # Step 2: Download the ZIP
                r2 = await client.get(export_url, timeout=30)
                r2.raise_for_status()

                # Step 3: Extract and parse the CSV
                zf = zipfile.ZipFile(io.BytesIO(r2.content))
                csv_name = zf.namelist()[0]

                with zf.open(csv_name) as f:
                    text = f.read().decode("utf-8", errors="replace")

                rows = [
                    line.split("\t")
                    for line in text.strip().split("\n")
                    if line.strip()
                ]
                logger.info(
                    "[gdelt] Downloaded %d events from %s",
                    len(rows),
                    export_url.split("/")[-1],
                )

            except Exception as e:
                logger.error("[gdelt] Failed to download GDELT export: %s", e)
                logger.info(f"  [gdelt] Failed to download GDELT export: {e}")
                return 0

        if not rows:
            logger.info("  [gdelt] No events in latest GDELT export")
            return 0

        now = datetime.now(timezone.utc)

        for row in rows:
            if len(row) < 61:
                continue

            # Filter: only conflict quadrants (verbal + material conflict)
            quad_class = row[COL_QUAD_CLASS].strip()
            if quad_class not in QUAD_CONFLICT:
                continue

            # Must involve energy/strategic country
            country = row[COL_ACTION_GEO_COUNTRY].strip()
            if country not in ENERGY_COUNTRIES:
                continue

            # Filter: Goldstein severity — only genuinely alarming events
            goldstein = _safe_float(row[COL_GOLDSTEIN_SCALE])
            if goldstein is not None and goldstein > MAX_GOLDSTEIN:
                continue

            # Filter: minimum media mentions — skip low-signal noise
            num_mentions = (
                int(row[COL_NUM_MENTIONS])
                if row[COL_NUM_MENTIONS].strip().isdigit()
                else 0
            )
            if num_mentions < MIN_MENTIONS:
                continue

            # Must have valid coordinates
            lat = _safe_float(row[COL_ACTION_GEO_LAT])
            lon = _safe_float(row[COL_ACTION_GEO_LONG])
            if lat is None or lon is None:
                continue
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                continue

            event_id = row[COL_GLOBAL_EVENT_ID].strip()
            geo_fullname = row[COL_ACTION_GEO_FULLNAME].strip()[:200]
            actor1 = row[COL_ACTOR1_NAME].strip()[:100]
            actor2 = row[COL_ACTOR2_NAME].strip()[:100]
            tone = _safe_float(row[COL_AVG_TONE])
            source_url = (
                row[COL_SOURCE_URL].strip() if len(row) > COL_SOURCE_URL else ""
            )

            # Build headline: location-first, severity-based description
            # Avoid raw GDELT actor names (e.g. "NATIONAL ECONOMIC COUNCIL") — they're misleading
            severity_label = "⚪ Low"
            if goldstein is not None:
                if goldstein <= -8:
                    severity_label = "🔴 Critical"
                elif goldstein <= -5:
                    severity_label = "🟠 High"
                elif goldstein <= -3:
                    severity_label = "🟡 Medium"

            # Use geo + actors for context
            location_part = geo_fullname or f"Unknown location ({country})"
            actors_part = ""
            if actor1 and actor2:
                actors_part = f"{actor1} vs {actor2}"
            elif actor1:
                actors_part = actor1
            elif actor2:
                actors_part = actor2

            if actors_part:
                headline = f"{location_part} — {actors_part}"
            else:
                headline = location_part

            headline = headline[:500]

            # Build themes from CAMEO metadata
            root_code = row[COL_EVENT_ROOT_CODE].strip()
            themes = [f"CAMEO_{root_code}", f"QUAD_{quad_class}", f"GEO_{country}"]
            themes_json = json.dumps(themes)

            row_id = _make_id(event_id)

            # Extract domain from source URL
            source_domain = ""
            if source_url:
                try:
                    source_domain = urlparse(source_url).netloc
                except Exception:
                    pass

            db.execute(
                """
                INSERT INTO global.war_news_feed
                (id, headline, url, source_domain, latitude, longitude,
                 location_name, tone, themes, timestamp, data_source, collected_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'gdelt', %s)
                ON CONFLICT (id) DO UPDATE SET
                    tone = EXCLUDED.tone,
                    collected_at = EXCLUDED.collected_at
            """,
                [
                    row_id,
                    headline,
                    source_url,
                    source_domain,
                    lat,
                    lon,
                    geo_fullname,
                    tone,
                    themes_json,
                    now,
                    now,
                ],
            )
            total_count += 1

        logger.info(
            "[gdelt] %d war news articles written from GDELT export", total_count
        )
        logger.info(f"  [gdelt] {total_count} war news articles written")
        return total_count


async def collect_all() -> dict:
    """Main entry point. Extracts war/oil news from latest GDELT event export."""
    count = await collect_gdelt_news()
    return {"war_news": count}

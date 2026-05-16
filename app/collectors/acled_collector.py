"""
Conflict Event Collector — GDELT Event CSV export-based conflict extraction.

Pure data collector. No LLM calls.
Writes to: global.conflict_events
No API key needed — uses GDELT's free raw CSV exports.

Instead of the rate-limited GDELT API, downloads the raw GDELT Event
CSV files (updated every 15 minutes) and filters for conflict events
using CAMEO event codes.

CAMEO conflict root codes:
  14 = Protest
  15 = Exhibit force posture
  17 = Coerce
  18 = Assault
  19 = Fight (armed conflict)
  20 = Use unconventional mass violence

Data source: http://data.gdeltproject.org/gdeltv2/lastupdate.txt
Docs: https://www.gdeltproject.org/data/documentation/GDELT-Event_Codebook-V2.0.pdf
"""

import hashlib
import io
import logging
import zipfile
from datetime import datetime, timezone

import httpx

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── GDELT Raw Data URLs ────────────────────────────────────────
LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# CAMEO root codes — ONLY genuine armed conflict & military posture
# Removed 14 (Protest) and 17 (Coerce) — too noisy for trading intel
CONFLICT_ROOTS = {"15", "18", "19", "20"}

# CAMEO root code → ACLED-compatible event type mapping
CAMEO_TYPE_MAP = {
    "15": ("Strategic developments", "Exhibit force posture"),
    "18": ("Battles", "Assault"),
    "19": ("Battles", "Fight/Armed clash"),
    "20": ("Violence against civilians", "Unconventional mass violence"),
}

# ── Strategic country filter — only geopolitically relevant nations ──
# Oil producers, conflict zones near chokepoints, major military powers
STRATEGIC_COUNTRIES = {
    # Middle East / oil
    "IR",
    "IZ",
    "SA",
    "KU",
    "QA",
    "AE",
    "BA",
    "MU",
    "YM",
    "LE",
    "SY",
    "IS",
    "JO",
    "TU",
    # Russia / Ukraine / Eastern Europe
    "RS",
    "UP",
    "BO",
    "MD",
    # Africa (oil + conflict)
    "LY",
    "NG",
    "SU",
    "SO",
    "ET",
    "ER",
    "DJ",
    # Asia (strategic)
    "CH",
    "TW",
    "KN",
    "KS",
    "PK",
    "AF",
    # South-East Asia (Malacca)
    "BM",
    "RP",
    # South America (oil)
    "VE",
}

# Minimum Goldstein severity — only keep events scoring <= this
# Goldstein scale: -10 (extreme conflict) to +10 (extreme cooperation)
# -5.0 filters out minor coercion, keeps real military action
GOLDSTEIN_THRESHOLD = -3.0

# More specific CAMEO code → sub-type mapping
CAMEO_SUBTYPE_MAP = {
    "181": ("Battles", "Abduction/Forced disappearance"),
    "182": ("Battles", "Armed assault"),
    "183": ("Explosions/Remote violence", "Suicide bombing"),
    "190": ("Battles", "Use conventional military force"),
    "191": ("Battles", "Armed engagement"),
    "192": ("Battles", "Military attack"),
    "193": ("Battles", "Battle/Military clash"),
    "194": ("Battles", "Military occupation"),
    "195": ("Explosions/Remote violence", "Air/Drone strike"),
    "200": ("Violence against civilians", "Mass violence"),
    "201": ("Violence against civilians", "Mass expulsion"),
    "202": ("Violence against civilians", "Mass killing"),
    "203": ("Violence against civilians", "Ethnic cleansing"),
    "204": ("Violence against civilians", "Use of WMDs"),
}

# Maritime chokepoint bounding boxes (lat_min, lat_max, lon_min, lon_max)
CHOKEPOINTS = {
    "hormuz": (25.5, 27.0, 55.5, 57.5),
    "suez": (29.5, 31.5, 32.0, 34.0),
    "bab_el_mandeb": (12.0, 13.5, 42.5, 44.5),
    "malacca": (1.0, 4.0, 100.0, 104.5),
    "taiwan_strait": (23.0, 26.0, 117.0, 121.0),
}

PROXIMITY_DEGREES = 3.0

# GDELT Event CSV column positions (0-indexed, tab-delimited)
COL_GLOBAL_EVENT_ID = 0
COL_DAY = 1
COL_YEAR = 3
COL_ACTOR1_NAME = 6
COL_ACTOR1_COUNTRY = 7
COL_ACTOR2_NAME = 16
COL_ACTOR2_COUNTRY = 17
COL_EVENT_CODE = 26
COL_EVENT_BASE_CODE = 27
COL_EVENT_ROOT_CODE = 28
COL_GOLDSTEIN_SCALE = 30
COL_NUM_MENTIONS = 31
COL_AVG_TONE = 34
COL_ACTION_GEO_FULLNAME = 52
COL_ACTION_GEO_COUNTRY = 53
COL_ACTION_GEO_ADM1 = 54
COL_ACTION_GEO_LAT = 56
COL_ACTION_GEO_LONG = 57
COL_SOURCE_URL = 60

# Region mapping — ActionGeo country code → region label
COUNTRY_REGION_MAP = {
    "IR": "Middle East",
    "IZ": "Middle East",
    "SY": "Middle East",
    "YM": "Middle East",
    "IS": "Middle East",
    "LE": "Middle East",
    "SA": "Middle East",
    "QA": "Middle East",
    "BA": "Middle East",
    "KU": "Middle East",
    "MU": "Middle East",
    "JO": "Middle East",
    "AE": "Middle East",
    "TU": "Middle East",
    "UP": "Eastern Europe",
    "RS": "Eastern Europe",
    "BO": "Eastern Europe",
    "MD": "Eastern Europe",
    "LY": "Northern Africa",
    "EG": "Northern Africa",
    "TS": "Northern Africa",
    "AG": "Northern Africa",
    "SU": "Northern Africa",
    "SO": "Eastern Africa",
    "ET": "Eastern Africa",
    "ER": "Eastern Africa",
    "DJ": "Eastern Africa",
    "IN": "Southern Asia",
    "PK": "Southern Asia",
    "AF": "Southern Asia",
    "CH": "Eastern Asia",
    "TW": "Eastern Asia",
    "KN": "Eastern Asia",
    "KS": "Eastern Asia",
    "BM": "South-Eastern Asia",
    "RP": "South-Eastern Asia",
    "US": "North America",
    "CA": "North America",
    "UK": "Western Europe",
    "FR": "Western Europe",
    "GM": "Western Europe",
    "IT": "Western Europe",
}


def _check_chokepoint(lat: float, lon: float) -> str:
    """Check if an event is near a critical maritime chokepoint."""
    for name, (lat_min, lat_max, lon_min, lon_max) in CHOKEPOINTS.items():
        if (
            lat_min - PROXIMITY_DEGREES <= lat <= lat_max + PROXIMITY_DEGREES
            and lon_min - PROXIMITY_DEGREES <= lon <= lon_max + PROXIMITY_DEGREES
        ):
            return name
    return "none"


def _make_id(event_id: str) -> str:
    """Deterministic ID from GDELT event ID."""
    return hashlib.md5(f"gdelt_event:{event_id}".encode()).hexdigest()


def _classify_event(event_code: str, root_code: str) -> tuple[str, str]:
    """Map CAMEO event code to ACLED-compatible event type."""
    # Check specific codes first
    if event_code in CAMEO_SUBTYPE_MAP:
        return CAMEO_SUBTYPE_MAP[event_code]
    # Fall back to root code
    if root_code in CAMEO_TYPE_MAP:
        return CAMEO_TYPE_MAP[root_code]
    return ("Battles", "Armed event")


def _safe_float(val: str) -> float | None:
    """Safely convert string to float."""
    try:
        return float(val) if val.strip() else None
    except (ValueError, TypeError):
        return None


async def _download_latest_export(client: httpx.AsyncClient) -> list[list[str]]:
    """Download and parse the latest GDELT Event CSV export file."""
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
        logger.warning("[conflict] No export file found in lastupdate.txt")
        return []

    # Step 2: Download the ZIP
    r2 = await client.get(export_url, timeout=30)
    r2.raise_for_status()

    # Step 3: Extract and parse the CSV
    zf = zipfile.ZipFile(io.BytesIO(r2.content))
    csv_name = zf.namelist()[0]

    with zf.open(csv_name) as f:
        text = f.read().decode("utf-8", errors="replace")

    rows = [line.split("\t") for line in text.strip().split("\n") if line.strip()]
    logger.info(
        "[conflict] Downloaded %d events from %s", len(rows), export_url.split("/")[-1]
    )
    logger.info(
        f"  [conflict] Downloaded {len(rows)} events from {export_url.split('/')[-1]}"
    )
    return rows


async def collect_conflict_events() -> int:
    """
    Download the latest GDELT Event CSV export, filter for conflict CAMEO codes,
    and write structured events to global.conflict_events.
    Returns total number of rows written.
    """
    total_count = 0

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            rows = await _download_latest_export(client)
        except Exception as e:
            logger.error("[conflict] Failed to download GDELT export: %s", e)
            logger.info(f"  [conflict] Failed to download GDELT export: {e}")
            return 0

    if not rows:
        logger.info("  [conflict] No events in latest GDELT export")
        return 0

    with get_db() as db:
        logger.info(f"  [conflict] Processing {len(rows)} events from GDELT export...")

        now = datetime.now(timezone.utc)
        chokepoint_count = 0

        skipped_short = 0
        skipped_non_conflict = 0
        skipped_no_coords = 0
        skipped_bad_coords = 0

        for row in rows:
            if len(row) < 61:
                skipped_short += 1
                continue

            # Filter: only conflict CAMEO root codes
            root_code = row[COL_EVENT_ROOT_CODE].strip()
            if root_code not in CONFLICT_ROOTS:
                skipped_non_conflict += 1
                continue

            # Filter: only strategic/geopolitical countries
            country_code_raw = row[COL_ACTION_GEO_COUNTRY].strip()
            if country_code_raw not in STRATEGIC_COUNTRIES:
                skipped_non_conflict += 1
                continue

            # Filter: only sufficiently severe events (Goldstein scale)
            goldstein_raw = _safe_float(row[COL_GOLDSTEIN_SCALE])
            if goldstein_raw is not None and goldstein_raw > GOLDSTEIN_THRESHOLD:
                skipped_non_conflict += 1
                continue

            # Must have valid coordinates
            lat = _safe_float(row[COL_ACTION_GEO_LAT])
            lon = _safe_float(row[COL_ACTION_GEO_LONG])
            if lat is None or lon is None:
                skipped_no_coords += 1
                continue

            # Sanity check coordinates
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                skipped_bad_coords += 1
                continue

            event_id = row[COL_GLOBAL_EVENT_ID].strip()
            event_code = row[COL_EVENT_CODE].strip()
            event_date_str = row[COL_DAY].strip()
            year = row[COL_YEAR].strip()

            # Parse event date (format: YYYYMMDD)
            try:
                event_date = datetime.strptime(event_date_str, "%Y%m%d").strftime(
                    "%Y-%m-%d"
                )
            except ValueError:
                event_date = now.strftime("%Y-%m-%d")

            actor1 = row[COL_ACTOR1_NAME].strip()[:100]
            actor2 = row[COL_ACTOR2_NAME].strip()[:100]
            country_code = row[COL_ACTION_GEO_COUNTRY].strip()
            geo_fullname = row[COL_ACTION_GEO_FULLNAME].strip()[:200]
            adm1 = row[COL_ACTION_GEO_ADM1].strip()[:50]

            # Map to ACLED-compatible types
            event_type, sub_event_type = _classify_event(event_code, root_code)

            # Derive region from country code
            region = COUNTRY_REGION_MAP.get(country_code, "Other")

            # Check chokepoint proximity
            chokepoint = _check_chokepoint(lat, lon)
            if chokepoint != "none":
                chokepoint_count += 1

            # Goldstein scale as a severity proxy (more negative = more conflict)
            goldstein = _safe_float(row[COL_GOLDSTEIN_SCALE])
            tone = _safe_float(row[COL_AVG_TONE])
            num_mentions = (
                int(row[COL_NUM_MENTIONS]) if row[COL_NUM_MENTIONS].strip() else 0
            )

            # Build a descriptive note from available data
            source_url = (
                row[COL_SOURCE_URL].strip() if len(row) > COL_SOURCE_URL else ""
            )
            note = f"{geo_fullname}"
            if goldstein is not None:
                note += f" | goldstein={goldstein:.1f}"
            if tone is not None:
                note += f" | tone={tone:.1f}"
            note += f" | mentions={num_mentions}"

            row_id = _make_id(event_id)

            try:
                db.execute(
                    """
                    INSERT INTO global.conflict_events
                    (id, event_id_acled, event_date, year, event_type, sub_event_type,
                     actor1, actor2, country, region, admin1,
                     latitude, longitude, fatalities, notes, source_acled,
                     chokepoint_proximity, source, collected_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'gdelt', %s)
                    ON CONFLICT (id) DO UPDATE SET
                        notes = EXCLUDED.notes,
                        collected_at = EXCLUDED.collected_at
                """,
                    [
                        row_id,
                        int(event_id) if event_id.isdigit() else 0,
                        event_date,
                        int(year) if year.isdigit() else now.year,
                        event_type,
                        sub_event_type,
                        actor1,
                        actor2,
                        country_code,
                        region,
                        adm1,
                        lat,
                        lon,
                        0,  # fatalities not available in GDELT events
                        note[:500],
                        source_url[:200],  # stored in source_acled for traceability
                        chokepoint,
                        now,
                    ],
                )
                total_count += 1
            except Exception as e:
                if total_count == 0:
                    # Log only the first error to avoid spam
                    logger.error("[conflict] DB write error: %s", e)
                    logger.info(f"  [conflict] DB write error (first): {e}")
                total_count += 0  # don't increment

        logger.info(
            "[conflict] %d conflict events written (%d near chokepoints)",
            total_count,
            chokepoint_count,
        )
        logger.info(
            f"  [conflict] {total_count} conflict events written "
            f"({chokepoint_count} near chokepoints)"
        )
        logger.info(
            f"  [conflict] Stats: skipped_short={skipped_short} "
            f"non_conflict={skipped_non_conflict} "
            f"no_coords={skipped_no_coords} "
            f"bad_coords={skipped_bad_coords}"
        )

        return {
            "written": total_count,
            "downloaded_rows": len(rows),
            "skipped_short": skipped_short,
            "skipped_non_conflict": skipped_non_conflict,
            "skipped_no_coords": skipped_no_coords,
            "skipped_bad_coords": skipped_bad_coords,
            "chokepoint_events": chokepoint_count,
        }


async def collect_all() -> dict:
    """Main entry point. Downloads latest GDELT event export and extracts conflicts."""
    stats = await collect_conflict_events()
    return {"conflict_events": stats.get("written", 0), "debug": stats}

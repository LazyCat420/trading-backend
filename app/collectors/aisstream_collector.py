"""
AISStream Collector — Real-time AIS vessel positions via WebSocket.

Pure data collector. No LLM calls.
Writes to: global.tanker_positions
Requires: AISSTREAM_API_KEY in .env (free at aisstream.io)

Key data collected:
  - Tanker positions in 4 critical maritime chokepoints
  - Vessel name, speed, heading, destination, flag
  - Client-side filtering for tanker-type vessels

WebSocket API: wss://stream.aisstream.io/v0/stream
Docs: https://aisstream.io/documentation
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone

from app.config import settings
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── Chokepoint Bounding Boxes (AISStream format: [[lat_min, lon_min], [lat_max, lon_max]])
CHOKEPOINT_BBOXES = {
    "hormuz": [
        [23.0, 53.0],
        [28.0, 60.0],
    ],  # Expanded to Gulf of Oman & lower Persian Gulf
    "bab_el_mandeb": [
        [10.0, 41.0],
        [18.0, 46.0],
    ],  # Expanded Southern Red Sea & Gulf of Aden
    "suez": [[27.0, 31.0], [33.0, 35.0]],  # Expanded Northern Red Sea & Eastern Med
    "malacca": [[0.5, 99.5], [4.5, 105.0]],  # (Kept original Malacca)
}

# Vessel type keywords that indicate tankers
TANKER_TYPES = {
    "tanker",
    "oil tanker",
    "chemical tanker",
    "lng carrier",
    "lpg carrier",
    "crude oil tanker",
    "product tanker",
}

POSITION_MESSAGE_TYPES = {
    "PositionReport": "PositionReport",
    "StandardClassBPositionReport": "StandardClassBPositionReport",
    "ExtendedClassBPositionReport": "ExtendedClassBPositionReport",
}

# How long to listen per snapshot (seconds)
LISTEN_DURATION = 30


def _make_id(mmsi: str, zone: str, timestamp: datetime) -> str:
    """Deterministic ID for tanker position snapshot."""
    ts_str = timestamp.strftime("%Y%m%d%H%M")
    return hashlib.md5(f"ais:{mmsi}:{zone}:{ts_str}".encode()).hexdigest()


def _identify_zone(lat: float, lon: float) -> str | None:
    """Determine which chokepoint zone a position falls in."""
    for zone, bbox in CHOKEPOINT_BBOXES.items():
        lat_min, lon_min = bbox[0]
        lat_max, lon_max = bbox[1]
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return zone
    return None


def _normalize_vessel_type(raw_type) -> str | None:
    """Map AIS vessel type fields into energy-shipping labels when possible."""
    if raw_type is None:
        return None

    if isinstance(raw_type, int):
        if 80 <= raw_type <= 89:
            return "Tanker"
        return None

    text = str(raw_type).strip()
    if not text:
        return None

    lowered = text.lower()
    if any(keyword in lowered for keyword in TANKER_TYPES):
        if "lng" in lowered:
            return "LNG Carrier"
        if "lpg" in lowered:
            return "LPG Carrier"
        if "chemical" in lowered:
            return "Chemical Tanker"
        if "product" in lowered:
            return "Product Tanker"
        if "crude" in lowered:
            return "Crude Oil Tanker"
        return "Tanker"

    return None


def _extract_static_profile(msg_type: str, message: dict, metadata: dict) -> dict:
    """Extract reusable vessel metadata from AIS static or extended reports."""
    profile = {
        "vessel_name": (metadata.get("ShipName") or "").strip(),
        "vessel_type": None,
        "destination": "",
        "flag": "",
    }

    if msg_type == "ShipStaticData":
        ship_data = message.get("ShipStaticData", {})
        profile["vessel_name"] = (
            ship_data.get("Name") or profile["vessel_name"] or ""
        ).strip()
        profile["vessel_type"] = _normalize_vessel_type(ship_data.get("Type"))
        profile["destination"] = (ship_data.get("Destination") or "").strip()
        return profile

    if msg_type == "StaticDataReport":
        static_data = message.get("StaticDataReport", {})
        report_a = static_data.get("ReportA", {})
        report_b = static_data.get("ReportB", {})
        profile["vessel_name"] = (
            report_a.get("Name") or profile["vessel_name"] or ""
        ).strip()
        profile["vessel_type"] = _normalize_vessel_type(report_b.get("ShipType"))
        return profile

    if msg_type == "ExtendedClassBPositionReport":
        ext = message.get("ExtendedClassBPositionReport", {})
        profile["vessel_name"] = (
            ext.get("Name") or profile["vessel_name"] or ""
        ).strip()
        profile["vessel_type"] = _normalize_vessel_type(ext.get("Type"))
        return profile

    return profile


def _load_recent_vessel_profiles(db) -> dict[str, dict]:
    """Reuse recent vessel metadata so short live snapshots still classify ships."""
    profiles: dict[str, dict] = {}
    try:
        rows = db.execute("""
            WITH ranked AS (
                SELECT mmsi, vessel_name, vessel_type, destination, flag,
                       ROW_NUMBER() OVER (
                           PARTITION BY mmsi
                           ORDER BY timestamp DESC, collected_at DESC
                       ) AS row_num
                FROM global.tanker_positions
                WHERE collected_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
            )
            SELECT mmsi, vessel_name, vessel_type, destination, flag
            FROM ranked
            WHERE row_num = 1
        """).fetchall()
        for mmsi, vessel_name, vessel_type, destination, flag in rows:
            profiles[str(mmsi)] = {
                "vessel_name": vessel_name or "",
                "vessel_type": vessel_type or None,
                "destination": destination or "",
                "flag": flag or "",
            }
    except Exception as exc:
        logger.debug("[ais] Could not preload vessel profiles: %s", exc)
    return profiles


async def collect_tanker_positions(
    duration_seconds: int = LISTEN_DURATION,
) -> int:
    """
    Connect to AISStream WebSocket, collect tanker positions for
    a brief window, then disconnect.

    Returns number of tanker positions written.
    """
    key = settings.AISSTREAM_API_KEY
    if not key:
        logger.warning("[ais] AISSTREAM_API_KEY not set — skipping")
        logger.info("  [ais] AISSTREAM_API_KEY not set — skipping")
        return 0

    try:
        import websockets
    except ImportError:
        logger.warning(
            "[ais] websockets package not installed — skipping AIS collection"
        )
        logger.info(
            "  [ais] websockets package not installed. Run: pip install websockets"
        )
        return 0

    # Build subscription with all chokepoint bounding boxes
    all_bboxes = list(CHOKEPOINT_BBOXES.values())
    subscription = {
        "Apikey": key,
        "BoundingBoxes": all_bboxes,
        "FilterMessageTypes": [
            "PositionReport",
            "ShipStaticData",
            "StaticDataReport",
            "StandardClassBPositionReport",
            "ExtendedClassBPositionReport",
        ],
    }

    with get_db() as db:
        count = 0
        vessel_profiles = _load_recent_vessel_profiles(db)

        try:
            async with websockets.connect(
                "wss://stream.aisstream.io/v0/stream",
                ping_interval=20,
                ping_timeout=10,
            ) as ws:
                await ws.send(json.dumps(subscription))
                logger.info(
                    "[ais] Connected to AISStream, listening for %ds...",
                    duration_seconds,
                )
                logger.info(
                    f"  [ais] WebSocket connected — listening for {duration_seconds}s across {len(all_bboxes)} zones..."
                )

                # Listen for duration_seconds
                end_time = asyncio.get_event_loop().time() + duration_seconds

                while asyncio.get_event_loop().time() < end_time:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        msg = json.loads(raw)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.debug("[ais] Message parse error: %s", e)
                        continue

                    msg_type = msg.get("MessageType", "")
                    metadata = msg.get("MetaData", {})
                    message = msg.get("Message", {})

                    mmsi = str(metadata.get("MMSI", ""))
                    if not mmsi:
                        continue

                    if msg_type in {
                        "ShipStaticData",
                        "StaticDataReport",
                        "ExtendedClassBPositionReport",
                    }:
                        profile = _extract_static_profile(msg_type, message, metadata)
                        existing = vessel_profiles.get(mmsi, {})
                        vessel_profiles[mmsi] = {
                            "vessel_name": profile.get("vessel_name")
                            or existing.get("vessel_name", ""),
                            "vessel_type": profile.get("vessel_type")
                            or existing.get("vessel_type"),
                            "destination": profile.get("destination")
                            or existing.get("destination", ""),
                            "flag": profile.get("flag") or existing.get("flag", ""),
                        }
                        if msg_type not in POSITION_MESSAGE_TYPES:
                            continue

                    # Process position reports
                    if msg_type in POSITION_MESSAGE_TYPES:
                        pos = message.get(POSITION_MESSAGE_TYPES[msg_type], {})
                        lat = metadata.get("latitude") or pos.get("Latitude")
                        lon = metadata.get("longitude") or pos.get("Longitude")

                        if lat is None or lon is None:
                            continue

                        lat = float(lat)
                        lon = float(lon)
                        zone = _identify_zone(lat, lon)
                        if not zone:
                            continue

                        speed = float(pos.get("Sog", 0) or 0)
                        heading = float(pos.get("TrueHeading", 0) or 0)
                        profile = vessel_profiles.get(mmsi, {})
                        vessel_name = (
                            metadata.get("ShipName")
                            or profile.get("vessel_name")
                            or "UNKNOWN"
                        ).strip()
                        now = datetime.now(timezone.utc)
                        row_id = _make_id(mmsi, zone, now)

                        vessel_type = profile.get("vessel_type") or "Unknown"
                        destination = profile.get("destination", "")
                        flag = profile.get("flag", "")

                        db.execute(
                            """
                            INSERT INTO global.tanker_positions
                            (id, mmsi, vessel_name, vessel_type, latitude, longitude,
                             speed, heading, destination, flag, zone, timestamp, collected_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                vessel_name = EXCLUDED.vessel_name,
                                vessel_type = EXCLUDED.vessel_type,
                                latitude = EXCLUDED.latitude,
                                longitude = EXCLUDED.longitude,
                                speed = EXCLUDED.speed,
                                heading = EXCLUDED.heading,
                                destination = EXCLUDED.destination,
                                flag = EXCLUDED.flag,
                                collected_at = EXCLUDED.collected_at
                        """,
                            [
                                row_id,
                                mmsi,
                                vessel_name,
                                vessel_type,
                                lat,
                                lon,
                                speed,
                                heading,
                                destination,
                                flag,
                                zone,
                                now,
                                now,
                            ],
                        )
                        count += 1

        except Exception as e:
            logger.error("[ais] WebSocket error: %s", e)
            logger.info(f"  [ais] WebSocket connection error: {e}")

        # Zone breakdown
        zone_counts: dict[str, int] = {}
        try:
            rows = db.execute("""
                SELECT zone, COUNT(*) FROM global.tanker_positions
                WHERE collected_at >= CURRENT_TIMESTAMP - INTERVAL '5 minutes'
                GROUP BY zone
            """).fetchall()
            for r in rows:
                zone_counts[r[0]] = r[1]
        except Exception:
            pass

        logger.info("[ais] %d vessel positions written", count)
        logger.info(f"  [ais] {count} vessel positions written")
        if zone_counts:
            for z, c in sorted(zone_counts.items()):
                logger.info(f"    {z}: {c} vessels")
        elif count == 0:
            logger.info(
                "  [ais] (No vessels detected in chokepoint zones during this window — this is normal)"
            )
        return count


async def collect_all() -> dict:
    """Main entry point. Takes a 30-second AIS snapshot."""
    count = await collect_tanker_positions(duration_seconds=LISTEN_DURATION)
    return {"tanker_positions": count}

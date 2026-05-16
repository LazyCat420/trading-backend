"""
Chokepoint Alert Engine — derived analytics from conflict + tanker + news data.

Computes alert levels for critical maritime chokepoints by combining:
  - GDELT-sourced conflict events near the zone (global.conflict_events)
  - AIS tanker positions in the zone (global.tanker_positions)
  - GDELT war news mentioning the zone (global.war_news_feed)

Alert levels:
  NORMAL   — < 3 conflict events in 24h, normal tanker flow
  ELEVATED — 3-10 conflict events OR tanker speed anomaly OR negative news spike
  CRITICAL — active strikes within zone OR tanker reroutes detected

Writes to: global.chokepoint_alerts
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── Chokepoint definitions ──────────────────────────────────────
# Same bounding boxes as acled_collector.py (GDELT conflict collector) for consistency
CHOKEPOINTS = {
    "hormuz": {
        "label": "Strait of Hormuz",
        "bbox": (25.0, 27.5, 55.0, 58.0),
        "keywords": ["hormuz", "iran", "persian gulf"],
    },
    "bab_el_mandeb": {
        "label": "Bab el-Mandeb",
        "bbox": (11.5, 14.0, 42.0, 45.0),
        "keywords": ["bab el-mandeb", "yemen", "houthi", "red sea"],
    },
    "suez": {
        "label": "Suez Canal",
        "bbox": (29.0, 32.0, 31.5, 34.5),
        "keywords": ["suez", "suez canal"],
    },
    "malacca": {
        "label": "Strait of Malacca",
        "bbox": (0.5, 4.5, 99.5, 105.0),
        "keywords": ["malacca", "singapore strait"],
    },
}

PROXIMITY_DEGREES = 3.0  # ~300km expanded search radius


def _make_id(zone: str, timestamp: datetime) -> str:
    """Deterministic ID for chokepoint alert snapshot."""
    ts_str = timestamp.strftime("%Y%m%d%H%M")
    return hashlib.md5(f"chokepoint:{zone}:{ts_str}".encode()).hexdigest()


def _count_zone_conflicts(
    db, zone: str, bbox: tuple, hours: int = 24
) -> tuple[int, int]:
    """Count conflict events near a zone in the last N hours.
    Returns (total_events, strike_events).
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")

    rows = db.execute(
        """
        SELECT COUNT(*) as total,
               SUM(CASE WHEN event_type LIKE '%%Explosion%%'
                        OR event_type LIKE '%%Remote violence%%'
                        THEN 1 ELSE 0 END) as strikes
        FROM global.conflict_events
        WHERE event_date >= %s
          AND latitude BETWEEN %s AND %s
          AND longitude BETWEEN %s AND %s
    """,
        [
            cutoff,
            lat_min - PROXIMITY_DEGREES,
            lat_max + PROXIMITY_DEGREES,
            lon_min - PROXIMITY_DEGREES,
            lon_max + PROXIMITY_DEGREES,
        ],
    ).fetchone()

    return (rows[0] or 0, rows[1] or 0)


def _count_zone_tankers(db, zone: str, hours: int = 24) -> tuple[int, float]:
    """Count tankers in a zone and average speed.
    Returns (tanker_count, avg_speed).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    row = db.execute(
        """
        SELECT COUNT(DISTINCT mmsi) as tanker_count,
               AVG(speed) as avg_speed
        FROM global.tanker_positions
        WHERE zone = %s
          AND timestamp >= %s
    """,
        [zone, cutoff],
    ).fetchone()

    return (row[0] or 0, row[1] or 0.0)


def _count_zone_news(
    db, zone_keywords: list[str], hours: int = 24
) -> tuple[int, float]:
    """Count war news articles mentioning the zone and average tone.
    Returns (news_count, avg_tone).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Build LIKE conditions for zone keywords
    conditions = " OR ".join(["LOWER(headline) LIKE ?"] * len(zone_keywords))
    params = [cutoff] + [f"%{kw}%" for kw in zone_keywords]

    row = db.execute(
        f"""
        SELECT COUNT(*) as news_count,
               AVG(tone) as avg_tone
        FROM global.war_news_feed
        WHERE timestamp >= %s
          AND ({conditions})
    """,
        params,
    ).fetchone()

    return (row[0] or 0, row[1] or 0.0)


def _compute_alert_level(
    conflict_count: int,
    strike_count: int,
    tanker_count: int,
    avg_speed: float,
    news_count: int,
    avg_tone: float,
) -> tuple[str, float]:
    """Compute alert level and price impact score.
    Returns (alert_level, price_impact_score).
    """
    # Start with base score
    score = 0.0

    # Conflict intensity (0-0.4)
    if conflict_count >= 20:
        score += 0.4
    elif conflict_count >= 10:
        score += 0.3
    elif conflict_count >= 3:
        score += 0.15
    elif conflict_count >= 1:
        score += 0.05

    # Active strikes are high severity (0-0.3)
    if strike_count >= 5:
        score += 0.3
    elif strike_count >= 2:
        score += 0.2
    elif strike_count >= 1:
        score += 0.1

    # Tanker anomalies (0-0.15)
    if tanker_count > 0 and avg_speed < 3.0:
        score += 0.15  # Congestion or stopped tankers
    elif tanker_count > 0 and avg_speed < 5.0:
        score += 0.08

    # Negative news spike (0-0.15)
    if news_count >= 10 and avg_tone < -5.0:
        score += 0.15
    elif news_count >= 5 and avg_tone < -3.0:
        score += 0.08
    elif news_count >= 3:
        score += 0.03

    # Determine alert level
    if score >= 0.5 or strike_count >= 3:
        alert_level = "CRITICAL"
    elif score >= 0.2 or conflict_count >= 3:
        alert_level = "ELEVATED"
    else:
        alert_level = "NORMAL"

    return alert_level, min(1.0, score)


async def compute_chokepoint_alerts(hours: int = 24) -> dict:
    """
    Compute alert levels for all 4 chokepoints.
    Combines conflict, tanker, and news data.
    Returns {zone: alert_data}.
    """
    with get_db() as db:
        now = datetime.now(timezone.utc)
        results = {}

        for zone, config in CHOKEPOINTS.items():
            bbox = config["bbox"]
            keywords = config["keywords"]

            # Count events from each source
            conflict_count, strike_count = _count_zone_conflicts(db, zone, bbox, hours)
            tanker_count, avg_speed = _count_zone_tankers(db, zone, hours)
            news_count, avg_tone = _count_zone_news(db, keywords, hours)

            # Compute derived alert
            alert_level, price_impact = _compute_alert_level(
                conflict_count,
                strike_count,
                tanker_count,
                avg_speed,
                news_count,
                avg_tone,
            )

            row_id = _make_id(zone, now)

            # Upsert alert
            db.execute(
                """
                INSERT INTO global.chokepoint_alerts
                (id, zone, alert_level, tanker_count, nearby_conflict_count,
                 avg_tanker_speed, reroute_count, war_news_count,
                 price_impact_score, timestamp, computed_at)
                VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    alert_level = EXCLUDED.alert_level,
                    tanker_count = EXCLUDED.tanker_count,
                    nearby_conflict_count = EXCLUDED.nearby_conflict_count,
                    avg_tanker_speed = EXCLUDED.avg_tanker_speed,
                    war_news_count = EXCLUDED.war_news_count,
                    price_impact_score = EXCLUDED.price_impact_score,
                    computed_at = EXCLUDED.computed_at
            """,
                [
                    row_id,
                    zone,
                    alert_level,
                    tanker_count,
                    conflict_count,
                    avg_speed,
                    news_count,
                    price_impact,
                    now,
                    now,
                ],
            )

            results[zone] = {
                "label": config["label"],
                "alert_level": alert_level,
                "conflicts": conflict_count,
                "strikes": strike_count,
                "tankers": tanker_count,
                "avg_speed": round(avg_speed, 1),
                "news": news_count,
                "price_impact": round(price_impact, 3),
            }

            logger.info(
                "[chokepoint] %s: %s (conflicts=%d, strikes=%d, tankers=%d, news=%d, impact=%.3f)",
                zone,
                alert_level,
                conflict_count,
                strike_count,
                tanker_count,
                news_count,
                price_impact,
            )

        # Summary
        critical = sum(1 for r in results.values() if r["alert_level"] == "CRITICAL")
        elevated = sum(1 for r in results.values() if r["alert_level"] == "ELEVATED")
        logger.info(
            f"  [chokepoint] 4 zones computed: {critical} CRITICAL, {elevated} ELEVATED"
        )

        return results


async def collect_all() -> dict:
    """Main entry point. Recomputes all chokepoint alerts."""
    return await compute_chokepoint_alerts(hours=24)

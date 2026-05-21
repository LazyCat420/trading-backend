"""
War Context Builder — Assembles geopolitical intelligence for LLM prompt injection.

This service aggregates data from multiple sources into a structured context
block that can be injected into the trading bot's LLM prompts, enabling the
model to reason about:
  - Supply disruption risks (chokepoint blockades, tanker reroutes)
  - Conflict escalation trajectories
  - Energy market correlations with geopolitical events

Output format: structured dict suitable for JSON serialization into prompts.

Usage:
  from app.cognition.war_context_builder import build_war_oil_context
  context = build_war_oil_context(window_hours=24)
  # Inject context["prompt_block"] into the LLM system prompt
"""

import logging
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.config.context_budget import get_context_budget
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def build_war_oil_context(window_hours: int = 24) -> dict:
    """
    Assemble geopolitical intelligence context for LLM prompt injection.

    Returns a dict with:
      - conflict_summary: key conflict metrics
      - chokepoint_status: alert levels for 4 maritime zones
      - tanker_disruptions: anomalous tanker behavior
      - war_news_highlights: top negative news headlines
      - gpr_index: latest Geopolitical Risk Index
      - risk_signals: quantified risk flags (0-1 scores)
      - prompt_block: formatted text block ready for LLM injection
    """
    if not settings.WAR_CONTEXT_ENABLED:
        return {"enabled": False, "prompt_block": ""}

    with get_db() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

        # ── 1. Conflict summary ──────────────────────────────────────
        conflict_summary = _get_conflict_summary(db, cutoff)

        # ── 2. Chokepoint alerts ─────────────────────────────────────
        chokepoint_status = _get_chokepoint_alerts(db)

        # ── 3. Tanker anomalies ──────────────────────────────────────
        tanker_disruptions = _get_tanker_anomalies(db, cutoff)

        # ── 4. War news highlights ───────────────────────────────────
        war_news = _get_war_news(db, cutoff, limit=8)

        # ── 5. GPR Index ─────────────────────────────────────────────
        gpr_index = _get_latest_gpr(db)

    # ── 6. Compute risk signals ──────────────────────────────────
    risk_signals = _compute_risk_signals(
        conflict_summary, chokepoint_status, tanker_disruptions, war_news
    )

    # ── 7. Build prompt block ────────────────────────────────────
    prompt_block = _format_prompt_block(
        conflict_summary,
        chokepoint_status,
        tanker_disruptions,
        war_news,
        gpr_index,
        risk_signals,
    )

    # ── Budget-aware truncation ──
    budget = get_context_budget()
    if len(prompt_block) > budget.war_context_chars:
        original_len = len(prompt_block)
        prompt_block = prompt_block[: budget.war_context_chars] + "\n=== END GEOPOLITICAL CONTEXT (TRUNCATED) ==="
        logger.info(
            "[WAR_CONTEXT] Prompt block truncated: %d -> %d chars (budget=%d)",
            original_len,
            len(prompt_block),
            budget.war_context_chars,
        )

    return {
        "enabled": True,
        "conflict_summary": conflict_summary,
        "chokepoint_status": chokepoint_status,
        "tanker_disruptions": tanker_disruptions,
        "war_news_highlights": war_news,
        "gpr_index": gpr_index,
        "risk_signals": risk_signals,
        "prompt_block": prompt_block,
    }


def _get_conflict_summary(db, cutoff: datetime) -> dict:
    """Summarize recent conflict events."""
    try:
        row = db.execute(
            """
            SELECT
                COUNT(*) as total_events,
                SUM(CASE WHEN event_type LIKE '%%Explosion%%'
                    OR event_type LIKE '%%Remote violence%%'
                    THEN 1 ELSE 0 END) as strikes,
                SUM(COALESCE(fatalities, 0)) as total_fatalities,
                COUNT(DISTINCT country) as countries_affected
            FROM global.conflict_events
            WHERE event_date >= %s
        """,
            [cutoff.strftime("%Y-%m-%d")],
        ).fetchone()

        return {
            "total_events": row[0] or 0,
            "strikes": row[1] or 0,
            "total_fatalities": row[2] or 0,
            "countries_affected": row[3] or 0,
        }
    except Exception as e:
        logger.warning("[war_context] Conflict summary error: %s", e)
        return {
            "total_events": 0,
            "strikes": 0,
            "total_fatalities": 0,
            "countries_affected": 0,
        }


def _get_chokepoint_alerts(db) -> list[dict]:
    """Get latest alert for each chokepoint."""
    try:
        rows = db.execute("""
            SELECT zone, alert_level, nearby_conflict_count,
                   tanker_count, price_impact_score
            FROM global.chokepoint_alerts
            WHERE (zone, timestamp) IN (
                SELECT zone, MAX(timestamp) FROM global.chokepoint_alerts GROUP BY zone
            )
        """).fetchall()

        return [
            {
                "zone": r[0],
                "alert_level": r[1],
                "conflicts": r[2],
                "tankers": r[3],
                "price_impact": r[4] or 0.0,
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning("[war_context] Chokepoint alerts error: %s", e)
        return []


def _get_tanker_anomalies(db, cutoff: datetime) -> dict:
    """Detect anomalous tanker behavior (stopped, slow speed)."""
    try:
        row = db.execute(
            """
            SELECT
                COUNT(DISTINCT mmsi) as total_tankers,
                SUM(CASE WHEN speed < 1.0 THEN 1 ELSE 0 END) as stopped_tankers,
                SUM(CASE WHEN speed BETWEEN 1.0 AND 3.0 THEN 1 ELSE 0 END) as slow_tankers,
                COUNT(DISTINCT zone) as zones_active
            FROM global.tanker_positions
            WHERE timestamp >= %s
        """,
            [cutoff],
        ).fetchone()

        return {
            "total_tankers": row[0] or 0,
            "stopped": row[1] or 0,
            "slow_speed": row[2] or 0,
            "zones_active": row[3] or 0,
        }
    except Exception as e:
        logger.warning("[war_context] Tanker anomalies error: %s", e)
        return {"total_tankers": 0, "stopped": 0, "slow_speed": 0, "zones_active": 0}


def _get_war_news(db, cutoff: datetime, limit: int = 8) -> list[dict]:
    """Get most negative/impactful war news headlines."""
    try:
        rows = db.execute(
            """
            SELECT headline, source_domain, tone, location_name
            FROM global.war_news_feed
            WHERE timestamp >= %s
            ORDER BY tone ASC
            LIMIT %s
        """,
            [cutoff, limit],
        ).fetchall()

        return [
            {
                "headline": r[0],
                "source": r[1],
                "tone": r[2],
                "location": r[3],
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning("[war_context] War news error: %s", e)
        return []


def _get_latest_gpr(db) -> dict:
    """Get the latest Geopolitical Risk Index value from intelligence_briefs."""
    try:
        row = db.execute("""
            SELECT risk_level, anomaly_count, period_end
            FROM global.intelligence_briefs
            WHERE brief_type = 'geopolitical'
            ORDER BY period_end DESC
            LIMIT 1
        """).fetchone()

        if row:
            # Map risk_level to a numeric GPR proxy
            level_map = {"NORMAL": 50, "ELEVATED": 120, "CRITICAL": 250}
            gpr_value = level_map.get(row[0], 50)
            return {
                "value": gpr_value,
                "threats": row[1] or 0,
                "acts": 0,
                "date": str(row[2]),
            }
        return {"value": None}
    except Exception as e:
        logger.warning("[war_context] GPR index error: %s", e)
        return {"value": None}


def _compute_risk_signals(
    conflicts: dict,
    chokepoints: list[dict],
    tankers: dict,
    news: list[dict],
) -> dict:
    """Compute quantified risk signals from all sources."""
    # OIL_SUPPLY_DISRUPTION_RISK (0-1)
    disruption_risk = 0.0
    critical_zones = sum(1 for cp in chokepoints if cp.get("alert_level") == "CRITICAL")
    elevated_zones = sum(1 for cp in chokepoints if cp.get("alert_level") == "ELEVATED")

    disruption_risk += critical_zones * 0.3
    disruption_risk += elevated_zones * 0.1
    if tankers.get("stopped", 0) >= 3:
        disruption_risk += 0.2
    if conflicts.get("strikes", 0) >= 10:
        disruption_risk += 0.15

    # SHIPPING_REROUTE_ALERT
    reroute_alert = tankers.get("stopped", 0) >= 2 or critical_zones >= 1

    # HORMUZ_TENSION_INDEX (0-100)
    hormuz = next((cp for cp in chokepoints if cp.get("zone") == "hormuz"), {})
    hormuz_index = int(
        min(
            100,
            (hormuz.get("price_impact", 0) * 100) + (hormuz.get("conflicts", 0) * 2),
        )
    )

    # OVERALL_GEOPOLITICAL_RISK (0-1)
    avg_news_tone = 0.0
    if news:
        avg_news_tone = sum(n.get("tone", 0) for n in news) / len(news)
    overall_risk = min(1.0, disruption_risk + (abs(avg_news_tone) / 20))

    return {
        "OIL_SUPPLY_DISRUPTION_RISK": round(min(1.0, disruption_risk), 3),
        "SHIPPING_REROUTE_ALERT": reroute_alert,
        "HORMUZ_TENSION_INDEX": hormuz_index,
        "OVERALL_GEOPOLITICAL_RISK": round(overall_risk, 3),
    }


def _format_prompt_block(
    conflicts: dict,
    chokepoints: list[dict],
    tankers: dict,
    news: list[dict],
    gpr: dict,
    signals: dict,
) -> str:
    """Format all intel into a text block for LLM prompt injection."""
    lines = [
        "=== GEOPOLITICAL INTELLIGENCE CONTEXT ===",
        "",
    ]

    # Risk signals
    lines.append("RISK SIGNALS:")
    lines.append(
        f"  Oil Supply Disruption Risk: {signals['OIL_SUPPLY_DISRUPTION_RISK']:.1%}"
    )
    lines.append(
        f"  Shipping Reroute Alert: {'🚨 YES' if signals['SHIPPING_REROUTE_ALERT'] else 'No'}"
    )
    lines.append(f"  Hormuz Tension Index: {signals['HORMUZ_TENSION_INDEX']}/100")
    lines.append(
        f"  Overall Geopolitical Risk: {signals['OVERALL_GEOPOLITICAL_RISK']:.1%}"
    )
    lines.append("")

    # Conflict summary
    lines.append("CONFLICT SUMMARY (24h):")
    lines.append(
        f"  Events: {conflicts['total_events']} ({conflicts['strikes']} strikes)"
    )
    lines.append(f"  Fatalities: {conflicts['total_fatalities']}")
    lines.append(f"  Countries: {conflicts['countries_affected']}")
    lines.append("")

    # Chokepoints
    if chokepoints:
        lines.append("MARITIME CHOKEPOINTS:")
        for cp in chokepoints:
            icon = (
                "🔴"
                if cp.get("alert_level") == "CRITICAL"
                else "🟡"
                if cp.get("alert_level") == "ELEVATED"
                else "🟢"
            )
            lines.append(
                f"  {icon} {cp['zone'].replace('_', ' ').title()}: "
                f"{cp.get('alert_level', 'NORMAL')} "
                f"(conflicts={cp.get('conflicts', 0)}, "
                f"tankers={cp.get('tankers', 0)}, "
                f"impact={cp.get('price_impact', 0):.1%})"
            )
        lines.append("")

    # Tanker status
    if tankers.get("total_tankers", 0) > 0:
        lines.append("TANKER STATUS:")
        lines.append(
            f"  Active: {tankers['total_tankers']} | "
            f"Stopped: {tankers['stopped']} | "
            f"Slow: {tankers['slow_speed']}"
        )
        lines.append("")

    # GPR index
    if gpr.get("value") is not None:
        lines.append(f"GPR INDEX: {gpr['value']:.1f} (as of {gpr.get('date', '?')})")
        lines.append("")

    # Top negative headlines
    if news:
        lines.append("TOP WAR/OIL HEADLINES (most negative):")
        for n in news[:5]:
            tone_label = f"({n['tone']:.1f})" if n.get("tone") is not None else ""
            lines.append(f"  • {n['headline'][:100]} {tone_label}")
        lines.append("")

    # Trading implications
    if signals["OIL_SUPPLY_DISRUPTION_RISK"] >= 0.3:
        lines.append(
            "⚠️ TRADING IMPLICATION: Elevated supply disruption risk. "
            "Consider energy sector exposure (XLE, OXY, USO). "
            "Short-term WTI/Brent upside probable."
        )
    elif signals["OVERALL_GEOPOLITICAL_RISK"] >= 0.5:
        lines.append(
            "⚠️ TRADING IMPLICATION: High geopolitical risk environment. "
            "Consider defensive positioning, gold exposure (GLD), "
            "and reduced equity beta."
        )

    lines.append("=== END GEOPOLITICAL CONTEXT ===")

    return "\n".join(lines)

"""
Subsystem Benchmarks — Per-cycle metrics for every pipeline subsystem.

Records granular performance data that powers:
  1. Cycle-over-cycle trend analysis
  2. Rollback degradation detection
  3. Frontend observability dashboard
"""

import json
import logging
import uuid

from app.db.connection import get_db

logger = logging.getLogger(__name__)


def record_subsystem(cycle_id: str, subsystem: str, metrics: dict):
    """Record metrics for a single subsystem in a given cycle."""
    with get_db() as db:
        try:
            db.execute(
                "INSERT INTO subsystem_benchmarks (id, cycle_id, subsystem, metrics) "
                "VALUES (%s, %s, %s, %s)",
                [str(uuid.uuid4()), cycle_id, subsystem, json.dumps(metrics)],
            )
        except Exception as e:
            logger.warning(
                "[SUB-BENCH] Failed to record %s metrics for %s: %s",
                subsystem,
                cycle_id,
                e,
            )


def record_debate_metrics(cycle_id: str):
    """Snapshot debate quality metrics from pending_evolution_fixes."""
    with get_db() as db:
        try:
            rows = db.execute(
                "SELECT status, judge_score FROM pending_evolution_fixes "
                "WHERE cycle_id = %s",
                [cycle_id],
            ).fetchall()

            if not rows:
                return

            total = len(rows)
            approved = sum(
                1 for r in rows if r[0] in ("pending", "approved", "deployed")
            )
            rejected = sum(1 for r in rows if r[0] == "rejected")
            rolled_back = sum(1 for r in rows if r[0] == "rolled_back")
            scores = [r[1] for r in rows if r[1] is not None]
            avg_score = sum(scores) / len(scores) if scores else 0

            record_subsystem(
                cycle_id,
                "debate",
                {
                    "proposals_total": total,
                    "approved": approved,
                    "rejected": rejected,
                    "rolled_back": rolled_back,
                    "approval_rate": round(approved / max(total, 1) * 100, 1),
                    "avg_judge_score": round(avg_score, 2),
                },
            )

        except Exception as e:
            logger.warning("[SUB-BENCH] Failed to record debate metrics: %s", e)


def record_autoresearch_metrics(cycle_id: str):
    """Snapshot autoresearch scores for trend tracking."""
    with get_db() as db:
        try:
            row = db.execute(
                "SELECT overall_score, data_quality_score, "
                "       decision_quality_score, llm_performance_score "
                "FROM autoresearch_reports WHERE cycle_id = %s",
                [cycle_id],
            ).fetchone()

            if not row:
                return

            record_subsystem(
                cycle_id,
                "autoresearch",
                {
                    "overall_score": row[0] or 0,
                    "data_quality_score": row[1] or 0,
                    "decision_quality_score": row[2] or 0,
                    "llm_performance_score": row[3] or 0,
                },
            )

        except Exception as e:
            logger.warning("[SUB-BENCH] Failed to record autoresearch metrics: %s", e)


def record_collection_metrics(cycle_id: str):
    """Snapshot collection timing from cycle_benchmarks."""
    with get_db() as db:
        try:
            row = db.execute(
                "SELECT collect_ms, cache_hit_pct, steps_total, "
                "       steps_skipped, steps_error "
                "FROM cycle_benchmarks WHERE cycle_id = %s",
                [cycle_id],
            ).fetchone()

            if not row:
                return

            record_subsystem(
                cycle_id,
                "collection",
                {
                    "collect_ms": row[0] or 0,
                    "cache_hit_pct": row[1] or 0,
                    "steps_total": row[2] or 0,
                    "steps_skipped": row[3] or 0,
                    "steps_error": row[4] or 0,
                },
            )

        except Exception as e:
            logger.warning("[SUB-BENCH] Failed to record collection metrics: %s", e)


def record_all(cycle_id: str):
    """Record all subsystem benchmarks for a cycle. Safe to call multiple times."""
    logger.info("[SUB-BENCH] Recording subsystem benchmarks for %s", cycle_id)
    record_autoresearch_metrics(cycle_id)
    record_debate_metrics(cycle_id)
    record_collection_metrics(cycle_id)
    logger.info("[SUB-BENCH] Subsystem benchmarks recorded for %s", cycle_id)


def get_trends(subsystem: str, limit: int = 10) -> list[dict]:
    """Get the last N benchmark entries for a subsystem, ordered by time."""
    with get_db() as db:
        try:
            rows = db.execute(
                "SELECT cycle_id, metrics, created_at "
                "FROM subsystem_benchmarks "
                "WHERE subsystem = %s "
                "ORDER BY created_at DESC LIMIT %s",
                [subsystem, limit],
            ).fetchall()

            return [
                {
                    "cycle_id": r[0],
                    "metrics": json.loads(r[1]) if isinstance(r[1], str) else r[1],
                    "created_at": str(r[2]),
                }
                for r in reversed(rows)  # Return oldest-first for charts
            ]

        except Exception as e:
            logger.warning("[SUB-BENCH] Failed to get trends for %s: %s", subsystem, e)
            return []


def get_all_trends(limit: int = 10) -> dict[str, list[dict]]:
    """Get trends for ALL subsystems in one call."""
    subsystems = ["autoresearch", "debate", "collection"]
    return {s: get_trends(s, limit) for s in subsystems}

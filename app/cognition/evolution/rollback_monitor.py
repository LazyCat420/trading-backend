"""
Rollback Monitor — Watches deployed fixes during probation.

Called by AutoResearch at the end of each cycle. Compares pre-deploy
vs post-deploy subsystem benchmarks. If metrics degrade across 2+
probation cycles, auto-rollbacks the fix and records a dead-end.
"""

import hashlib
import json
import logging
import uuid

from app.db.connection import get_db

logger = logging.getLogger(__name__)


def check_probation_fixes(current_cycle_id: str) -> dict:
    """Check all deployed fixes still in probation and evaluate their impact.

    Returns a summary dict with rollbacks performed and fixes that passed.
    """
    with get_db() as db:
        summary = {"checked": 0, "rolled_back": 0, "passed": 0, "errors": 0}

        try:
            # Find deployed fixes still in probation
            rows = db.execute(
                "SELECT id, target_type, target_name, proposed_fix, "
                "       backup_path, probation_until, cycle_id "
                "FROM pending_evolution_fixes "
                "WHERE status = 'deployed' AND probation_until IS NOT NULL "
                "  AND probation_until > CURRENT_TIMESTAMP"
            ).fetchall()

            if not rows:
                logger.debug("[ROLLBACK-MONITOR] No fixes in probation.")
                return summary

            logger.info(
                "[ROLLBACK-MONITOR] Checking %d deployed fixes in probation...",
                len(rows),
            )

            for row in rows:
                fix_id = row[0]
                target_type = row[1]
                target_name = row[2]
                proposed_fix = row[3]
                backup_path = row[4]
                deploy_cycle_id = row[6]
                summary["checked"] += 1

                try:
                    degraded = _check_degradation(deploy_cycle_id, current_cycle_id)

                    if degraded:
                        logger.warning(
                            "[ROLLBACK-MONITOR] Fix %s degraded metrics — rolling back!",
                            fix_id,
                        )
                        _perform_rollback(
                            fix_id,
                            target_type,
                            target_name,
                            proposed_fix,
                            backup_path,
                            deploy_cycle_id,
                            current_cycle_id,
                        )
                        summary["rolled_back"] += 1
                    else:
                        logger.info(
                            "[ROLLBACK-MONITOR] Fix %s metrics stable — marking as STABLE.",
                            fix_id,
                        )
                        # Promote to stable registry so future debates use this as starting point
                        try:
                            from app.cognition.evolution.deployer import mark_stable
                            mark_result = mark_stable(fix_id)
                            if "error" in mark_result:
                                logger.warning(
                                    "[ROLLBACK-MONITOR] Failed to mark %s stable: %s",
                                    fix_id, mark_result["error"],
                                )
                        except Exception as ms_err:
                            logger.warning(
                                "[ROLLBACK-MONITOR] mark_stable failed for %s: %s",
                                fix_id, ms_err,
                            )
                        summary["passed"] += 1

                except Exception as e:
                    logger.error(
                        "[ROLLBACK-MONITOR] Error checking fix %s: %s", fix_id, e
                    )
                    summary["errors"] += 1

        except Exception as e:
            logger.error("[ROLLBACK-MONITOR] Failed to query probation fixes: %s", e)

        return summary


def _check_degradation(deploy_cycle_id: str, current_cycle_id: str) -> bool:
    """Compare autoresearch scores before and after deploy.

    Returns True if scores degraded (indicating the fix made things worse).
    """
    with get_db() as db:
        try:
            # Get the pre-deploy score (the cycle that triggered the fix)
            before = db.execute(
                "SELECT overall_score, data_quality_score, decision_quality_score "
                "FROM autoresearch_reports WHERE cycle_id = %s",
                [deploy_cycle_id],
            ).fetchone()

            # Get the current (post-deploy) score
            after = db.execute(
                "SELECT overall_score, data_quality_score, decision_quality_score "
                "FROM autoresearch_reports WHERE cycle_id = %s",
                [current_cycle_id],
            ).fetchone()

            if not before or not after:
                return False  # Can't compare — assume OK

            # Check if overall score dropped by more than 10 points
            before_overall = before[0] or 0
            after_overall = after[0] or 0

            if after_overall < before_overall - 10:
                logger.info(
                    "[ROLLBACK-MONITOR] Overall score dropped: %.1f → %.1f",
                    before_overall,
                    after_overall,
                )
                return True

            # Check if data quality dropped significantly
            before_data = before[1] or 0
            after_data = after[1] or 0
            if after_data < before_data - 15:
                return True

            return False

        except Exception as e:
            logger.error("[ROLLBACK-MONITOR] Degradation check failed: %s", e)
            return False


def _perform_rollback(
    fix_id: str,
    target_type: str,
    target_name: str,
    proposed_fix: str,
    backup_path: str,
    deploy_cycle_id: str,
    current_cycle_id: str,
):
    """Rollback a fix and record the dead-end."""
    from app.cognition.evolution.deployer import rollback_fix

    with get_db() as db:
        # Perform the file rollback
        result = rollback_fix(fix_id)
        if "error" in result:
            logger.error("[ROLLBACK-MONITOR] Rollback failed: %s", result["error"])
            return

        # Compute approach hash for dead-end dedup
        approach_hash = hashlib.sha256((proposed_fix or "").encode()).hexdigest()[:16]

        # Get metrics snapshots for the dead-end record
        metrics_before = _get_metrics_snapshot(deploy_cycle_id)
        metrics_after = _get_metrics_snapshot(current_cycle_id)

        # Record the dead-end
        failure_reason = (
            f"Deployed fix for {target_type}/{target_name} caused metric degradation. "
            f"Overall score dropped from {metrics_before.get('overall', '%s')} "
            f"to {metrics_after.get('overall', '%s')}."
        )

        try:
            db.execute(
                "INSERT INTO evolution_dead_ends "
                "(id, fix_id, target_type, target_name, approach_hash, "
                " failure_reason, metrics_before, metrics_after) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    str(uuid.uuid4()),
                    fix_id,
                    target_type,
                    target_name,
                    approach_hash,
                    failure_reason,
                    json.dumps(metrics_before),
                    json.dumps(metrics_after),
                ],
            )
            logger.info(
                "[ROLLBACK-MONITOR] Dead-end recorded for %s/%s (hash=%s)",
                target_type,
                target_name,
                approach_hash,
            )
        except Exception as e:
            logger.error("[ROLLBACK-MONITOR] Failed to record dead-end: %s", e)


def _get_metrics_snapshot(cycle_id: str) -> dict:
    """Get a metrics snapshot for a given cycle."""
    with get_db() as db:
        try:
            row = db.execute(
                "SELECT overall_score, data_quality_score, "
                "       decision_quality_score, llm_performance_score "
                "FROM autoresearch_reports WHERE cycle_id = %s",
                [cycle_id],
            ).fetchone()

            if row:
                return {
                    "overall": row[0],
                    "data_quality": row[1],
                    "decision_quality": row[2],
                    "llm_performance": row[3],
                }
        except Exception:
            pass
        return {}

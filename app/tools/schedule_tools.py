"""
Schedule Management Tools — Agentic self-scheduling for the trading bot.

Allows the bot to create, update, and inspect its own cycle schedules.
This closes the autonomy gap where cycles complete but the bot has no
mechanism to plan its own next execution.

Safety limits:
  - Max 6 active schedules (prevents runaway schedule creation)
  - Minimum interval: 1 hour (prevents self-DoS)
  - Maximum interval: 168 hours / 1 week
  - Cannot delete the last active schedule via tool (must use UI)
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from app.db.connection import get_db
from app.tools.registry import registry, PermissionLevel

logger = logging.getLogger(__name__)

# ── Hard Safety Limits ──
MAX_ACTIVE_SCHEDULES = 6
MIN_INTERVAL_HOURS = 1.0
MAX_INTERVAL_HOURS = 168.0  # 1 week
DEFAULT_INTERVAL_HOURS = 4.0
DEFAULT_SCHEDULE_NAME = "Auto-Recovery Schedule"


def _count_active_schedules() -> int:
    """Count currently active schedules in the database."""
    with get_db() as db:
        row = db.execute(
            "SELECT COUNT(*) FROM cycle_schedules WHERE is_active = TRUE"
        ).fetchone()
        return row[0] if row else 0


def _ensure_default_schedule(
    interval_hours: float = DEFAULT_INTERVAL_HOURS,
    market_hours_only: bool = True,
) -> dict:
    """Create a default schedule if none exist. Used by the Schedule Guardian.

    This is a deterministic safety net — not LLM-driven. It ensures the bot
    always has at least one schedule to wake itself up.

    Returns:
        dict with status and job_id (if created)
    """
    active = _count_active_schedules()
    if active > 0:
        return {"status": "exists", "active_count": active}

    with get_db() as db:
        job_id = "sch-default"
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            """
            INSERT INTO cycle_schedules (
                id, name, schedule_type, cron_expression, interval_hours,
                collect, "analyze", trade, tickers, max_tickers, market_hours_only,
                is_active, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                job_id,
                DEFAULT_SCHEDULE_NAME,
                "interval",
                None,
                interval_hours,
                True,  # collect
                True,  # analyze
                None,  # trade = armed (default)
                "[]",  # watchlist
                None,  # max_tickers = unlimited
                market_hours_only,
                True,  # is_active
                now,
                now,
            ],
        )

        # Verify the INSERT persisted
        verify = db.execute(
            "SELECT id FROM cycle_schedules WHERE id = %s", [job_id]
        ).fetchone()
        if not verify:
            logger.error("[SCHEDULE-TOOL] INSERT did not persist for %s!", job_id)
            return {"status": "error", "reason": "INSERT did not persist"}

    # Refresh the APScheduler engine so it picks up the new job
    try:
        from app.services.cycle_scheduler import SchedulerService

        SchedulerService.refresh_job(job_id)
    except Exception as e:
        logger.warning(
            "[SCHEDULE-TOOL] APScheduler refresh failed (non-fatal, will load on next start): %s",
            e,
        )

    logger.info(
        "[SCHEDULE-TOOL] Created default schedule: %s (every %.1fh, market_hours=%s) — verified in DB",
        job_id,
        interval_hours,
        market_hours_only,
    )
    return {"status": "created", "job_id": job_id, "interval_hours": interval_hours}


# ═══════════════════════════════════════════════════════════════════
# TOOL: create_or_update_schedule
# ═══════════════════════════════════════════════════════════════════


@registry.register(
    name="create_or_update_schedule",
    description=(
        "Create a new automated trading cycle schedule or update an existing one. "
        "Use this after completing a trading cycle to ensure the bot runs again. "
        "Parameters: name (str), interval_hours (float, 1-168), collect (bool), "
        "analyze (bool), market_hours_only (bool). "
        "Safety: max 6 active schedules, min 1h interval."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name for the schedule (e.g. 'Post-Trade Follow-Up')",
            },
            "interval_hours": {
                "type": "number",
                "description": "Hours between runs (1-168). Default: 4",
            },
            "cron_expression": {
                "type": "string",
                "description": "Cron expression (e.g. '30 9 * * 1-5'). Use INSTEAD of interval_hours for clock-based scheduling.",
            },
            "collect": {
                "type": "boolean",
                "description": "Whether to collect fresh data. Default: true",
            },
            "analyze": {
                "type": "boolean",
                "description": "Whether to run analysis. Default: true",
            },
            "market_hours_only": {
                "type": "boolean",
                "description": "Only run during US market hours. Default: true",
            },
            "update_schedule_id": {
                "type": "string",
                "description": "If provided, update this existing schedule instead of creating a new one.",
            },
        },
        "required": ["name"],
    },
    permission=PermissionLevel.WRITE,
    tier=2,
    source="scheduler",
    tags=["schedule", "autonomy", "self-management"],
)
async def create_or_update_schedule(
    name: str,
    interval_hours: float = DEFAULT_INTERVAL_HOURS,
    cron_expression: str | None = None,
    collect: bool = True,
    analyze: bool = True,
    market_hours_only: bool = True,
    update_schedule_id: str | None = None,
) -> str:
    """Create or update a cycle schedule."""
    try:
        # ── Validate interval bounds ──
        if not cron_expression:
            if interval_hours < MIN_INTERVAL_HOURS:
                return json.dumps(
                    {
                        "error": f"Interval too short. Minimum is {MIN_INTERVAL_HOURS}h to prevent self-DoS.",
                        "min_interval_hours": MIN_INTERVAL_HOURS,
                    }
                )
            if interval_hours > MAX_INTERVAL_HOURS:
                return json.dumps(
                    {
                        "error": f"Interval too long. Maximum is {MAX_INTERVAL_HOURS}h (1 week).",
                        "max_interval_hours": MAX_INTERVAL_HOURS,
                    }
                )

        schedule_type = "cron" if cron_expression else "interval"

        with get_db() as db:
            if update_schedule_id:
                # ── UPDATE existing schedule ──
                now = datetime.now(timezone.utc).isoformat()
                db.execute(
                    """
                    UPDATE cycle_schedules SET
                        name = %s, schedule_type = %s, cron_expression = %s,
                        interval_hours = %s, collect = %s, "analyze" = %s,
                        market_hours_only = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    [
                        name,
                        schedule_type,
                        cron_expression,
                        interval_hours if not cron_expression else None,
                        collect,
                        analyze,
                        market_hours_only,
                        now,
                        update_schedule_id,
                    ],
                )

                try:
                    from app.services.cycle_scheduler import SchedulerService

                    SchedulerService.refresh_job(update_schedule_id)
                except Exception as e:
                    logger.warning("[SCHEDULE-TOOL] Refresh failed: %s", e)

                logger.info(
                    "[SCHEDULE-TOOL] Updated schedule %s: %s (%s)",
                    update_schedule_id,
                    name,
                    schedule_type,
                )
                return json.dumps(
                    {
                        "status": "updated",
                        "id": update_schedule_id,
                        "name": name,
                        "schedule_type": schedule_type,
                        "interval_hours": interval_hours
                        if not cron_expression
                        else None,
                        "cron_expression": cron_expression,
                    }
                )
            else:
                # ── CREATE new schedule ──
                active = _count_active_schedules()
                if active >= MAX_ACTIVE_SCHEDULES:
                    return json.dumps(
                        {
                            "error": f"Cannot create schedule: {active} active schedules already exist (max {MAX_ACTIVE_SCHEDULES}).",
                            "active_count": active,
                            "suggestion": "Update an existing schedule instead using update_schedule_id.",
                        }
                    )

                job_id = f"sch-bot-{uuid.uuid4().hex[:8]}"
                now = datetime.now(timezone.utc).isoformat()

                db.execute(
                    """
                    INSERT INTO cycle_schedules (
                        id, name, schedule_type, cron_expression, interval_hours,
                        collect, "analyze", trade, tickers, max_tickers,
                        market_hours_only, is_active, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        job_id,
                        name,
                        schedule_type,
                        cron_expression,
                        interval_hours if not cron_expression else None,
                        collect,
                        analyze,
                        None,  # trade = armed
                        "[]",  # watchlist
                        None,  # max_tickers
                        market_hours_only,
                        True,
                        now,
                        now,
                    ],
                )

                try:
                    from app.services.cycle_scheduler import SchedulerService

                    SchedulerService.refresh_job(job_id)
                except Exception as e:
                    logger.warning("[SCHEDULE-TOOL] Refresh failed: %s", e)

                logger.info(
                    "[SCHEDULE-TOOL] Created schedule %s: %s (%s, %.1fh)",
                    job_id,
                    name,
                    schedule_type,
                    interval_hours,
                )
                return json.dumps(
                    {
                        "status": "created",
                        "id": job_id,
                        "name": name,
                        "schedule_type": schedule_type,
                        "interval_hours": interval_hours
                        if not cron_expression
                        else None,
                        "cron_expression": cron_expression,
                        "active_count": active + 1,
                    }
                )

    except Exception as e:
        logger.error("[SCHEDULE-TOOL] create_or_update failed: %s", e)
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════
# TOOL: list_active_schedules
# ═══════════════════════════════════════════════════════════════════


@registry.register(
    name="list_active_schedules",
    description=(
        "List all active cycle schedules. Shows schedule name, type, interval, "
        "last run time, and next run time. Use this to inspect the bot's "
        "current scheduling state before creating or modifying schedules."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
    permission=PermissionLevel.READ_ONLY,
    tier=0,
    source="scheduler",
    tags=["schedule", "autonomy", "status"],
)
async def list_active_schedules() -> str:
    """List all active schedules with their status."""
    try:
        with get_db() as db:
            rows = db.execute(
                """
                SELECT id, name, schedule_type, cron_expression, interval_hours,
                       is_active, last_run_at, next_run_at, run_count, last_status
                FROM cycle_schedules
                ORDER BY is_active DESC, created_at DESC
                """
            ).fetchall()

        schedules = []
        for r in rows:
            schedules.append(
                {
                    "id": r[0],
                    "name": r[1],
                    "schedule_type": r[2],
                    "cron_expression": r[3],
                    "interval_hours": r[4],
                    "is_active": bool(r[5]),
                    "last_run_at": r[6].isoformat() if r[6] else None,
                    "next_run_at": r[7].isoformat() if r[7] else None,
                    "run_count": r[8] or 0,
                    "last_status": r[9],
                }
            )

        return json.dumps(
            {
                "total": len(schedules),
                "active": sum(1 for s in schedules if s["is_active"]),
                "schedules": schedules,
            }
        )

    except Exception as e:
        logger.error("[SCHEDULE-TOOL] list failed: %s", e)
        return json.dumps({"error": str(e), "schedules": []})

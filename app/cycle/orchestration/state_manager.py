"""Pipeline state persistence via PostgreSQL."""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

from app.db import connection

logger = logging.getLogger(__name__)


def _stringify_timestamp(value):
    if not value:
        return None
    if isinstance(value, str):
        if not value.endswith("Z") and "+" not in value:
            return value + "Z"
        return value
    if hasattr(value, "tzinfo") and value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat() if hasattr(value, "isoformat") else value


_OPERATIONAL_PHASES = {
    "created",
    "queued",
    "started",
    "collecting",
    "analyzing",
    "gated",
    "traded",
    "persisted",
    "evaluated",
    "closed",
    "done",
    "error",
    "stopped",
}


class PipelineStateDB:
    SINGLETON_ID = "current"

    @classmethod
    def get_state(cls, summary_only: bool = False) -> dict:
        """Read full cycle state from PostgreSQL."""
        try:
            with connection.get_db() as db:
                row = db.execute(
                    "SELECT * FROM pipeline_state WHERE singleton_id = %s",
                    [cls.SINGLETON_ID],
                ).fetchone()

                if row:
                    cols = [desc[0] for desc in db.description]
                    d = dict(zip(cols, row))

                    # Parse JSON cols
                    for jcol in ("tickers",):
                        try:
                            parsed = d.get(jcol)
                            if isinstance(parsed, str):
                                d[jcol] = json.loads(parsed)
                            elif parsed is None:
                                d[jcol] = []
                        except Exception:
                            d[jcol] = []

                    # Stringify timestamps for API compatibility
                    for tcol in ("started_at", "finished_at"):
                        if d.get(tcol):
                            d[tcol] = _stringify_timestamp(d[tcol])

                    d.pop("singleton_id", None)

                    if summary_only:
                        d["events"] = []
                        d["results"] = []
                        return d

                    # Fetch relational events mapped for UI
                    if d.get("cycle_id"):
                        ev_rows = db.execute(
                            "SELECT timestamp as ts, phase, step, detail, status, data_json, elapsed_ms "
                            "FROM pipeline_events WHERE cycle_id = %s ORDER BY timestamp ASC",
                            [d["cycle_id"]],
                        ).fetchall()
                        events = []
                        for erow in ev_rows:
                            ts = _stringify_timestamp(erow[0])
                            events.append(
                                {
                                    "ts": ts,
                                    "phase": erow[1],
                                    "step": erow[2],
                                    "detail": erow[3],
                                    "status": erow[4],
                                    "data": json.loads(erow[5]) if erow[5] else {},
                                    "elapsed_ms": erow[6] or 0,
                                }
                            )
                        d["events"] = events
                    else:
                        d["events"] = []

                    # Fetch relational results map
                    if d.get("cycle_id"):
                        ar_rows = db.execute(
                            "SELECT ticker, result_json FROM analysis_results WHERE cycle_id = %s",
                            [d["cycle_id"]],
                        ).fetchall()
                        results = []
                        for ar in ar_rows:
                            try:
                                res = json.loads(ar[1])
                                if "ticker" not in res:
                                    res["ticker"] = ar[0]
                                results.append(res)
                            except Exception:
                                pass
                        d["results"] = results
                    else:
                        d["results"] = []

                    return d
        except Exception as e:
            logger.error("[PipelineStateDB] Failed to read state: %s", e)

        return cls.default_state()

    @classmethod
    def save_state(cls, state: dict):
        """Write core state scalar values to PostgreSQL."""
        try:
            with connection.get_db() as db:
                tickers_str = json.dumps(state.get("tickers", []))

                started_at = state.get("started_at")
                finished_at = state.get("finished_at")

                db.execute(
                    """
                    INSERT INTO pipeline_state (
                        singleton_id, status, cycle_id, started_at, finished_at,
                        requested_pipeline_version, effective_pipeline_version,
                        benchmark_group, execution_mode, v2_stage,
                        tickers, progress, error, phase,
                        operational_phase, step_count, total_steps,
                        collect_flag, analyze_flag, trade_flag
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s::jsonb, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s
                    )
                ON CONFLICT (singleton_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    cycle_id = EXCLUDED.cycle_id,
                    started_at = EXCLUDED.started_at,
                    finished_at = EXCLUDED.finished_at,
                    requested_pipeline_version = EXCLUDED.requested_pipeline_version,
                    effective_pipeline_version = EXCLUDED.effective_pipeline_version,
                    benchmark_group = EXCLUDED.benchmark_group,
                    execution_mode = EXCLUDED.execution_mode,
                    v2_stage = EXCLUDED.v2_stage,
                    tickers = EXCLUDED.tickers,
                    progress = EXCLUDED.progress,
                    error = EXCLUDED.error,
                    phase = EXCLUDED.phase,
                    operational_phase = EXCLUDED.operational_phase,
                    step_count = EXCLUDED.step_count,
                    total_steps = EXCLUDED.total_steps,
                    collect_flag = EXCLUDED.collect_flag,
                    analyze_flag = EXCLUDED.analyze_flag,
                    trade_flag = EXCLUDED.trade_flag
                """,
                    [
                        cls.SINGLETON_ID,
                        state.get("status", "idle"),
                        state.get("cycle_id"),
                        started_at,
                        finished_at,
                        state.get("requested_pipeline_version", "v2"),
                        state.get("effective_pipeline_version", "v2"),
                        state.get("benchmark_group", "baseline"),
                        state.get("execution_mode", "production"),
                        state.get("v2_stage", 0),
                        tickers_str,
                        state.get("progress", ""),
                        state.get("error"),
                        state.get("phase", ""),
                        state.get("operational_phase", ""),
                        state.get("step_count", 0),
                        state.get("total_steps", 0),
                        state.get("collect_flag", True),
                        state.get("analyze_flag", True),
                        state.get("trade_flag", False),
                    ],
                )
        except Exception as e:
            logger.error("[PipelineStateDB] Failed to save DB core state: %s", e)

    @classmethod
    def append_event(cls, cycle_id: str, event: dict):
        cls.append_events(cycle_id, [event])

    @classmethod
    def append_events(cls, cycle_id: str, events: list[dict]):
        """Append multiple real-time events directly to PostgreSQL."""
        try:
            if not cycle_id or not events:
                return
            with connection.get_db() as db:
                import uuid

                rows = [
                    (
                        f"evt_{uuid.uuid4().hex[:8]}",
                        cycle_id,
                        e.get("ts"),
                        e.get("phase"),
                        e.get("step"),
                        e.get("detail"),
                        e.get("status", "ok"),
                        json.dumps(e.get("data", {})),
                        e.get("elapsed_ms", 0),
                    )
                    for e in events
                ]
                db.executemany(
                    """
                    INSERT INTO pipeline_events 
                    (id, cycle_id, timestamp, phase, step, detail, status, data_json, elapsed_ms) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    rows,
                )
        except Exception as e:
            logger.error("[PipelineStateDB] Failed to append SQL events: %s", e)

    @classmethod
    def get_cycle_events(cls, cycle_id: str | None) -> list[dict]:
        """Fetch persisted events for a cycle without loading the full state."""
        if not cycle_id:
            return []
        try:
            with connection.get_db() as db:
                rows = db.execute(
                    "SELECT timestamp as ts, phase, step, detail, status, data_json, elapsed_ms "
                    "FROM pipeline_events WHERE cycle_id = %s ORDER BY timestamp ASC",
                    [cycle_id],
                ).fetchall()
                events = []
                for row in rows:
                    ts = _stringify_timestamp(row[0])
                    events.append(
                        {
                            "ts": ts,
                            "phase": row[1],
                            "step": row[2],
                            "detail": row[3],
                            "status": row[4],
                            "data": json.loads(row[5]) if row[5] else {},
                            "elapsed_ms": row[6] or 0,
                        }
                    )
                return events
        except Exception as e:
            logger.error("[PipelineStateDB] Failed to fetch events: %s", e)
            return []

    @classmethod
    def log_execution_error(
        cls,
        cycle_id: str,
        phase: str,
        ticker: str,
        error_type: str,
        error_message: str,
        stack_trace: str,
    ):
        """Log a pipeline execution error to the database for post-cycle reporting."""
        try:
            with connection.get_db() as db:
                import uuid

                db.execute(
                    """
                    INSERT INTO execution_errors 
                    (id, cycle_id, phase, ticker, error_type, error_message, stack_trace) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        f"err_{uuid.uuid4().hex[:8]}",
                        cycle_id or "unknown",
                        phase or "unknown",
                        ticker or "system",
                        error_type,
                        error_message,
                        stack_trace,
                    ],
                )
        except Exception as e:
            logger.error("[PipelineStateDB] Failed to log execution error: %s", e)

    @classmethod
    def safe_log_execution_error(
        cls,
        cycle_id: str | None,
        phase: str | None,
        error_type: str,
        error: Exception | str,
        ticker: str = "system",
    ):
        """Safely log an execution error by handling string slicing and stack trace generation internally.
        Guaranteed not to raise an exception.
        """
        try:
            import traceback
            error_message = str(error)[:500]
            stack_trace = traceback.format_exc()[:2000]
            # If traceback has no active exception stack, use empty string
            if "NoneType: None" in stack_trace:
                stack_trace = ""
            cls.log_execution_error(
                cycle_id=cycle_id or "unknown",
                phase=phase or "unknown",
                ticker=ticker,
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
            )
        except Exception as e:
            logger.error("[PipelineStateDB] safe_log_execution_error failed: %s", e)

    @classmethod
    def default_state(cls) -> dict:
        return {
            "status": "idle",
            "cycle_id": None,
            "requested_pipeline_version": "v2",
            "effective_pipeline_version": "v2",
            "benchmark_group": "baseline",
            "execution_mode": "production",
            "v2_stage": 0,
            "started_at": None,
            "finished_at": None,
            "tickers": [],
            "progress": "",
            "results": [],
            "error": None,
            "events": [],
            "phase": "",
            "operational_phase": "",
            "step_count": 0,
            "total_steps": 0,
            "collect_flag": True,
            "analyze_flag": True,
            "trade_flag": False,
        }

    # ─── Cycle checkpoint methods (resume after crash) ───

    @classmethod
    def save_checkpoint(
        cls,
        cycle_id: str,
        completed_phases: list[str],
        completed_tickers: dict[str, list[str]],
        cycle_config: dict,
        original_started_at: str | None = None,
    ):
        """Persist a checkpoint so an interrupted cycle can be resumed."""
        try:
            with connection.get_db() as db:
                db.execute(
                    """
                    INSERT INTO cycle_resume_state (
                        cycle_id, status, completed_phases, completed_tickers,
                        cycle_config, checkpoint_ts, original_started_at
                    ) VALUES (%s, 'interrupted', %s::jsonb, %s::jsonb, %s::jsonb, CURRENT_TIMESTAMP, %s)
                    ON CONFLICT (cycle_id) DO UPDATE SET
                        completed_phases = EXCLUDED.completed_phases,
                        completed_tickers = EXCLUDED.completed_tickers,
                        cycle_config = EXCLUDED.cycle_config,
                        checkpoint_ts = CURRENT_TIMESTAMP
                    """,
                    [
                        cycle_id,
                        json.dumps(completed_phases),
                        json.dumps(completed_tickers),
                        json.dumps(cycle_config),
                        original_started_at,
                    ],
                )
                logger.info(
                    "[CHECKPOINT] Saved checkpoint for %s (phases: %s, tickers: %s)",
                    cycle_id,
                    completed_phases,
                    {k: len(v) for k, v in completed_tickers.items()},
                )
        except Exception as e:
            logger.error("[CHECKPOINT] Failed to save checkpoint: %s", e)
            raise

    @classmethod
    def get_checkpoint(cls, cycle_id: str | None = None) -> dict | None:
        """Retrieve the most recent checkpoint. If cycle_id is None, get any
        checkpoint with status='interrupted'."""
        try:
            with connection.get_db() as db:
                if cycle_id:
                    row = db.execute(
                        "SELECT cycle_id, status, completed_phases, completed_tickers, "
                        "cycle_config, checkpoint_ts, original_started_at "
                        "FROM cycle_resume_state WHERE cycle_id = %s AND status = 'interrupted'",
                        [cycle_id],
                    ).fetchone()
                else:
                    row = db.execute(
                        "SELECT cycle_id, status, completed_phases, completed_tickers, "
                        "cycle_config, checkpoint_ts, original_started_at "
                        "FROM cycle_resume_state WHERE status = 'interrupted' "
                        "ORDER BY checkpoint_ts DESC LIMIT 1"
                    ).fetchone()

                if not row:
                    return None

                checkpoint_ts = row[5]
                if checkpoint_ts:
                    checkpoint_ts = _stringify_timestamp(checkpoint_ts)

                original_started = row[6]
                if original_started:
                    original_started = _stringify_timestamp(original_started)

                return {
                    "cycle_id": row[0],
                    "status": row[1],
                    "completed_phases": json.loads(row[2])
                    if isinstance(row[2], str)
                    else (row[2] or []),
                    "completed_tickers": json.loads(row[3])
                    if isinstance(row[3], str)
                    else (row[3] or {}),
                    "cycle_config": json.loads(row[4])
                    if isinstance(row[4], str)
                    else (row[4] or {}),
                    "checkpoint_ts": checkpoint_ts,
                    "original_started_at": original_started,
                }
        except Exception as e:
            logger.error("[CHECKPOINT] Failed to read checkpoint: %s", e)
            return None

    @classmethod
    def clear_checkpoint(cls, cycle_id: str):
        """Delete a checkpoint after successful completion or user discard."""
        try:
            with connection.get_db() as db:
                db.execute(
                    "DELETE FROM cycle_resume_state WHERE cycle_id = %s", [cycle_id]
                )
                logger.info("[CHECKPOINT] Cleared checkpoint for %s", cycle_id)
        except Exception as e:
            logger.error("[CHECKPOINT] Failed to clear checkpoint: %s", e)

    @classmethod
    def expire_old_checkpoints(cls, max_age_hours: int = 6):
        """Mark checkpoints older than max_age_hours as expired so they
        aren't offered for resume."""
        try:
            from datetime import timedelta

            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            with connection.get_db() as db:
                db.execute(
                    "UPDATE cycle_resume_state SET status = 'expired' "
                    "WHERE status = 'interrupted' "
                    "AND checkpoint_ts < %s",
                    [cutoff],
                )
        except Exception as e:
            logger.error("[CHECKPOINT] Failed to expire old checkpoints: %s", e)


class PipelineStateMixin:
    _state = PipelineStateDB.default_state()
    _cycle_task = None
    _scout_task = None
    _consumer_task = None
    _checkpoint_task = None
    _macro_task = None
    _analysis_task = None
    _cycle_summary = {}

    _emit_events = []
    _emit_timer = None
    _emit_lock = threading.Lock()
    _cached_states = {}
    _cache_lock = threading.Lock()

    @classmethod
    def load_state(cls, summary_only: bool = False):
        """Load state from PostgreSQL to memory."""
        cls._state = PipelineStateDB.get_state(summary_only=summary_only)

    @classmethod
    def save_state(cls):
        """Save memory state to PostgreSQL."""
        PipelineStateDB.save_state(cls._state)

    @classmethod
    def get_current_state(cls, summary_only: bool = False) -> dict:
        now = time.time()
        with cls._cache_lock:
            cached = cls._cached_states.get(summary_only)
            if cached:
                cache_ts, cached_state = cached
                if now - cache_ts < 0.25:  # 250ms TTL
                    return cached_state

        state = PipelineStateDB.get_state(summary_only=summary_only)
        try:
            from app.services.vllm_client import llm
            from app.monitoring.llm_tracker import tracker

            active_requests = 0
            queued_requests = 0
            per_box = {}
            for ep in llm._endpoints.values():
                # Use client-side state combined with container metrics for real-time responsiveness
                ep_active = max(ep.active_count, ep.requests_running)
                ep_queued = max(ep.queue.qsize() if ep.queue else 0, ep.requests_waiting)
                active_requests += ep_active
                queued_requests += ep_queued
                per_box[ep.name] = {
                    "active": ep_active,
                    "queued": ep_queued,
                    "max_concurrent": ep.max_concurrent,
                }

            # Per-endpoint TPS from recent call history
            tps_by_ep = tracker.get_recent_tps_by_endpoint(60)
            for ep_name, tps_val in tps_by_ep.items():
                if ep_name in per_box:
                    per_box[ep_name]["tps"] = tps_val

            state["llm_stats"] = {
                "active_requests": active_requests,
                "queued_requests": queued_requests,
                "tokens_per_second": tracker.get_recent_tps(60),
                "total_calls": tracker.total_calls,
                "total_tokens": tracker.total_tokens,
                "per_box": per_box,
            }
        except Exception as e:
            logger.debug(f"[get_current_state] Could not attach llm_stats: {e}")

        # Attach checkpoint info when cycle is interrupted
        if state.get("status") == "interrupted":
            try:
                checkpoint = PipelineStateDB.get_checkpoint(state.get("cycle_id"))
                if checkpoint:
                    state["checkpoint"] = {
                        "cycle_id": checkpoint["cycle_id"],
                        "completed_phases": checkpoint["completed_phases"],
                        "completed_tickers": checkpoint["completed_tickers"],
                        "checkpoint_ts": checkpoint["checkpoint_ts"],
                        "original_started_at": checkpoint["original_started_at"],
                    }
                else:
                    state["checkpoint"] = None
            except Exception:
                state["checkpoint"] = None

        # Dynamically recalculate total_steps based on the actual number of tickers
        tickers = state.get("tickers") or []
        n = len(tickers)
        n_tickers = max(n, 5) if n == 0 else n
        c_steps = (6 * n_tickers + 9) if state.get("collect_flag", True) else 0
        a_steps = (9 * n_tickers) if state.get("analyze_flag", True) else 0
        t_steps = n_tickers if state.get("trade_flag", True) else 0

        # Add 10% buffer to prevent progress bar from hitting 100% too early due to unexpected events
        dynamic_total = int((c_steps + a_steps + t_steps) * 1.1)

        # Ensure total_steps is never smaller than step_count (to prevent >100% progress)
        step_count = state.get("step_count", 0)
        state["total_steps"] = max(dynamic_total, step_count + 1)

        with cls._cache_lock:
            cls._cached_states[summary_only] = (now, state)

        return state

    @classmethod
    def reset_on_boot(cls):
        """Called once on server startup to handle zombie cycles.

        If a checkpoint exists for the interrupted cycle, set status to
        'interrupted' so the frontend can offer Resume / Start Fresh.
        Otherwise, force-reset to idle as before.

        Also checks for orphaned checkpoints when status is 'stopped'
        (e.g. server was shut down gracefully mid-cycle but the shutdown
        handler set 'stopped' before checkpoint logic existed).

        # BUG: partial-write not handled
        # If the DB was mid-write when the server crashed (e.g. partial cycle row in DB,
        # status='running' rows, orphaned ticker_analysis rows linked to a dead cycle,
        # or file locks), this method does NOT currently clean them up. This can lead to
        # dirty database state.
        """

        cls._cycle_task = None
        if cls._checkpoint_task:
            cls._checkpoint_task.cancel()
            cls._checkpoint_task = None

        # Expire any checkpoints older than 6 hours
        PipelineStateDB.expire_old_checkpoints(max_age_hours=6)

        cls.load_state()
        prev_status = cls._state.get("status", "idle")

        # Statuses that might have a resumable checkpoint:
        #   - Active phases (collecting, analyzing, trading, starting, paused)
        #     → zombie cycle, process was killed
        #   - "stopped" → graceful shutdown set this, but checkpoint may exist
        #   - "interrupted" → already flagged from a prior boot, re-validate
        _terminal_no_checkpoint = ("idle", "done", "error", "stopped")

        if prev_status not in _terminal_no_checkpoint:
            zombie_cycle_id = cls._state.get("cycle_id")
            checkpoint = PipelineStateDB.get_checkpoint(zombie_cycle_id)

            if checkpoint:
                logger.warning(
                    "[CYCLE] Interrupted cycle detected (was '%s') — checkpoint found, "
                    "setting to 'interrupted' for possible resume (cycle: %s)",
                    prev_status,
                    zombie_cycle_id,
                )
                cls._state["status"] = "interrupted"
                cls._state["phase"] = "interrupted"
                cls._state["progress"] = (
                    f"Cycle {zombie_cycle_id} was interrupted. "
                    f"Completed phases: {', '.join(checkpoint['completed_phases']) or 'none'}. "
                    "Choose to resume or start fresh."
                )
                cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
                cls.save_state()
                return

            # No checkpoint — force-reset to idle
            if prev_status not in _terminal_no_checkpoint:
                zombie_cycle_id = cls._state.get("cycle_id")
                logger.warning(
                    "[CYCLE] Stale cycle detected (was '%s') — no checkpoint, "
                    "force-resetting to idle on boot",
                    prev_status,
                )
                if zombie_cycle_id:
                    try:
                        with connection.get_db() as db:
                            db.execute("DELETE FROM pipeline_events WHERE cycle_id = %s", [zombie_cycle_id])
                            db.execute("DELETE FROM analysis_results WHERE cycle_id = %s", [zombie_cycle_id])
                            db.execute("DELETE FROM debate_history WHERE cycle_id = %s", [zombie_cycle_id])
                            logger.info("[CYCLE] Cleaned up orphaned data for dead cycle %s", zombie_cycle_id)
                    except Exception as e:
                        logger.error("[CYCLE] Failed to clean up dead cycle data: %s", e)
                        
                cls._state = PipelineStateDB.default_state()
                cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
                cls.save_state()
                return

        # Also catch any truly anomalous status that slipped through
        if cls._state["status"] not in (
            "idle",
            "done",
            "error",
            "stopped",
            "interrupted",
        ):
            cls._state["status"] = "idle"
            cls._state["phase"] = ""
            cls._state["progress"] = ""
            cls._state["operational_phase"] = ""
            cls.save_state()

    @classmethod
    def emit(
        cls,
        phase: str,
        step: str,
        detail: str,
        status: str = "ok",
        data: dict | None = None,
        elapsed_ms: int = 0,
    ):
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "step": step,
            "detail": detail,
            "status": status,
            "data": data or {},
            "elapsed_ms": elapsed_ms,
        }

        cls._state.setdefault("step_count", 0)
        if status in ("ok", "error", "skipped"):
            cls._state["step_count"] += 1

        cls._state["phase"] = phase
        cls._state["progress"] = f"[{phase}] {step}: {detail}"
        if phase in _OPERATIONAL_PHASES:
            cls._state["operational_phase"] = phase

        cid = cls._state.get("cycle_id") or "no-id"
        logger.info("[CYCLE %s] %s/%s: %s (%s)", cid, phase, step, detail, status)

        state_copy = dict(cls._state)

        with cls._emit_lock:
            cls._emit_events.append(event)
            if cls._emit_timer is None:

                def flush():
                    with cls._emit_lock:
                        events_to_flush = list(cls._emit_events)
                        cls._emit_events.clear()
                        cls._emit_timer = None

                    if not events_to_flush:
                        return

                    try:
                        PipelineStateDB.save_state(state_copy)
                        PipelineStateDB.append_events(cid, events_to_flush)
                    except Exception as e:
                        logger.error("[PipelineStateMixin] emit flush failed: %s", e)

                cls._emit_timer = threading.Timer(0.1, flush)
                cls._emit_timer.start()

    @classmethod
    def flush_events(cls):
        """Synchronously flush any pending events to the database."""
        with cls._emit_lock:
            if cls._emit_timer:
                cls._emit_timer.cancel()
                cls._emit_timer = None
            events_to_flush = list(cls._emit_events)
            cls._emit_events.clear()

        if not events_to_flush:
            return

        cid = cls._state.get("cycle_id") or "no-id"
        try:
            PipelineStateDB.save_state(dict(cls._state))
            PipelineStateDB.append_events(cid, events_to_flush)
        except Exception as e:
            logger.error("[PipelineStateMixin] flush_events failed: %s", e)


import time
import asyncio
import logging
import uuid
import json
from datetime import datetime, timezone
from app.cycle.orchestration.state_manager import PipelineStateDB
from app.cycle.core import PipelineContext
from app.config import settings
from app.db.connection import get_db
from app.cognition.orchestration import resolve_cycle_runtime

logger = logging.getLogger(__name__)


class LifecycleControllerMixin:
    _action_lock = None

    @classmethod
    def _get_lock(cls):
        if cls._action_lock is None:
            cls._action_lock = asyncio.Lock()
        return cls._action_lock

    @classmethod
    async def start_cycle(
        cls,
        tickers: list[str],
        collect: bool = True,
        analyze: bool = True,
        trade: bool = True,  # Paper trading ALWAYS armed as per user directive.
        trigger_type: str = "manual",
        schedule_id: str | None = None,
        max_tickers: int | None = None,
        pipeline_version: str | None = None,
        benchmark_group: str | None = None,
    ):
        async with cls._get_lock():
            # Check memory state first to avoid race conditions with DB read lag
            current_status = cls._state.get("status", "idle")
            if current_status not in (
                "idle",
                "done",
                "error",
                "stopped",
                "interrupted",
            ):
                raise ValueError(f"Cycle already running: {current_status}")

            cls.load_state()
            if cls._state["status"] not in (
                "idle",
                "done",
                "error",
                "stopped",
                "interrupted",
            ):
                raise ValueError(f"Cycle already running: {cls._state['status']}")

            cycle_id = f"cycle-{int(time.time())}"

            cls._state.update(
                {
                    "status": "starting",
                    "cycle_id": cycle_id,
                    "progress": f"Initializing cycle {cycle_id}...",
                    "error": None,
                    "phase": "starting",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "finished_at": None,
                }
            )
            cls.save_state()

        logger.info(
            "[CYCLE] Start requested. Initializing in background for instant UI response."
        )

        loop = asyncio.get_running_loop()
        loop.create_task(
            cls._background_start_cycle(
                tickers=tickers,
                collect=collect,
                analyze=analyze,
                trade=trade,
                trigger_type=trigger_type,
                schedule_id=schedule_id,
                max_tickers=max_tickers,
                pipeline_version=pipeline_version,
                benchmark_group=benchmark_group,
                cycle_id=cycle_id,
            )
        )

        return {
            "status": "starting",
            "cycle_id": cycle_id,
            "message": "Cycle initialization started in background",
        }

    @classmethod
    async def _background_start_cycle(
        cls,
        tickers: list[str],
        collect: bool,
        analyze: bool,
        trade: bool,
        trigger_type: str,
        schedule_id: str | None,
        max_tickers: int | None,
        pipeline_version: str | None,
        benchmark_group: str | None,
        cycle_id: str,
    ):
        try:
            # If starting fresh from an interrupted state, clear the old checkpoint
            if cls._state.get("status") == "interrupted":
                old_cycle_id = cls._state.get("cycle_id")
                if old_cycle_id and old_cycle_id != cycle_id:
                    PipelineStateDB.clear_checkpoint(old_cycle_id)
                    logger.info(
                        "[CYCLE] Cleared old checkpoint for %s (user chose Start Fresh)",
                        old_cycle_id,
                    )

            trade = True  # Enforced: 100% armed, always paper trading.
            route = resolve_cycle_runtime(
                requested_version=pipeline_version,
                benchmark_group=benchmark_group,
            )

            # Compute hard total cap: max_tickers from UI > MAX_CYCLE_TICKERS > MAX_ANALYSIS_TICKERS
            if max_tickers is not None:
                cap = max_tickers
            elif settings.MAX_CYCLE_TICKERS > 0:
                cap = settings.MAX_CYCLE_TICKERS
            else:
                cap = settings.MAX_ANALYSIS_TICKERS  # fallback: 30

            logger.info("[CYCLE] Hard cap on TOTAL tickers: %d", cap)

            from app.pipeline.ticker_selector import TickerSelector



            selection = TickerSelector.select_tickers_for_cycle_v2(tickers, cap)
            tickers = selection.all_tickers

            if not tickers:
                logger.info(
                    "[CYCLE] No tickers available — running in discovery-only mode"
                )

            n_pos = len(selection.position_tickers)
            n_extra = len(selection.non_position_tickers)
            if n_pos:
                logger.info(
                    "[CYCLE] Portfolio positions (count against cap): %d — %s",
                    n_pos,
                    ", ".join(selection.position_tickers),
                )
            logger.info(
                "[CYCLE] Non-position tickers: %d (remaining slots from cap %d - %d positions)",
                n_extra,
                cap,
                n_pos,
            )

            n_tickers = max(len(tickers), 5)
            collect_steps = (6 * n_tickers + 9) if collect else 0
            analyze_steps = (9 * n_tickers) if analyze else 0
            trade_steps = n_tickers if trade else 0
            total = collect_steps + analyze_steps + trade_steps

            from app.cycle.orchestration.cycle_control import cycle_control

            cycle_control.reset()

            try:
                from app.services.memory.working_memory import working_memory
                from app.services.session_profile import profile_memory

                working_memory.clear()

                # Load persistent last cycle context into working memory
                last_context = profile_memory.get_last_trade_context()
                if last_context:
                    working_memory.add_event(
                        content=str(last_context),
                        source="last_trade_context",
                        ticker="global",
                    )
                    logger.info("[MEMORY] Loaded last trade context from disk profile")

                logger.debug("[MEMORY] Cleared working memory for new cycle")
            except ImportError:
                pass

            cls._state.update(
                {
                    "requested_pipeline_version": route["requested_version"],
                    "effective_pipeline_version": route["effective_version"],
                    "benchmark_group": route["benchmark_group"],
                    "execution_mode": route["execution_mode"],
                    "v2_stage": route["v2_stage"],
                    "tickers": tickers,
                    "position_tickers": selection.position_tickers,
                    "non_position_tickers": selection.non_position_tickers,
                    "progress": f"Starting cycle {cycle_id} for {len(tickers)} tickers ({n_pos} portfolio + {n_extra} new)",
                    "step_count": 0,
                    "total_steps": total,
                    "collect_flag": collect,
                    "analyze_flag": analyze,
                    "trade_flag": trade,
                }
            )
            cls.save_state()

            logger.info("=" * 70)
            logger.info(
                "  CYCLE %s STARTED — %d tickers (%d portfolio + %d new)",
                cycle_id,
                len(tickers),
                n_pos,
                n_extra,
            )
            logger.info("=" * 70)

            ticker_msg = (
                f"{len(tickers)} tickers ({n_pos} portfolio + {n_extra} new)"
                if tickers
                else "Discovery-only mode"
            )
            cls.emit(
                "starting",
                "init",
                f"Cycle started: {ticker_msg}",
                data={
                    "tickers": tickers,
                    "position_tickers": selection.position_tickers,
                    "non_position_tickers": selection.non_position_tickers,
                    "collect": collect,
                    "analyze": analyze,
                    "trade": trade,
                    "requested_version": route["requested_version"],
                    "effective_version": route["effective_version"],
                    "benchmark_group": route["benchmark_group"],
                    "execution_mode": route["execution_mode"],
                    "v2_stage": route["v2_stage"],
                },
            )

            ctx = PipelineContext(
                tickers=tickers,
                collect=collect,
                analyze=analyze,
                trade=trade,
                cycle_id=cycle_id,
                trigger_type=trigger_type,
                schedule_id=schedule_id,
                max_tickers=cap,
            )

            cls._cycle_task = asyncio.create_task(cls._run_cycle(ctx))

            cls._checkpoint_task = asyncio.create_task(
                cls._checkpoint_heartbeat(cycle_id)
            )

        except Exception as e:
            logger.error("[CYCLE] Failed to initialize cycle in background: %s", e)
            cls._state.update(
                {
                    "status": "error",
                    "progress": f"Failed to initialize cycle: {e}",
                    "error": str(e),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            cls.save_state()
            cls.emit(
                "error",
                "init_error",
                f"Failed to initialize cycle: {e}",
                status="error",
            )

    @classmethod
    async def stop_cycle(cls) -> dict:
        from app.cycle.orchestration.cycle_control import cycle_control

        cycle_control.stop()

        for name, task in [
            ("scout", getattr(cls, "_scout_task", None)),
            ("consumer", getattr(cls, "_consumer_task", None)),
            ("checkpoint", getattr(cls, "_checkpoint_task", None)),
        ]:
            if task and not task.done():
                logger.info("[CYCLE] Cancelling %s task...", name)
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
        cls._scout_task = None
        cls._consumer_task = None
        cls._checkpoint_task = None

        cycle_task = getattr(cls, "_cycle_task", None)
        if cycle_task is None or cycle_task.done():
            if cls._state["status"] in ("idle", "done", "error", "stopped"):
                return {"status": "already_idle", "message": "No cycle running"}
            # Do NOT return early. We might still need to create a synthetic checkpoint
            # for a cycle that was running when the server crashed/restarted.
            cls._cycle_task = None
        else:
            cycle_task.cancel()
            try:
                await asyncio.wait_for(cycle_task, timeout=15.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            cls._cycle_task = None

        # Check for checkpoint — if one exists, mark as interrupted (resumable)
        cycle_id = cls._state.get("cycle_id")
        checkpoint = PipelineStateDB.get_checkpoint(cycle_id) if cycle_id else None

        if not checkpoint and cycle_id:
            # Force create a synthetic checkpoint so the user can resume
            phase = cls._state.get("operational_phase", "collecting")
            completed_phases = []
            if phase == "analyzing":
                completed_phases.append("collecting")
            elif phase == "trading":
                completed_phases.extend(["collecting", "analyzing"])

            logger.info(
                "[CYCLE] Creating synthetic checkpoint for interrupted cycle %s",
                cycle_id,
            )
            try:
                PipelineStateDB.save_checkpoint(
                    cycle_id=cycle_id,
                    completed_phases=completed_phases,
                    completed_tickers={},
                    cycle_config={
                        "tickers": cls._state.get("tickers", []),
                        "collect_flag": cls._state.get("collect_flag", True),
                        "analyze_flag": cls._state.get("analyze_flag", True),
                        "trade_flag": cls._state.get("trade_flag", True),
                        "macro_memo": cls._state.get("macro_memo", ""),
                    },
                    original_started_at=cls._state.get("started_at"),
                )
                checkpoint = PipelineStateDB.get_checkpoint(cycle_id)
            except Exception as e:
                logger.error("[CYCLE] Failed to save synthetic checkpoint: %s", e)
                checkpoint = None

        if checkpoint:
            cls._state["status"] = "interrupted"
            cls._state["phase"] = "interrupted"
            cls._state["progress"] = (
                f"Cycle {cycle_id} stopped. "
                f"Completed phases: {', '.join(checkpoint['completed_phases']) or 'none'}. "
                "Resume or start fresh on next run."
            )
            cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
            cls.save_state()
            cls.emit(
                "interrupted",
                "user_stop",
                "Cycle stopped — checkpoint available for resume",
                status="ok",
            )
            return {
                "status": "interrupted",
                "message": "Cycle stopped. Checkpoint available for resume.",
            }

        cls._state["status"] = "stopped"
        cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
        cls.save_state()
        cls.emit("stopped", "user_stop", "Cycle stopped by user", status="ok")
        return {
            "status": "stopped",
            "message": "Cycle cancelled (Checkpoint failed to save)",
        }

    @classmethod
    def request_stop(cls) -> dict:
        """Non-blocking stop: signal cancellation immediately, clean up in background.

        This returns in <50ms so the frontend gets instant feedback.
        The heavy task cancellation (up to 7s of awaits) runs in a
        background asyncio task via the existing stop_cycle() method.
        """
        from app.cycle.orchestration.cycle_control import cycle_control

        prev_status = cls._state.get("status", "idle")
        if prev_status in ("idle", "done", "error", "stopped", "interrupted"):
            return {"status": "already_idle", "message": "No cycle running"}

        # 1. Signal the pipeline to stop (immediate flag flip)
        cycle_control.stop()

        # 2. Set state to 'stopping' so UI reflects it instantly
        cls._state["status"] = "stopping"
        cls._state["progress"] = "Stopping cycle..."
        cls.save_state()

        logger.info(
            "[CYCLE] Stop requested (non-blocking). "
            "Previous status: %s. Background cleanup scheduled.",
            prev_status,
        )

        # 3. Schedule the heavy cleanup as a background task
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(cls._background_stop_cleanup())
        except RuntimeError:
            # If no event loop, fall through — next status poll will
            # detect the 'stopping' state and the cycle task's own
            # cancellation handler will clean up.
            logger.warning(
                "[CYCLE] No event loop for background cleanup. "
                "Forcing immediate transition to interrupted."
            )
            cls._state["status"] = "interrupted"
            cls.save_state()

        return {
            "status": "stopping",
            "message": "Stop signal sent. Cycle is shutting down.",
        }

    @classmethod
    async def _background_stop_cleanup(cls):
        """Background task that performs the heavy stop cleanup."""
        try:
            await cls.stop_cycle()
        except Exception as e:
            logger.error("[CYCLE] Background stop cleanup failed: %s", e)
        finally:
            if cls._state.get("status") == "stopping":
                logger.warning("[CYCLE] Failsafe: Forcing status from 'stopping' to 'interrupted'")
                cls._state["status"] = "interrupted"
                cls.save_state()

    @classmethod
    def pause_cycle(cls):
        from app.cycle.orchestration.cycle_control import cycle_control

        if cls._state["status"] not in (
            "collecting",
            "analyzing",
            "trading",
            "starting",
        ):
            raise ValueError("No active cycle to pause")
        cycle_control.pause()
        cls._state["status"] = "paused"
        cls.save_state()
        cls.emit("paused", "user_pause", "Cycle paused by user", status="ok")

    @classmethod
    async def resume_cycle(cls):
        from app.cycle.orchestration.cycle_control import cycle_control

        if cls._state["status"] != "paused":
            raise ValueError("Cycle is not paused")

        resume_phase = cls._state.get("operational_phase") or "collecting"
        if resume_phase not in ("collecting", "analyzing", "trading"):
            resume_phase = "collecting"

        cls._state["status"] = resume_phase
        cls.save_state()
        cls.emit("resumed", "user_resume", "Cycle resumed by user", status="ok")

        cycle_task = getattr(cls, "_cycle_task", None)
        if cycle_task is None or cycle_task.done():
            # The original async task is dead (crashed or was cancelled).
            # Instead of blindly restarting from scratch (which causes
            # collisions and data loss), mark as interrupted and delegate
            # to resume_interrupted_cycle() which has full checkpoint
            # recovery logic.
            cycle_id = cls._state.get("cycle_id")
            logger.warning(
                "[CYCLE] resume_cycle() found dead task — delegating to "
                "checkpoint-based resume for cycle %s",
                cycle_id,
            )
            # Force a checkpoint save so resume_interrupted_cycle has data
            cls.force_save_checkpoint()
            cls._state["status"] = "interrupted"
            cls.save_state()
            try:
                await cls.resume_interrupted_cycle()
            except ValueError as e:
                # If no checkpoint exists, fall back to a fresh cycle start
                logger.warning(
                    "[CYCLE] Checkpoint resume failed (%s) — falling back to idle",
                    e,
                )
                cls._state["status"] = "idle"
                cls.save_state()
                raise ValueError(
                    "Cycle task crashed and no checkpoint available. "
                    "Please start a fresh cycle."
                )
            return

        cycle_control.resume()

    @classmethod
    async def cancel_cycle_shutdown(cls):
        """Graceful shutdown handler — preserves checkpoints for resume.

        If a checkpoint exists for the running cycle, status is set to
        'interrupted' so that reset_on_boot() on the next startup will
        offer the user a Resume / Start Fresh choice.
        """
        # If cycle already completed successfully, skip all checkpoint logic.
        # This prevents uvicorn --reload from overriding "done" to "stopped".
        current_status = cls._state.get("status", "idle")
        if current_status in ("idle", "done", "error"):
            logger.info(
                "[SHUTDOWN] Cycle already in terminal state '%s' — no checkpoint preservation needed",
                current_status,
            )
            # Still cancel background tasks to avoid orphaned coroutines
            for name, task in [
                ("scout", getattr(cls, "_scout_task", None)),
                ("consumer", getattr(cls, "_consumer_task", None)),
                ("checkpoint", getattr(cls, "_checkpoint_task", None)),
            ]:
                if task and not task.done():
                    task.cancel()
            cls._scout_task = None
            cls._consumer_task = None
            cls._checkpoint_task = None
            cls._cycle_task = None
            return

        for name, task in [
            ("scout", getattr(cls, "_scout_task", None)),
            ("consumer", getattr(cls, "_consumer_task", None)),
            ("checkpoint", getattr(cls, "_checkpoint_task", None)),
        ]:
            if task and not task.done():
                task.cancel()
        cls._scout_task = None
        cls._consumer_task = None
        cls._checkpoint_task = None

        cycle_task = getattr(cls, "_cycle_task", None)
        if cycle_task and not cycle_task.done():
            cycle_task.cancel()
            try:
                await asyncio.wait_for(cycle_task, timeout=15.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            cls._cycle_task = None

            # Preserve checkpoint for resume on next boot
            cycle_id = cls._state.get("cycle_id")
            checkpoint = PipelineStateDB.get_checkpoint(cycle_id) if cycle_id else None
            if checkpoint:
                logger.info(
                    "[SHUTDOWN] Checkpoint found for cycle %s — marking as interrupted for resume",
                    cycle_id,
                )
                cls._state["status"] = "interrupted"
                cls._state["phase"] = "interrupted"
                cls._state["progress"] = (
                    f"Cycle {cycle_id} interrupted by server shutdown. "
                    f"Completed phases: {', '.join(checkpoint['completed_phases']) or 'none'}."
                )
            else:
                cls._state["status"] = "stopped"

            cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
            cls.save_state()
        else:
            if cls._state["status"] not in (
                "idle",
                "done",
                "error",
                "stopped",
                "interrupted",
            ):
                logger.info(
                    "[SHUTDOWN] Cycle task finished but state is '%s' — forcing checkpoint",
                    cls._state["status"],
                )
                cls.force_save_checkpoint()
                cycle_id = cls._state.get("cycle_id")
                checkpoint = (
                    PipelineStateDB.get_checkpoint(cycle_id) if cycle_id else None
                )
                if checkpoint:
                    cls._state["status"] = "interrupted"
                    cls._state["phase"] = "interrupted"
                    cls._state["progress"] = (
                        f"Cycle {cycle_id} interrupted by server shutdown (task done but DB pending). "
                        f"Completed phases: {', '.join(checkpoint['completed_phases']) or 'none'}."
                    )
                else:
                    cls._state["status"] = "stopped"
                cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
                cls.save_state()

    @classmethod
    async def _checkpoint_heartbeat(cls, cycle_id: str):
        """Background task that saves a checkpoint every 30 seconds while cycle is running."""
        try:
            while True:
                await asyncio.sleep(30)
                # Only save if we are actually starting, collecting, analyzing, or trading
                status = cls._state.get("status")
                if status in (
                    "starting",
                    "collecting",
                    "analyzing",
                    "trading",
                    "paused",
                ):
                    try:
                        cls.force_save_checkpoint()
                    except Exception as e:
                        logger.warning("[CHECKPOINT] Heartbeat save skipped: %s", e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("[CHECKPOINT] Heartbeat crashed: %s", e)

    @classmethod
    async def resume_interrupted_cycle(cls) -> dict:
        """Resume an interrupted cycle from its last checkpoint.

        This method returns instantly. The heavy initialization runs in background.
        """
        cls.load_state()
        if cls._state.get("status") != "interrupted":
            raise ValueError(
                f"No interrupted cycle to resume (status: {cls._state.get('status')})"
            )

        cycle_id = cls._state.get("cycle_id")

        cls._state.update(
            {
                "status": "starting",
                "phase": "starting",
                "progress": f"Resuming cycle {cycle_id} in background...",
                "error": None,
                "finished_at": None,
            }
        )
        cls.save_state()

        logger.info(
            "[RESUME] Resume requested for %s. Initializing in background.", cycle_id
        )

        loop = asyncio.get_running_loop()
        loop.create_task(cls._background_resume_cycle(cycle_id))

        return {
            "status": "starting",
            "cycle_id": cycle_id,
            "message": "Cycle resume initialized in background",
        }

    @classmethod
    async def _background_resume_cycle(cls, cycle_id: str):
        try:
            checkpoint = await asyncio.to_thread(
                PipelineStateDB.get_checkpoint, cycle_id
            )
            if not checkpoint:
                logger.error("[RESUME_FAIL] Missing checkpoint for cycle %s", cycle_id)
                raise ValueError(f"No checkpoint found for cycle {cycle_id}")

            config = checkpoint["cycle_config"]
            completed_phases = set(checkpoint["completed_phases"])

            logger.info(
                "[RESUME] Parsed checkpoint for %s. Completed phases: %s",
                cycle_id,
                list(completed_phases),
            )

            if "analyzing" in completed_phases:
                resume_from = "trading"
            elif "collecting" in completed_phases:
                resume_from = "analyzing"
            else:
                resume_from = "collecting"

            logger.info("[RESUME] Determined resume_from phase: %s", resume_from)

            tickers = config.get("tickers", cls._state.get("tickers", []))
            collect = config.get("collect_flag", True)
            analyze = config.get("analyze_flag", True)
            trade = True  # Always armed

            def fetch_already_analyzed():
                with get_db() as db:
                    rows = db.execute(
                        "SELECT DISTINCT ticker FROM analysis_results WHERE cycle_id = %s",
                        [cycle_id],
                    ).fetchall()
                return [r[0] for r in rows]

            already_analyzed = await asyncio.to_thread(fetch_already_analyzed)
            if already_analyzed:
                logger.info(
                    "[RESUME] Found %d already-analyzed tickers in DB",
                    len(already_analyzed),
                )

            def fetch_existing_results():
                results = []
                if resume_from in ("analyzing", "trading"):
                    with get_db() as db:
                        ar_rows = db.execute(
                            "SELECT ticker, result_json FROM analysis_results WHERE cycle_id = %s",
                            [cycle_id],
                        ).fetchall()
                    for ar in ar_rows:
                        try:
                            res = json.loads(ar[1])
                            if "ticker" not in res:
                                res["ticker"] = ar[0]
                            results.append(res)
                        except Exception:
                            pass
                return results

            existing_results = await asyncio.to_thread(fetch_existing_results)
            if existing_results:
                logger.info(
                    "[RESUME] Loaded %d existing analysis results from DB",
                    len(existing_results),
                )

            from app.cycle.orchestration.cycle_control import cycle_control

            cycle_control.reset()

            cls._state.update(
                {
                    "progress": f"Resuming cycle {cycle_id} from {resume_from} phase",
                }
            )
            cls.save_state()

            cls.emit(
                "starting",
                "resume",
                f"♻️ Resuming interrupted cycle {cycle_id} from '{resume_from}' phase "
                f"({len(already_analyzed)} tickers already analyzed, "
                f"{len(tickers) - len(already_analyzed)} remaining)",
                status="ok",
                data={
                    "resume_from": resume_from,
                    "already_analyzed": already_analyzed,
                    "remaining": [t for t in tickers if t not in already_analyzed],
                },
            )

            logger.info("=" * 70)
            logger.info(
                "  CYCLE %s RESUMED from '%s' — %d tickers (%d already done)",
                cycle_id,
                resume_from,
                len(tickers),
                len(already_analyzed),
            )
            logger.info("=" * 70)

            ctx = PipelineContext(
                tickers=tickers,
                collect=collect,
                analyze=analyze,
                trade=trade,
                cycle_id=cycle_id,
                trigger_type="resume",
                resume_from=resume_from,
                already_analyzed=already_analyzed,
                existing_results=existing_results,
                macro_memo=config.get("macro_memo", ""),
            )

            cls._cycle_task = asyncio.create_task(cls._run_cycle(ctx))

            cls._checkpoint_task = asyncio.create_task(
                cls._checkpoint_heartbeat(cycle_id)
            )

        except Exception as e:
            logger.error("[RESUME] Failed to resume cycle in background: %s", e)
            cls._state.update(
                {
                    "status": "error",
                    "progress": f"Failed to resume cycle: {e}",
                    "error": str(e),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            cls.save_state()
            cls.emit(
                "error", "resume_error", f"Failed to resume cycle: {e}", status="error"
            )

    @classmethod
    def discard_checkpoint(cls) -> dict:
        """Discard the checkpoint for an interrupted cycle and reset to idle."""
        cls.load_state()
        cycle_id = cls._state.get("cycle_id")

        if cycle_id:
            PipelineStateDB.clear_checkpoint(cycle_id)
            logger.info("[CHECKPOINT] User discarded checkpoint for %s", cycle_id)

        cls._state = PipelineStateDB.default_state()
        cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
        cls.save_state()
        return {"status": "idle", "message": f"Checkpoint discarded for {cycle_id}"}

    @classmethod
    def force_save_checkpoint(cls):
        """Manually or periodically trigger a checkpoint save using the current state."""
        cycle_id = cls._state.get("cycle_id")
        if not cycle_id:
            return

        phase = cls._state.get("operational_phase", "")
        completed_phases = []
        if phase == "analyzing":
            completed_phases.append("collecting")
        elif phase == "trading":
            completed_phases.extend(["collecting", "analyzing"])

        cycle_config = {
            "tickers": cls._state.get("tickers", []),
            "collect_flag": cls._state.get("collect_flag", True),
            "analyze_flag": cls._state.get("analyze_flag", True),
            "trade_flag": cls._state.get("trade_flag", True),
            "macro_memo": cls._state.get("macro_memo", ""),
        }

        completed_tickers = {}
        try:
            with get_db() as db:
                rows = db.execute(
                    "SELECT DISTINCT ticker FROM analysis_results WHERE cycle_id = %s",
                    [cycle_id],
                ).fetchall()
                if rows:
                    completed_tickers["analyzing"] = [r[0] for r in rows]
        except Exception as e:
            logger.warning("[CHECKPOINT] Failed to query completed tickers: %s", e)

        PipelineStateDB.save_checkpoint(
            cycle_id=cycle_id,
            completed_phases=completed_phases,
            completed_tickers=completed_tickers,
            cycle_config=cycle_config,
            original_started_at=cls._state.get("started_at"),
        )
        logger.debug(
            "[CHECKPOINT] Time-based or manual checkpoint saved for cycle %s", cycle_id
        )

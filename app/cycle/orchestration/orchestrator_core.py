import logging
import time
from datetime import datetime, timezone
import traceback
import asyncio
from typing import Any

from app.config import settings
from app.cycle.orchestration.state_manager import PipelineStateDB
from app.cycle.orchestration.cycle_auditor import CycleAuditor

from app.cycle.phases.phase1_health import run_phase1_health
from app.cycle.phases.phase2_collection import run_phase2_collection
from app.cycle.phases.phase3_macro import run_phase3_macro
from app.cycle.phases.phase4_analysis import run_phase4_analysis
from app.cycle.phases.phase5_trading import run_phase5_trading
from app.cycle.phases.phase6_post import run_phase6_post

logger = logging.getLogger(__name__)
_auditor = CycleAuditor()


class OrchestratorCoreMixin:
    """
    Linear Orchestrator for the Autonomous Trading Cycle.
    Completely decoupled, strictly timed, and immune to hung workers.
    Replaces the legacy orchestrator_v1.py.
    """

    @classmethod
    async def _execute_cycle(cls, ctx: Any) -> None:
        """
        Main loop for the trading cycle.
        """
        from app.monitoring.pipeline_profiler import profiler as pipeline_profiler

        pipeline_profiler.start_cycle(ctx.cycle_id)
        cls._start_time = time.monotonic()
        loop_start = datetime.now(timezone.utc)

        logger.info(
            "===== CYCLE START: %s =====", loop_start.strftime("%Y-%m-%d %H:%M:%S")
        )

        cls._cycle_summary = {
            "cycle_id": ctx.cycle_id,
            "trigger_type": ctx.trigger_type,
            "schedule_id": getattr(ctx, "schedule_id", None),
            "execution_mode": cls._state.get("execution_mode", "production"),
            "started_at": loop_start.isoformat(),
            "finished_at": None,
            "status": "starting",
            "tickers_requested": ctx.tickers,
            "tickers_final": [],
            "collect_requested": ctx.collect,
            "analyze_requested": ctx.analyze,
            "trade_requested": ctx.trade,
            "jetson_healthy_start": False,
            "collector_ok": 0,
            "collector_skipped": 0,
            "collector_error": 0,
            "collector_failures": [],
            "analysis_results_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
            "review_count": 0,
            "trade_attempted": 0,
            "trade_executed": 0,
            "trade_failed": 0,
            "trade_skip_categories": {},
            "no_trade_reason": None,
            "primary_failure_reason": None,
        }

        bot_id = settings.BOT_ID
        try:
            from app.services.bot_manager import get_active_bot_id

            bot_id = get_active_bot_id()
        except Exception:
            pass
        try:
            await cls._execute_cycle_impl(ctx, bot_id)
        finally:
            for name, task in [
                ("scout", getattr(cls, "_scout_task", None)),
                ("consumer", getattr(cls, "_consumer_task", None)),
                ("checkpoint", getattr(cls, "_checkpoint_task", None)),
                ("autoresearch", getattr(cls, "_autoresearch_task", None)),
            ]:
                if task and not task.done():
                    task.cancel()
            cls._scout_task = None
            cls._consumer_task = None
            cls._checkpoint_task = None
            cls._autoresearch_task = None

            # Re-pause the system so background tasks go dormant
            # until the next cycle is explicitly started.
            import os as _os
            _start_paused = _os.getenv("START_PAUSED", "true").lower() in ("true", "1", "yes")
            if _start_paused:
                from app.pipeline.orchestration.cycle_control import cycle_control
                cycle_control.pause()
                logger.info("[CYCLE] Cycle ended — re-pausing system (background tasks dormant).")

    @classmethod
    async def _execute_cycle_impl(cls, ctx: Any, bot_id: str) -> None:
        """
        Executes the concurrent cycle:
          Phase 1 (Health) → Concurrent Core (Collection + Macro + Analysis) → Trading → Bounded Housekeeping

        Collection pushes tickers into an analysis queue as they finish.
        Analysis workers consume immediately — no waiting for all collection.
        Macro scout runs in parallel with collection.
        Post-cycle housekeeping and AutoResearch are timeout-bounded to prevent zombie loops.
        """
        try:
            from app.utils.trace import set_trace_id
            set_trace_id(ctx.cycle_id)

            # Checkpoint: 'created' -> 'queued'
            cls._cycle_summary["cycle_id"] = ctx.cycle_id
            cls._cycle_summary["status"] = "created"
            cls._cycle_summary["bot_id"] = bot_id
            cls._state["cycle_id"] = ctx.cycle_id
            cls._state["tickers"] = ctx.tickers
            cls._state["status"] = "created"
            
            cls._state["status"] = "queued"
            cls.emit("queued", "init", "Cycle queued for execution", status="ok")

            _auditor.phase_entry(ctx.cycle_id, "starting")

            # ── Phase 1: Health, Triage, Directives (MUST run first — safety) ──
            await run_phase1_health(
                ctx, bot_id, cls.emit, cls._cycle_summary, cls._state
            )
            _auditor.phase_exit(ctx.cycle_id, "starting")

            # ── Check Resume Skips ──
            _skip_collect = ctx.resume_from in ("analyzing", "trading")
            _skip_analyze = ctx.resume_from == "trading"

            results = []
            if _skip_analyze:
                results = cls._state.get("results", [])

            # ══════════════════════════════════════════════════════════
            # CONCURRENT CORE: Collection + Macro Scout + Analysis
            # All three run simultaneously like a real trading firm.
            # ══════════════════════════════════════════════════════════

            # Shared macro memo holder — macro scout fills this asynchronously.
            # Analysis workers read from it; if not ready yet, they proceed without.
            macro_memo_holder = {"memo": ""}

            # ── Launch Macro Scout (background) ──
            macro_task = None
            if ctx.collect and not _skip_collect:
                async def _macro_scout_bg():
                    try:
                        memo = await run_phase3_macro(cls.emit)
                        macro_memo_holder["memo"] = memo or ""
                        logger.info("[CYCLE] Macro scout complete (%d chars)", len(macro_memo_holder["memo"]))
                    except Exception as e:
                        logger.warning("[CYCLE] Macro scout failed (non-fatal): %s", e)
                        macro_memo_holder["memo"] = ""

                macro_task = asyncio.create_task(_macro_scout_bg())
                logger.info("[CYCLE] Launched macro scout in background")

            # Inject trigger_type context if this is an edge-case trigger
            if getattr(ctx, "trigger_type", "").startswith("edge_case_"):
                logger.info("[CYCLE] Injecting edge case context: %s", ctx.trigger_type)
                macro_memo_holder["memo"] += f"\n\n[URGENT] The bot has woken up specifically because an order trigger was hit: {ctx.trigger_type}. Evaluate this immediately and decide whether to execute the trade, hold, or adjust the trigger.\n\n"

            # ── Analysis Queue (tickers flow from collection → analysis) ──
            # Priority queue ensures portfolio holdings are analyzed first.
            analysis_queue = None
            if ctx.analyze and not _skip_analyze and not _skip_collect:
                from app.cycle.orchestration.priority_queue import PriorityAnalysisQueue
                _triage = cls._state.get("triage", {})
                analysis_queue = PriorityAnalysisQueue(
                    position_tickers=set(cls._state.get("position_tickers", [])),
                    deep_tickers=set(_triage.get("deep", [])),
                    glance_tickers=set(_triage.get("glance", [])),
                )

            # ── Launch Analysis Workers (background, consumes from queue) ──
            analysis_task = None
            if ctx.analyze and not _skip_analyze:
                cls._state["status"] = "started"
                cls.emit("started", "concurrent", "Starting concurrent collection + analysis", status="ok")

                if analysis_queue is not None:
                    cls.emit(
                        "analyzing",
                        "queue_created",
                        f"Analysis queue created — workers will wait for collection to push tickers",
                        status="running",
                        data={"mode": "queue", "initial_tickers": len(ctx.tickers)},
                    )

                async def _analysis_bg():
                    """Run analysis workers that consume from the queue."""
                    _auditor.phase_entry(
                        ctx.cycle_id, "analyzing", ticker_count=len(ctx.tickers)
                    )
                    nonlocal results
                    r = await run_phase4_analysis(
                        ctx, bot_id, macro_memo_holder, cls.emit, cls._cycle_summary, cls._state,
                        analysis_queue=analysis_queue,
                    )
                    results = r
                    _auditor.phase_exit(
                        ctx.cycle_id, "analyzing", results_count=len(results)
                    )

                analysis_task = asyncio.create_task(_analysis_bg())
                logger.info("[CYCLE] Launched analysis workers in background (consuming from queue)")

                # Tickers are now pushed to the analysis queue concurrently as they finish
                # collection, deduplication, summarization, and consensus in phase2_collection.

            elif _skip_analyze:
                cls.emit(
                    "analyzing",
                    "resume_skip",
                    "Skipping analysis — already completed.",
                    status="ok",
                )

            # ── Collection (pushes tickers to analysis_queue as they finish) ──
            if ctx.collect and not _skip_collect:
                _auditor.phase_entry(
                    ctx.cycle_id, "collecting", ticker_count=len(ctx.tickers)
                )

                _collection_start = time.monotonic()
                ctx.tickers = await run_phase2_collection(
                    ctx, cls.emit, cls._state, analysis_queue=analysis_queue
                )
                _collection_elapsed = int(time.monotonic() - _collection_start)

                _queue_depth = analysis_queue.qsize() if analysis_queue else 0
                cls.emit(
                    "collecting",
                    "collection_complete",
                    f"Collection finished: {len(ctx.tickers)} tickers in {_collection_elapsed}s. "
                    f"Analysis queue depth: {_queue_depth}",
                    status="ok",
                    data={
                        "tickers_count": len(ctx.tickers),
                        "elapsed_s": _collection_elapsed,
                        "queue_depth": _queue_depth,
                    },
                    elapsed_ms=_collection_elapsed * 1000,
                )

                _auditor.phase_exit(
                    ctx.cycle_id, "collecting", results_count=len(ctx.tickers)
                )
            elif _skip_collect:
                cls.emit(
                    "started",
                    "resume_skip",
                    "Skipping collection — already completed.",
                    status="ok",
                )

            # ── Signal analysis workers that collection is done ──
            if analysis_queue is not None:
                from app.config import settings as _s
                _worker_count = _s.V2_TICKER_CONCURRENCY or 3
                _pre_sentinel_depth = analysis_queue.qsize()
                for _ in range(_worker_count):
                    analysis_queue.put_nowait(None)  # sentinel
                logger.info(
                    "[CYCLE] Collection done — sent %d sentinels to analysis queue "
                    "(queue had %d items before sentinels)",
                    _worker_count, _pre_sentinel_depth,
                )
                cls.emit(
                    "analyzing",
                    "sentinels_sent",
                    f"Collection done → {_worker_count} shutdown signals sent to workers. "
                    f"Queue had {_pre_sentinel_depth} tickers pending.",
                    status="ok",
                    data={
                        "sentinel_count": _worker_count,
                        "queue_depth_before_sentinels": _pre_sentinel_depth,
                    },
                )

            # ── Wait for macro scout to finish ──
            if macro_task is not None:
                try:
                    await asyncio.wait_for(macro_task, timeout=330.0)  # 5.5min safety
                except asyncio.TimeoutError:
                    logger.warning("[CYCLE] Macro scout safety timeout — proceeding without")
                    macro_task.cancel()

            # ── Wait for analysis workers to finish ──
            if analysis_task is not None:
                cls._state["status"] = "analyzing"
                await analysis_task

            # ── Phase 5: Trading (MUST wait for all analysis) ──
            trade_result = None
            if ctx.trade:
                cls._state["status"] = "gated"
                cls.emit("gated", "trading", "Gating analysis results for trade execution", status="ok")
                trade_result = await run_phase5_trading(
                    ctx,
                    bot_id,
                    results,
                    cls.emit,
                    cls._cycle_summary,
                    cls._state,
                    _auditor,
                )
                cls._state["status"] = "traded"
                cls.emit("traded", "trading", "Trade execution completed", status="ok")

            # ── Phase 6: Post-Enrichment (bounded housekeeping) ──
            cls._state["status"] = "persisted"
            cls.emit("persisted", "post", "Persisting results and launching bounded housekeeping", status="ok")
            await run_phase6_post(
                ctx, bot_id, results, trade_result, cls.emit, cls._state, cls._cycle_summary
            )

            cls._state["status"] = "evaluated"
            cls.emit("evaluated", "post", "Cycle evaluations and metrics collected", status="ok")

            ended = datetime.now(timezone.utc).isoformat()
            cls._cycle_summary["status"] = "done"
            cls._cycle_summary["ended_at"] = ended

            cls._state["status"] = "done"
            cls._state["finished_at"] = ended
            cls.save_state()

            try:
                PipelineStateDB.clear_checkpoint(ctx.cycle_id)
            except Exception as e:
                logger.warning("Failed to clear checkpoint on success: %s", e)

            # ── Trigger AutoResearch (BOUNDED — was fire-and-forget, caused zombie loops) ──
            _AUTORESEARCH_TIMEOUT = 120  # seconds — hard cap
            try:
                from app.pipeline.analysis.autoresearch import run_autoresearch

                cls._autoresearch_task = asyncio.create_task(
                    run_autoresearch(ctx.cycle_id, dict(cls._cycle_summary))
                )
                logger.info("[CYCLE] Triggered AutoResearch for cycle %s (timeout=%ds)", ctx.cycle_id, _AUTORESEARCH_TIMEOUT)
                try:
                    await asyncio.wait_for(cls._autoresearch_task, timeout=_AUTORESEARCH_TIMEOUT)
                    logger.info("[CYCLE] AutoResearch completed successfully.")
                except asyncio.TimeoutError:
                    logger.warning(
                        "[CYCLE] AutoResearch timeout (%ds) — cancelling to prevent zombie loop",
                        _AUTORESEARCH_TIMEOUT,
                    )
                    cls._autoresearch_task.cancel()
                    try:
                        await cls._autoresearch_task
                    except (asyncio.CancelledError, Exception):
                        pass
            except Exception as ar_err:
                logger.warning("[CYCLE] Failed to trigger AutoResearch: %s", ar_err)

            cls.emit(
                "closed",
                "cycle_done",
                f"✅ Cycle {ctx.cycle_id} complete. ({len(results)} analyzed, "
                f"{cls._cycle_summary.get('trade_executed', 0)} executed)",
                status="ok",
                data=cls._cycle_summary,
            )

        except asyncio.CancelledError:
            # Differentiate between User Stop and Timeout based on elapsed time
            elapsed_sec = (time.monotonic() - cls._start_time) if hasattr(cls, "_start_time") else 0
            timeout_sec = int(getattr(settings, "CYCLE_TIMEOUT_MINUTES", 120)) * 60
            
            is_timeout = elapsed_sec >= timeout_sec
            cancel_reason = "System Timeout Hit (>%d min)" % (timeout_sec // 60) if is_timeout else "User manually stopped the cycle"
            
            logger.warning(
                f"[CYCLE] PIPELINE CANCELLED ({cancel_reason})."
            )
            cls._cycle_summary["status"] = "stopped"
            cls._state["status"] = "stopped"
            cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
            cls.save_state()

            if "primary_failure_reason" not in cls._cycle_summary:
                cls._cycle_summary["primary_failure_reason"] = cancel_reason
            cls.emit(
                "trading",
                "cancelled",
                f"Cycle cancelled: {cancel_reason}",
                status="error",
            )

            try:
                # If a checkpoint exists, it's an interrupted cycle, not a failed one.
                # But we definitely want to log it so we know *why* it stopped.
                PipelineStateDB.log_execution_error(
                    cycle_id=ctx.cycle_id,
                    phase=cls._cycle_summary.get("status", "unknown"),
                    ticker="system",
                    error_type="cycle_cancelled",
                    error_message="Task was cancelled (e.g. by user stop or timeout)",
                    stack_trace="CancelledError",
                )
            except Exception:
                pass
            raise

        except Exception as e:
            logger.error("[CYCLE] FATAL PIPELINE ERROR: %s", e)
            logger.debug(traceback.format_exc())
            cls._cycle_summary["status"] = "error"
            cls._state["status"] = "error"
            cls._state["finished_at"] = datetime.now(timezone.utc).isoformat()
            cls.save_state()

            if "primary_failure_reason" not in cls._cycle_summary:
                cls._cycle_summary["primary_failure_reason"] = f"Fatal Error: {e}"
            cls.emit("trading", "error", f"Cycle failed: {e}", status="error")

            try:
                PipelineStateDB.log_execution_error(
                    cycle_id=ctx.cycle_id,
                    phase=cls._cycle_summary.get("status", "unknown"),
                    ticker="system",
                    error_type="fatal_pipeline_crash",
                    error_message=str(e)[:500],
                    stack_trace=traceback.format_exc()[:2000],
                )
            except Exception:
                pass
            raise
        finally:
            cls._finalize_cycle_telemetry(ctx)

    @classmethod
    def _finalize_cycle_telemetry(cls, ctx: Any) -> None:
        """Flush remaining events, end the pipeline profiler, and persist the benchmark report."""
        try:
            if hasattr(cls, "flush_events"):
                cls.flush_events()
        except Exception as e:
            logger.warning("[CYCLE] Failed to flush events for telemetry: %s", e)

        try:
            from app.monitoring.pipeline_profiler import profiler as pipeline_profiler
            pipeline_profiler.end_cycle()
        except Exception as e:
            logger.warning("[CYCLE] Failed to end pipeline profiler: %s", e)

        try:
            from app.pipeline.analysis.benchmark import persist_benchmark
            bench_state = dict(cls._state)
            
            # Map requested/effective version keys
            bench_state["requested_version"] = cls._state.get("requested_pipeline_version", "v2")
            bench_state["effective_version"] = cls._state.get("effective_pipeline_version", "v2")
            
            # Query the final list of events from the DB so we have a complete log
            state_db = PipelineStateDB.get_state(summary_only=False)
            bench_state["events"] = state_db.get("events", [])
            
            persist_benchmark(bench_state)
        except Exception as e:
            logger.warning("[CYCLE] Failed to persist cycle benchmark: %s", e)


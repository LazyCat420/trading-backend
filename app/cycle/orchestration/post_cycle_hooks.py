import logging
from app.services.pipeline_service import PipelineService

logger = logging.getLogger(__name__)


async def run_post_cycle_hooks(
    ticker: str,
    result: dict,
    escalated: bool,
    cycle_id: str,
    final_action: str,
    final_confidence: int,
) -> None:
    """Execute all side effects (learning, tracking, health) after a decision."""
    # ── Cycle tracking: TSV recording ──
    try:
        from app.cycle.orchestration.cycle_recorder import append_cycle_result

        append_cycle_result(
            cycle_id=cycle_id or "manual",
            ticker=ticker,
            action=final_action,
            confidence=final_confidence,
            tokens_used=result.get("tokens", 0),
            elapsed_sec=result.get("elapsed_ms", 0) / 1000.0,
        )

        # A6: Save best result
        from app.pipeline.best_result_store import save_best_result

        save_best_result(ticker, result, final_confidence)

        # A7: cycle_diary.md appender
        from datetime import datetime, timezone
        from pathlib import Path

        DIARY_PATH = Path("memory/cycle_tracking/cycle_diary.md")
        DIARY_PATH.parent.mkdir(parents=True, exist_ok=True)

        diary_entry = f"## {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} - {ticker}\n"
        diary_entry += f"- **Action:** {final_action} @ {final_confidence}%\n"
        diary_entry += f"- **Rationale:** {result.get('rationale', '')[:200]}...\n"
        missing = result.get('missing_sources', [])
        if missing:
            diary_entry += f"- **Missing Data:** {', '.join(missing)}\n"
        diary_entry += "\n"

        with open(DIARY_PATH, "a", encoding="utf-8") as df:
            df.write(diary_entry)

        # A9: Snapshot archive (decision context for forensic replay)
        try:
            from app.pipeline.snapshot_writer import write_snapshot

            write_snapshot(
                cycle_id or "manual",
                ticker,
                {
                    "action": final_action,
                    "confidence": final_confidence,
                    "rationale": result.get("rationale", "")[:500],
                    "config_used": result.get("config_used", ""),
                    "escalated": escalated,
                },
            )
        except Exception as snap_err:
            logger.debug(
                "[PIPELINE] Snapshot write failed for %s (non-fatal): %s",
                ticker,
                snap_err,
            )

        # A8: Regression Detector — compare against best historical result
        try:
            from app.pipeline.best_result_store import load_best_result

            best = load_best_result(ticker)
            if best:
                old_conf = best.get("confidence", 0)
                if old_conf - final_confidence > 30:
                    logger.warning(
                        "[REGRESSION] %s: confidence dropped %d → %d (delta=%d)",
                        ticker,
                        old_conf,
                        final_confidence,
                        old_conf - final_confidence,
                    )
        except Exception:
            pass

    except Exception as track_err:
        logger.error(
            "[PIPELINE] [TRACKING] TSV append failed for %s: %s", ticker, track_err
        )

    # ── Post-cycle learning: extract reusable lesson for memory ──
    try:
        from app.cycle.orchestration.post_cycle_learn import maybe_learn

        await maybe_learn(ticker, result, escalated=escalated)
    except Exception as learn_err:
        logger.error(
            "[PIPELINE] [LEARN] Failed for %s: %s", ticker, learn_err, exc_info=True
        )
        PipelineService.emit(
            "analyzing",
            f"learn_error_{ticker}",
            f"{ticker}: Learning failure: {learn_err}",
            status="error",
        )

    # ── Outcome tracking: record BUY/SELL for trade journal validation ──
    try:
        from app.pipeline.analysis.outcome_tracker import record_decision

        outcome_id = record_decision(
            cycle_id=cycle_id or "manual",
            ticker=ticker,
            action=final_action,
            confidence=final_confidence,
            lesson=result.get("rationale", "")[:200],
        )

        # ── Strategy Tracking: Record the prompt that drove this outcome ──
        if outcome_id and final_action in ("BUY", "SELL"):
            from app.pipeline.strategy_tracker import record_strategy
            from app.db.connection import get_db

            # For the main pipeline, we track the hybrid config used
            config_used = result.get("config_used", "C")
            prompt_hash = f"static_hybrid_{config_used}"

            # Fetch the entry price from decision_outcomes
            with get_db() as db:
                ep_row = db.execute(
                    "SELECT entry_price FROM decision_outcomes WHERE id = %s",
                    [outcome_id],
                ).fetchone()
                ep = ep_row[0] if ep_row else None

                record_strategy(
                    strategy_candidate_id=None,
                    decision_outcome_id=outcome_id,
                    agent_prompt_hash=prompt_hash,
                    ticker=ticker,
                    signal=final_action,
                    entry_price=ep,
                )

    except Exception as outcome_err:
        logger.error(
            "[PIPELINE] [OUTCOME] Failed for %s: %s", ticker, outcome_err, exc_info=True
        )
        PipelineService.emit(
            "analyzing",
            f"outcome_error_{ticker}",
            f"{ticker}: Outcome tracking failure: {outcome_err}",
            status="error",
        )

    # ── Update watchlist health signals with analysis result ──
    try:
        from app.pipeline.watchlist_health import update_signals_from_analysis

        update_signals_from_analysis(
            ticker,
            {
                "action": final_action,
                "confidence": final_confidence,
            },
        )
    except Exception as health_err:
        logger.error(
            "[PIPELINE] [HEALTH] Signal update failed for %s: %s",
            ticker,
            health_err,
            exc_info=True,
        )
        PipelineService.emit(
            "analyzing",
            f"health_error_{ticker}",
            f"{ticker}: Watchlist health tracking failure: {health_err}",
            status="error",
        )

    # ── Living Graph: create TradeDecision node in ontology graph ──
    try:
        from app.cognition.ontology.graph_mutations import create_trade_decision

        create_trade_decision(
            ticker=ticker,
            action=final_action,
            confidence=final_confidence,
            cycle_id=cycle_id or "manual",
            rationale=result.get("rationale", "")[:200],
        )
    except Exception as graph_err:
        logger.debug(
            "[PIPELINE] [GRAPH] TradeDecision write failed for %s (non-fatal): %s",
            ticker,
            graph_err,
        )

    # ── Checkpoint cleanup: clear completed step records ──
    try:
        from app.db.checkpoints import checkpoint_manager

        checkpoint_manager.clear_cycle(cycle_id)
    except Exception as cp_err:
        logger.debug(
            "[PIPELINE] [CHECKPOINT] Cleanup failed for %s (non-fatal): %s",
            ticker,
            cp_err,
        )

    # ── Decoupled Async Eval Engine: process traces immediately without blocking ──
    try:
        from app.autoresearch.eval_worker import run_eval_worker
        import asyncio
        # Create an async task to run the eval worker without delaying the current trading cycle
        asyncio.create_task(run_eval_worker(limit=10))
    except Exception as eval_err:
        logger.debug(
            "[PIPELINE] [EVAL WORKER] Failed to spawn async eval worker (non-fatal): %s",
            eval_err,
        )

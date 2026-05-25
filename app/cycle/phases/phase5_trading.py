import logging
import traceback
from typing import Callable, Any

from app.cycle.orchestration.state_manager import PipelineStateDB
from app.services.logging.cycle_auditor import CycleAuditor

logger = logging.getLogger(__name__)


async def run_phase5_trading(
    ctx: Any,
    bot_id: str,
    results: list[dict],
    emit: Callable,
    cycle_summary: dict,
    state: dict,
    auditor: CycleAuditor,
) -> dict:
    """
    Phase 5: Trading
    Executes trades explicitly based on the analysis results.
    """
    trade_summary = {"executed": 0, "skipped": 0, "portfolio": {}}
    trade_result = None

    if ctx.trade and results:
        state["status"] = "trading"
        auditor.phase_entry(ctx.cycle_id, "trading", ticker_count=len(results))
        emit(
            "trading",
            "start",
            f"Executing trade decisions for {len(results)} tickers",
        )
        try:
            from app.cycle.trading_phase import execute_decisions

            trade_result = await execute_decisions(
                results, bot_id=bot_id, cycle_id=ctx.cycle_id
            )
            counts = trade_result.get("counts", {})
            trade_summary = {
                "executed": counts.get("buy_executed", 0)
                + counts.get("sell_executed", 0),
                "skipped": counts.get("holds", 0)
                + counts.get("human_review", 0)
                + counts.get("buy_failed", 0)
                + counts.get("sell_failed", 0),
                "portfolio": trade_result.get("portfolio", {}),
            }

            cycle_summary["trade_attempted"] = len(results)
            cycle_summary["trade_executed"] = trade_summary["executed"]
            cycle_summary["trade_failed"] = counts.get("buy_failed", 0) + counts.get(
                "sell_failed", 0
            )
            cycle_summary["trade_skip_categories"] = counts

            emit(
                "trading",
                "complete",
                f"Executed {trade_summary['executed']} trades",
                status="ok",
                data=trade_summary,
            )
            auditor.phase_exit(
                ctx.cycle_id,
                "trading",
                results_count=trade_summary["executed"],
                errors_count=counts.get("buy_failed", 0) + counts.get("sell_failed", 0),
            )
        except Exception as e:
            logger.error("Trading crashed: %s", e)
            emit("trading", "fatal", f"Trading crashed: {e}", status="error")
            state["error"] = str(e)
            try:
                PipelineStateDB.log_execution_error(
                    cycle_id=ctx.cycle_id,
                    phase="trading",
                    ticker="system",
                    error_type="trading_crash",
                    error_message=str(e)[:500],
                    stack_trace=traceback.format_exc()[:2000],
                )
            except Exception:
                pass

        try:
            from app.trading.portfolio import take_snapshot

            take_snapshot(bot_id)
            emit("trading", "snapshot", "Portfolio snapshot saved", status="ok")
        except Exception as e:
            logger.warning("Snapshot failed: %s", e)

    elif ctx.trade and not results:
        emit("trading", "skipped", "No analysis results to trade", status="skipped")
        cycle_summary["no_trade_reason"] = "zero_results"

    return trade_result

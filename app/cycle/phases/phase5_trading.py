import logging
from typing import Callable

from app.cycle.orchestration.state_manager import PipelineStateDB
from app.services.logging.cycle_auditor import CycleAuditor
from app.cycle.trading_phase import execute_decisions
from app.trading.portfolio import take_snapshot
from app.cycle.context import CycleContext
from app.utils.emit import noop_emit

logger = logging.getLogger(__name__)


async def run_phase5_trading(
    ctx: CycleContext,
    bot_id: str,
    results: list[dict],
    emit: Callable = noop_emit,
    cycle_summary: dict = None,
    state: dict = None,
    auditor: CycleAuditor = None,
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
            PipelineStateDB.safe_log_execution_error(
                cycle_id=ctx.cycle_id,
                phase="trading",
                error_type="trading_crash",
                error=e,
            )

        try:
            take_snapshot(bot_id)
            emit("trading", "snapshot", "Portfolio snapshot saved", status="ok")
        except Exception as e:
            logger.warning("Snapshot failed: %s", e)

    elif ctx.trade and not results:
        emit("trading", "skipped", "No analysis results to trade", status="skipped")
        cycle_summary["no_trade_reason"] = "zero_results"

    return trade_result

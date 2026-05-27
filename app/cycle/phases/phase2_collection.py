import logging
from typing import Callable, Any
import asyncio

logger = logging.getLogger(__name__)


async def run_phase2_collection(
    ctx: Any, emit: Callable, state: dict,
    analysis_queue: asyncio.Queue | None = None,
) -> list[str]:
    """
    Phase 2: Data Collection
    Collects data for all tickers. When analysis_queue is provided,
    tickers are pushed to analysis as they finish (concurrent pipelining).
    """
    try:
        from app.pipeline.data.data_phase import run as run_data

        data_results = await run_data(
            ctx.tickers,
            emit=emit,
            force_global=None,
            position_tickers=state.get("position_tickers", []),
            triage_data=state.get("triage", {}),
            analysis_queue=analysis_queue,
            max_tickers=getattr(ctx, "max_tickers", None),
        )

        # Merge updated ticker list (may include discovered tickers)
        if "tickers" in data_results:
            ctx.tickers = data_results["tickers"]
            state["tickers"] = ctx.tickers

        collected_count = data_results.get("collected_count", len(ctx.tickers))
        total_count = len(ctx.tickers) or 1
        coverage_pct = round((collected_count / total_count) * 100, 1)
        state["data_coverage_pct"] = coverage_pct
        
        # Smart Pipeline Phase 4: Pass redundant tickers to state for deep research
        if "processors" in data_results and "highly_redundant_tickers" in data_results["processors"]:
            redundant = data_results["processors"]["highly_redundant_tickers"]
            state["highly_redundant_tickers"] = redundant
            if redundant:
                logger.info(f"[PIPELINE] Phase 2 passing {len(redundant)} redundant tickers to Phase 4 for Deep Dive.")
                
        emit(
            "collecting",
            "data_coverage",
            f"Data coverage: {coverage_pct}% ({collected_count}/{total_count} tickers)",
            status="ok" if coverage_pct >= 80 else "warning",
            data={"coverage_pct": coverage_pct, "collected": collected_count, "total": total_count},
        )

        return ctx.tickers
    except Exception as e:
        logger.error("Collection crashed: %s", e)
        emit("collecting", "fatal", f"Collection crashed: {e}", status="error")
        # Fix E.1: Signal downstream phases that collection failed so they can
        # enforce stricter data quality gates instead of analyzing stale data.
        state["collection_failed"] = True
        # We do not raise here, we allow the cycle to gracefully continue with what it has,
        # or fail gracefully in Phase 4 if tickers are missing data.
        return ctx.tickers


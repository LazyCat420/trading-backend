import asyncio
import logging
import datetime as _dt
from typing import Callable

from app.db.connection import get_db

logger = logging.getLogger(__name__)

async def startup_vllm_discovery():
    # ── Auto-discover vLLM models on all endpoints ──
    try:
        from app.services.vllm_client import llm

        roles = await llm.discover_roles()
        ep_summary = []
        for ep_name, ep_obj in llm._endpoints.items():
            status = "✓" if ep_obj.model else "✗ OFFLINE"
            ep_summary.append(f"{ep_name}={ep_obj.model or 'N/A'} [{status}]")
        logger.info("vLLM endpoints discovered: %s", " | ".join(ep_summary))
    except Exception as e:
        logger.warning("vLLM model discovery failed (non-fatal): %s", e)

async def warmup_embedder():
    try:
        from app.services.embedding_service import embedder

        await asyncio.to_thread(embedder.embed_text, "warmup")
        logger.info("Embedding model loaded.")
    except Exception as e:
        logger.warning("Embedding model not loaded: %s", e)

def _is_data_fresh(table: str, where_clause: str, max_age_days: int) -> bool:
    try:
        with get_db() as db:
            latest = db.execute(
                f"SELECT MAX(date) FROM {table} WHERE {where_clause}"
            ).fetchone()
        if latest and latest[0]:
            age_days = (
                (_dt.date.today() - latest[0]).days
                if hasattr(latest[0], "days")
                else 0
            )
            if age_days < max_age_days:
                return True
    except Exception:
        pass
    return False

async def startup_fred_refresh(is_shutting_down: Callable[[], bool]):
    await asyncio.sleep(3)  # let server fully boot first
    if is_shutting_down():
        return
    # Skip if we already have fresh FRED data (avoids 2+ minute delay on
    # every server restart, which is critical when --reload kills cycles)
    if _is_data_fresh("macro_indicators", "source = 'fred'", 1):
        logger.info("[startup] FRED data already fresh, skipping refresh")
        return

    logger.info("[startup] Refreshing FRED macro indicators (background thread)...")
    try:
        from app.collectors.fred_collector import sync_collect_fred
        total = await asyncio.to_thread(sync_collect_fred, is_shutting_down)
        logger.info("[startup] FRED refresh complete: %d total rows", total)
    except asyncio.CancelledError:
        logger.info("[startup] FRED refresh cancelled.")
    except Exception as e:
        logger.warning("[startup] FRED refresh failed (non-fatal): %s", e)

async def startup_market_collect(is_shutting_down: Callable[[], bool]):
    """Background: collect market regime data (indexes, VIX, yields, ETFs)."""
    if is_shutting_down():
        return
    with get_db() as db:
        # Check if we have recent data
        recent = db.execute(
            "SELECT COUNT(*) FROM asset_prices WHERE date >= CURRENT_DATE - INTERVAL '1 day'"
        ).fetchone()[0]

        # Check if we explicitly have commodity data
        commodities = db.execute(
            "SELECT COUNT(*) FROM asset_prices WHERE asset_class = 'commodity'"
        ).fetchone()[0]

    needs_collect = recent < 50 or commodities == 0

    if needs_collect:
        logger.info(
            "[startup] Collecting market regime data (background)... (commodities=%d)",
            commodities,
        )
        try:
            from app.collectors.market_regime_collector import collect_market_data

            result = await collect_market_data(period="6mo")
            logger.info(
                "[startup] Market data collected: %s", result.get("total", 0)
            )
        except asyncio.CancelledError:
            logger.info("[startup] Market collect cancelled.")
        except Exception as e:
            logger.warning("[startup] Market collect failed (non-fatal): %s", e)
    else:
        logger.info(
            "[startup] Market data already fresh (%d recent rows, %d commodities), skipping collection",
            recent,
            commodities,
        )

    # ALWAYS compute regime + breadth + correlations so cache & DB are populated
    try:
        from app.data.market_regime_engine import (
            compute_market_regime,
            compute_sector_breadth,
        )
        from app.data.sector_correlation_engine import compute_all_correlations
        from app.data.sector_aggregator import backfill_sector_performance

        await compute_market_regime()
        await compute_sector_breadth()
        await backfill_sector_performance()
        await compute_all_correlations()
        logger.info("[startup] Analytics and cross-asset correlations computed")
    except Exception as e:
        logger.warning("[startup] Market compute failed (non-fatal): %s", e)

async def startup_sp500_seed(is_shutting_down: Callable[[], bool]):
    """Background: seed SP500 universe from hardcoded list if DB is empty.

    Uses app/data/sp500_constituents.py — no network required.
    """
    if is_shutting_down():
        return
    with get_db() as db:
        sp500_count = db.execute(
            "SELECT COUNT(*) FROM ticker_metadata WHERE sp500=TRUE"
        ).fetchone()[0]

    if sp500_count > 400:
        logger.info(
            "[startup] SP500 universe already loaded (%d tickers)", sp500_count
        )
        return

    logger.info(
        "[startup] SP500 universe missing or incomplete (%d tickers) — seeding from hardcoded list...",
        sp500_count,
    )
    try:
        from app.data.sp500_universe import load_sp500_universe

        result = await load_sp500_universe(enrich=False)
        logger.info("[startup] SP500 universe loaded: %d tickers", result)
    except asyncio.CancelledError:
        logger.info("[startup] SP500 seed cancelled.")
    except Exception as e:
        logger.warning("[startup] SP500 seed failed (non-fatal): %s", e)

async def startup_all(is_shutting_down: Callable[[], bool]):
    """Run all startup data tasks sequentially.

    Tasks are run in sequence to avoid overwhelming external APIs
    during startup.
    """
    try:
        await startup_fred_refresh(is_shutting_down)
    except Exception as e:
        logger.error("[startup] FRED task failed: %s", e, exc_info=True)
    try:
        await startup_market_collect(is_shutting_down)
    except Exception as e:
        logger.error("[startup] Market task failed: %s", e, exc_info=True)
    try:
        await startup_sp500_seed(is_shutting_down)
    except Exception as e:
        logger.error("[startup] SP500 task failed: %s", e, exc_info=True)

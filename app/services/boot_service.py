import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class BootService:
    @classmethod
    async def startup(cls):
        """Main startup sequence coordinator."""
        logger.info("[Boot] Starting application boot sequence...")

        # --- Required Boot Stages ---
        cls._run_stage("DB Connection & Schema", cls._init_database, required=True)
        cls._run_stage("Vector Store Indexes", cls._init_vector_indices, required=True)
        cls._run_stage("Reset Application State", cls._reset_app_state, required=True)

        # --- Optional / Degraded Boot Stages ---
        cls._run_stage("Scheduler Start", cls._start_scheduler, required=False)
        cls._run_stage("Embedding Warmup", cls._warmup_models, required=False)

        # --- Background Tasks ---
        # Spawns a background, non-blocking task for long-running startup data refreshes
        asyncio.create_task(cls._start_background_tasks())

        logger.info("[Boot] Application boot sequence completed successfully.")

    @classmethod
    async def shutdown(cls):
        """Main shutdown sequence coordinator."""
        logger.info("[Boot] Shutting down...")

        # Cancel any running trading cycle
        try:
            from app.services.pipeline_service import PipelineService

            await PipelineService.cancel_cycle_shutdown()
        except Exception as e:
            logger.warning("[Boot] Cycle cancellation on shutdown: %s", e)

        # Stop cycle scheduler
        try:
            from app.services.cycle_scheduler import SchedulerService

            SchedulerService.stop()
        except Exception as e:
            logger.warning("[Boot] Scheduler shutdown error: %s", e)

        # Close the vLLM HTTP client
        try:
            from app.services.vllm_client import llm

            await llm.close()
        except Exception as e:
            logger.warning("[Boot] vLLM client close: %s", e)

        # Close PostgreSQL connection pool
        try:
            from app.db.connection import close_db

            close_db()
            logger.info("[Boot] PostgreSQL connection pool closed.")
        except Exception as e:
            logger.warning("[Boot] PostgreSQL close: %s", e)

        logger.info("[Boot] Shutdown complete.")

    @classmethod
    def _run_stage(cls, name: str, stage_func, required: bool = True):
        t0 = time.perf_counter()
        try:
            stage_func()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Boot] Stage '{name}' completed in {elapsed_ms:.1f}ms")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if required:
                logger.error(
                    f"[Boot] Stage '{name}' FAILED in {elapsed_ms:.1f}ms: {e}. Aborting boot."
                )
                raise e
            else:
                logger.warning(
                    f"[Boot] Stage '{name}' FAILED in {elapsed_ms:.1f}ms: {e}. Proceeding in degraded mode."
                )

    # -------------------------------------------------------------------------
    # INDIVIDUAL STAGES
    # -------------------------------------------------------------------------

    @classmethod
    def _init_database(cls):
        from app.db.connection import get_db

        get_db()

    @classmethod
    def _init_vector_indices(cls):
        # pgvector HNSW + FTS indexes are created in schema_pg.sql
        pass

    @classmethod
    def _reset_app_state(cls):
        from app.services.pipeline_service import PipelineService

        PipelineService.reset_on_boot()

        # Start the system PAUSED by default on boot.
        # This prevents all scheduled LLM tasks (morning briefing, flash briefing,
        # janitor, eval worker, etc.) from firing until the user explicitly starts
        # a trading run or resumes via the UI.
        # Override with START_PAUSED=false in env to auto-start.
        import os
        start_paused = os.getenv("START_PAUSED", "true").lower() in ("true", "1", "yes")
        if start_paused:
            from app.pipeline.orchestration.cycle_control import cycle_control
            cycle_control.pause()
            logger.info("[Boot] System starts PAUSED — LLM tasks gated until user resumes or starts a cycle.")

    @classmethod
    def _start_scheduler(cls):
        from app.services.cycle_scheduler import SchedulerService

        SchedulerService.start()

    @classmethod
    def _warmup_models(cls):
        from app.services.embedding_service import embedder

        embedder.embed_text("warmup")
        logger.info("[Boot] Embedding model loaded.")

    @classmethod
    async def _start_background_tasks(cls):
        """Run all startup data tasks sequentially.

        Tasks are run in sequence to avoid overwhelming external APIs
        during startup.
        """
        try:
            await cls._startup_fred_refresh()
        except Exception as e:
            logger.warning("[startup] FRED task failed: %s", e)
        try:
            await cls._startup_market_collect()
        except Exception as e:
            logger.warning("[startup] Market task failed: %s", e)
        try:
            await cls._startup_sp500_seed()
        except Exception as e:
            logger.warning("[startup] SP500 task failed: %s", e)

    @staticmethod
    def _sync_collect_fred():
        """Fetch all FRED series (runs in a thread)."""
        import datetime
        import time
        from fredapi import Fred
        from app.config import settings
        from app.db.connection import get_db

        key = settings.FRED_API_KEY
        if not key:
            logger.warning("[startup] FRED_API_KEY not set — skipping FRED refresh")
            return 0

        from app.collectors.fred_collector import SERIES

        client = Fred(api_key=key)

        start = datetime.date.today() - datetime.timedelta(days=30 * 365)
        total = 0

        for name, series_id in SERIES.items():
            try:
                data = client.get_series(series_id, observation_start=start)
                if data is None or data.empty:
                    continue
                # Batch collect rows then executemany
                rows = []
                for date, value in data.items():
                    if str(value) == "nan" or value is None:
                        continue
                    rows.append((name, date.date(), float(value), "US", "fred"))
                if rows:
                    with get_db() as db:
                        for row in rows:
                            db.execute(
                                "INSERT INTO macro_indicators "
                                "(indicator, date, value, country, source) "
                                "VALUES (%s, %s, %s, %s, %s) "
                                "ON CONFLICT (indicator, date, country) DO NOTHING",
                                list(row),
                            )
                        total += len(rows)
                logger.info("[startup] FRED %s: %d rows", name, len(rows))
            except Exception as e:
                logger.warning("[startup] FRED %s failed: %s", name, e)
            # Yield CPU between series
            time.sleep(0.1)

        return total

    @classmethod
    async def _startup_fred_refresh(cls):
        await asyncio.sleep(3)  # let server fully boot first
        logger.info("[startup] Refreshing FRED macro indicators (background thread)...")
        try:
            total = await asyncio.to_thread(cls._sync_collect_fred)
            logger.info("[startup] FRED refresh complete: %d total rows", total)
        except Exception as e:
            logger.warning("[startup] FRED refresh failed (non-fatal): %s", e)

    @classmethod
    async def _startup_market_collect(cls):
        """Background: collect market regime data (indexes, VIX, yields, ETFs)."""
        from app.db.connection import get_db

        with get_db() as db:
            # Skip if we already have recent data
            recent = db.execute(
                "SELECT COUNT(*) FROM asset_prices WHERE date >= CURRENT_DATE - INTERVAL '1 day'"
            ).fetchone()[0]
            if recent > 50:
                logger.info(
                    "[startup] Market data already fresh (%d recent rows), skipping",
                    recent,
                )
                return
            logger.info("[startup] Collecting market regime data (background)...")
            try:
                from app.collectors.market_regime_collector import collect_market_data

                result = await collect_market_data(period="6mo")
                logger.info(
                    "[startup] Market data collected: %s", result.get("total", 0)
                )
                # Compute regime + breadth
                from app.data.market_regime_engine import (
                    compute_market_regime,
                    compute_sector_breadth,
                )

                await compute_market_regime()
                await compute_sector_breadth()
            except Exception as e:
                logger.warning("[startup] Market collect failed (non-fatal): %s", e)

    @classmethod
    async def _startup_sp500_seed(cls):
        """Background: seed SP500 universe + prices if DB is empty."""
        from app.db.connection import get_db

        with get_db() as db:
            sp500_count = db.execute(
                "SELECT COUNT(*) FROM ticker_metadata WHERE sp500=TRUE"
            ).fetchone()[0]
            if sp500_count > 400:
                logger.info(
                    "[startup] SP500 universe already loaded (%d tickers)", sp500_count
                )
                # Check if price data exists
                price_count = db.execute(
                    "SELECT COUNT(*) FROM price_history"
                ).fetchone()[0]
                if price_count == 0:
                    logger.info(
                        "[startup] No price data — collecting SP500 prices (background)..."
                    )
                    try:
                        from app.data.sp500_price_collector import collect_sp500_prices

                        price_result = await collect_sp500_prices(period="6mo")
                        logger.info(
                            "[startup] SP500 prices collected: %s",
                            price_result.get("total", 0),
                        )
                    except Exception as e:
                        logger.warning("[startup] Price collection failed: %s", e)
                # Compute sector analytics if missing
                perf_count = db.execute(
                    "SELECT COUNT(*) FROM sector_performance"
                ).fetchone()[0]
                if perf_count == 0:
                    logger.info("[startup] Computing sector analytics...")
                    try:
                        from app.data.sector_aggregator import (
                            compute_sector_performance,
                            backfill_sector_performance,
                        )

                        await backfill_sector_performance()
                        await compute_sector_performance()
                    except Exception as e:
                        logger.warning("[startup] Sector compute failed: %s", e)
                return
            logger.info("[startup] Seeding SP500 universe (background)...")
            try:
                from app.data.sp500_universe import load_sp500_universe

                result = await load_sp500_universe(enrich=False)
                logger.info("[startup] SP500 universe loaded: %s", result)
                # Collect prices in background
                from app.data.sp500_price_collector import collect_sp500_prices

                price_result = await collect_sp500_prices(period="6mo")
                logger.info(
                    "[startup] SP500 prices collected: %s", price_result.get("total", 0)
                )
                # Compute sector analytics
                from app.data.sector_aggregator import (
                    compute_sector_performance,
                    backfill_sector_performance,
                )

                await backfill_sector_performance()
                await compute_sector_performance()
            except Exception as e:
                logger.warning("[startup] SP500 seed failed (non-fatal): %s", e)

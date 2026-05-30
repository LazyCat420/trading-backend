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

        # --- Crash Recovery Detection ---
        cls._run_stage("Crash Recovery Scan", cls._detect_crashed_cycles, required=False)

        # --- Optional / Degraded Boot Stages ---
        cls._run_stage("Scheduler Start", cls._start_scheduler, required=False)
        cls._run_stage("Embedding Warmup", cls._warmup_models, required=False)
        cls._run_stage("Auto-Register MCP Servers", cls._register_mcp_servers, required=False)

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

        # Reset any zombie-state pruned tools from the ToolOptimizer.
        # Prism-routed agents never reported tool usage, causing all tools to
        # get pruned after 4+ cycles. This clears that state on every boot.
        try:
            from app.services.tool_optimizer import reset_all_pruned
            reset_all_pruned()
        except Exception as e:
            logger.warning("[Boot] Failed to reset pruned tools (non-fatal): %s", e)

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
    def _detect_crashed_cycles(cls):
        """Scan cycle logs for incomplete cycles from previous container runs."""
        from app.log_manager import log_manager

        crashed = log_manager.detect_and_log_crashed_cycles(max_age_hours=48)
        if crashed:
            logger.warning(
                "[Boot] CRASH RECOVERY: Found %d interrupted cycle(s) from previous runs:",
                len(crashed),
            )
            for c in crashed:
                logger.warning(
                    "[Boot]   → %s: last_step=%s, last_ticker=%s, "
                    "%d/%d tickers abandoned",
                    c["cycle_id"],
                    c["last_step"],
                    c.get("last_ticker", "?"),
                    len(c.get("abandoned", [])),
                    c.get("total_tickers", 0),
                )
        else:
            logger.info("[Boot] No crashed cycles detected from previous runs.")

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

    @classmethod
    def _register_mcp_servers(cls):
        """Register lazy-tool-service as an MCP server in Prism's MongoDB and trigger connection."""
        import os
        import datetime
        from urllib.parse import urlparse
        import pymongo
        import httpx
        from app.config import settings

        if not settings.PRISM_ENABLED or not settings.PRISM_MONGO_URI:
            logger.info("[MCP-Reg] Prism integration is disabled or PRISM_MONGO_URI is empty.")
            return

        try:
            # 1. Connect to MongoDB
            client = pymongo.MongoClient(settings.PRISM_MONGO_URI)
            db_name = settings.PRISM_MONGO_DB or "prism"
            db = client[db_name]
            col = db["mcp_servers"]

            # 2. Determine MCP server URL dynamically
            prism_host = settings.DEFAULT_HOST
            if settings.PRISM_URL:
                try:
                    prism_parsed = urlparse(settings.PRISM_URL)
                    prism_host = prism_parsed.hostname or settings.DEFAULT_HOST
                except Exception:
                    pass

            port = os.getenv("LAZY_TOOL_SERVICE_PORT", "5591")
            mcp_url = f"http://{prism_host}:{port}/mcp/sse"
            logger.info(f"[MCP-Reg] Calculated MCP URL: {mcp_url}")

            configs = [
                {
                    "project": "coding",
                    "username": "admin",
                    "name": "lazy-tool-service",
                    "displayName": "Lazy Tool Service",
                    "transport": "sse",
                    "url": mcp_url,
                    "enabled": True,
                },
                {
                    "project": "vllm-trading-bot",
                    "username": "lazy-trader",
                    "name": "lazy-tool-service",
                    "displayName": "Lazy Tool Service",
                    "transport": "sse",
                    "url": mcp_url,
                    "enabled": True,
                }
            ]

            now = datetime.datetime.utcnow()

            for config in configs:
                col.update_one(
                    {
                        "project": config["project"],
                        "username": config["username"],
                        "name": config["name"],
                    },
                    {
                        "$set": {
                            "displayName": config["displayName"],
                            "transport": config["transport"],
                            "url": config["url"],
                            "enabled": config["enabled"],
                            "updatedAt": now,
                        },
                        "$setOnInsert": {
                            "createdAt": now,
                        }
                    },
                    upsert=True
                )
            logger.info("[MCP-Reg] Registered lazy-tool-service in MongoDB.")

            # 3. Trigger immediate connection in Prism via API calls
            if settings.PRISM_URL:
                logger.info("[MCP-Reg] Triggering immediate MCP connections in Prism...")
                for config in configs:
                    try:
                        headers = {
                            "x-project": config["project"],
                            "x-username": config["username"]
                        }
                        # GET existing servers for this project/username to get the server ID
                        r_list = httpx.get(f"{settings.PRISM_URL}/mcp-servers", headers=headers, timeout=5.0)
                        if r_list.status_code == 200:
                            servers = r_list.json()
                            target_id = None
                            for s in servers:
                                if s.get("name") == config["name"]:
                                    target_id = s.get("id") or s.get("_id")
                                    break
                            
                            if target_id:
                                # Trigger connect
                                r_conn = httpx.post(
                                    f"{settings.PRISM_URL}/mcp-servers/{target_id}/connect",
                                    headers=headers,
                                    timeout=10.0
                                )
                                if r_conn.status_code == 200:
                                    logger.info(f"[MCP-Reg] Connected to '{config['name']}' for project '{config['project']}'/username '{config['username']}'")
                                else:
                                    logger.warning(f"[MCP-Reg] Failed to connect '{config['name']}' (HTTP {r_conn.status_code}): {r_conn.text}")
                            else:
                                logger.warning(f"[MCP-Reg] Could not find registered '{config['name']}' in Prism GET list.")
                        else:
                            logger.warning(f"[MCP-Reg] Failed to fetch server list from Prism (HTTP {r_list.status_code})")
                    except Exception as e:
                        logger.warning(f"[MCP-Reg] Failed to trigger connect via Prism API for {config['project']}: {e}")

        except Exception as e:
            logger.warning(f"[MCP-Reg] Auto-registration failed: {e}")


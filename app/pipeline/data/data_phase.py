"""
Data Phase -- orchestrates all collectors + processors before analysis.

Runs on the PC. Collects raw data into PostgreSQL, then processes it.
No LLM calls. Pure data acquisition + deterministic computation.

ORDER OF OPERATIONS:
  1. Global collectors (News, Reddit, YouTube, Congress, FRED, CoinGecko, SEC)
  2. Ticker Discovery (scan scraped data for ticker mentions)
  3. Merge discovered tickers into watchlist tickers
  4. Per-ticker collection (yfinance, finnhub, ticker-specific reddit/youtube/news)
  5. Deduplication
  6. Technical processors

Usage:
    await run(["NVDA", "AAPL"], emit=my_callback)
"""

import time
import asyncio
import logging
from typing import Callable
from app.pipeline.data.collection_scheduler import should_collect, record_collection
from app.pipeline.orchestration.cycle_control import cycle_control
from app.utils.pipeline_utils import noop as _noop, elapsed_ms
from app.monitoring.pipeline_profiler import profiler as pipeline_profiler

logger = logging.getLogger(__name__)

# Timeout for individual source coroutines (seconds).
# Prevents a single hanging API call from blocking the semaphore indefinitely.
SOURCE_TIMEOUT = 120.0


async def run(
    tickers: list[str],
    emit: Callable | None = None,
    analysis_queue: asyncio.Queue | None = None,
    force_global: bool | None = None,
    position_tickers: list[str] | None = None,
    triage_data: dict | None = None,
    max_tickers: int | None = None,
) -> dict:
    """
    Full data collection + processing for a list of tickers.

    Order: Global collect → Discovery → Merge → Per-ticker collect → Dedup → Technicals

    Args:
        tickers: list of ticker symbols (from watchlist)
        emit: callback(phase, step, detail, status, data, elapsed_ms)
        analysis_queue: if provided, tickers are pushed here after collection
                        so analysis can start in parallel (pipelining mode)
        force_global: if True, force all global collectors to re-run even if fresh
                      (used in discovery-only mode when watchlist is empty)
        position_tickers: tickers that represent open portfolio positions;
                          these are NEVER dropped by the internal analysis cap.
        triage_data: dict with 'glance', 'standard', 'deep' lists from ticker_triage.
                     Glance tickers skip per-ticker collection; Deep tickers force re-collection.
        max_tickers: user-specified cap on total non-position tickers to process.
                     When set, overrides settings.MAX_ANALYSIS_TICKERS.
    """
    if emit is None:
        emit = _noop

    # Parse triage data into sets for O(1) lookups
    _glance_set: set[str] = set(triage_data.get("glance", [])) if triage_data else set()
    _deep_set: set[str] = set(triage_data.get("deep", [])) if triage_data else set()

    # Auto-force global collectors when watchlist is empty (discovery-only mode)
    if not tickers and force_global is None:
        force_global = True
        logger.info(
            "[PIPELINE] [DATA PHASE] Discovery-only mode — forcing all global collectors"
        )
    elif force_global is None:
        force_global = False

    start = time.monotonic()
    results = {"tickers": tickers, "collectors": {}, "processors": {}}

    # Observability summary for cycle diagnosis (consumed by PipelineService)
    _summary = {
        "collector_ok": 0,
        "collector_skipped": 0,
        "collector_error": 0,
        "failed_collectors": [],
    }

    from app.config import settings
    if getattr(settings, "EXECUTION_MODE", "production") == "simulation":
        logger.info("[PIPELINE] [DATA PHASE] Running in SIMULATION mode.")
        emit("collecting", "simulation_start", f"Running World Simulator for {len(tickers)} tickers...", status="running")
        
        # 1. Generate the simulated data
        from app.pipeline.data.world_simulator import generate_simulated_world
        generate_simulated_world(tickers, settings.SIMULATION_TREND, settings.SIMULATION_NEWS_SENTIMENT)
        
        # 2. Push tickers to analysis queue if provided
        if analysis_queue is not None:
            for t in tickers:
                analysis_queue.put_nowait(t)
            logger.info("[PIPELINE] Pushed all %d tickers to analysis queue.", len(tickers))
            
        _total_ms = int((time.monotonic() - start) * 1000)
        results["total_ms"] = _total_ms
        results["collector_ok"] = len(tickers)
        
        emit("collecting", "complete", f"Data simulation complete: {_total_ms / 1000:.1f}s total, {len(tickers)} tickers", data={
            "total_ms": _total_ms,
            "collector_count": 1,
            "processor_count": 0,
            "tickers": tickers,
        })
        return results

    logger.info(f"[PIPELINE] \n{'=' * 60}")
    logger.info(
        f"[PIPELINE] DATA PHASE: {len(tickers)} watchlist tickers"
        + (" (DISCOVERY-ONLY — forcing collectors)" if force_global else "")
    )
    logger.info(f"[PIPELINE] {'=' * 60}")

    num_t = len(tickers)
    # Determine the intensity of the global collection scrape based on ticker volume
    if num_t == 0:
        intensity = "full"
    elif num_t <= 3:
        intensity = "micro"
    elif num_t <= 10:
        intensity = "light"
    else:
        intensity = "full"

    logger.info(
        f"[PIPELINE] Global Collection intensity: {intensity} (target volume: {num_t})"
    )

    from app.services.api_rate_limiter import rate_limiter
    if intensity == "micro":
        rate_limiter.enable_burst_mode(True)
    else:
        rate_limiter.enable_burst_mode(False)

    # ═══════════════════════════════════════════════════════════
    # PASS 1.5 & 1.6: DATABASE CURATION & JANITOR (Background tasks)
    # Started early to run concurrently with global collection
    # ═══════════════════════════════════════════════════════════
    async def _run_curation_bg():
        await cycle_control.wait_if_paused()
        logger.info("[PIPELINE] \n--- Pass 1.5: Database Curation (Background) ---")
        async with pipeline_profiler.phase("pass1_5_curation"):
            try:
                from app.pipeline.database_curator import run_data_curation
                curation_metrics = await run_data_curation(emit=emit)
                results["processors"]["data_curation"] = curation_metrics
            except Exception as e:
                logger.error(f"[PIPELINE]   [Curation] Failed: {e}")

    async def _run_janitor_bg():
        await cycle_control.wait_if_paused()
        logger.info("[PIPELINE] \n--- Pass 1.6: Data Janitor (Background) ---")
        async with pipeline_profiler.phase("pass1_6_janitor"):
            try:
                from app.pipeline.data.data_janitor import run_data_janitor
                janitor_metrics = await run_data_janitor(emit=emit, tickers=list(tickers))
                results["processors"]["data_janitor"] = janitor_metrics
            except Exception as e:
                logger.error(f"[PIPELINE]   [Janitor] Failed: {e}")
    curation_task = None
    janitor_task = None
    if len(tickers) == 1:
        logger.info("[PIPELINE] [DATA PHASE] Single-ticker mode detected: skipping database curation")
    else:
        curation_task = asyncio.create_task(_run_curation_bg())
        # Simple janitor background task disabled to prevent duplicate LLM calls; smart_janitor handles filtering and extraction
        # janitor_task = asyncio.create_task(_run_run_janitor_bg())

    from app.pipeline.data.data_global_collection import run_global_collection
    from app.config import settings

    _effective_cap = max_tickers if max_tickers is not None else settings.MAX_ANALYSIS_TICKERS
    _protected = set(position_tickers) if position_tickers else set()

    # ═══════════════════════════════════════════════════════════
    # STREAMING ARCHITECTURE: Two parallel tracks
    #
    # Track A (background): Global collection → Discovery → Merge
    #   Finds NEW tickers and pushes them to the analysis queue.
    #   RSS can take 10+ minutes — we don't block on it.
    #
    # Track B (foreground): Per-ticker collection for WATCHLIST tickers
    #   Starts immediately. Watchlist tickers have cached data from
    #   prior cycles. Analysis workers are already consuming them.
    #
    # Both tracks push tickers to the analysis_queue. Dedup in the
    # analysis workers (phase4_analysis.py _seen_tickers) prevents
    # double-processing.
    # ═══════════════════════════════════════════════════════════

    # Shared list for discovered tickers to merge after both tracks complete
    _discovered_final: list[str] = []
    _discovery_merged_tickers: list[str] = []

    async def _track_a_global_and_discovery():
        """Track A: Global collection → Discovery → Merge → push new tickers to queue."""
        nonlocal tickers

        logger.info("[PIPELINE] [TRACK A] Starting global collection + discovery (background)")
        emit(
            "collecting",
            "track_a_start",
            "Background track: global collection → discovery → merge",
            status="running",
        )

        # ── Pass 1: Global Collection ──
        await run_global_collection(
            tickers=tickers,
            force_global=force_global,
            intensity=intensity,
            emit=emit,
            results=results,
            _summary=_summary,
        )

        # ── Pass 2: Ticker Discovery ──
        from app.pipeline.data.data_ticker_discovery import run_ticker_discovery_and_gates

        discovered_tickers = await run_ticker_discovery_and_gates(
            tickers=list(tickers),
            discovered_tickers=[],
            emit=emit,
            results=results,
            _summary=_summary,
        )

        # ── Pass 3: Merge discovered tickers ──
        original_tickers = list(tickers)
        _at_cap = len(tickers) >= _effective_cap

        if _at_cap and discovered_tickers:
            logger.info(
                "[PIPELINE]   [merge] Skipping discovery merge — already at hard cap "
                "(%d total tickers >= %d cap). %d discovered tickers "
                "saved to DB for future cycles.",
                len(tickers),
                _effective_cap,
                len(discovered_tickers),
            )
            emit(
                "collecting",
                "merge",
                f"Skipping discovery merge — already at hard cap ({len(tickers)}/{_effective_cap}). "
                f"{len(discovered_tickers)} tickers saved for future cycles.",
                status="ok",
                data={
                    "skipped": True,
                    "discovered_count": len(discovered_tickers),
                    "cap": _effective_cap,
                    "current_total": len(tickers),
                },
            )
        elif discovered_tickers:
            # Add discovered tickers not already in watchlist
            new_tickers = [t for t in discovered_tickers if t not in tickers]
            if new_tickers:
                _discovery_merged_tickers.extend(new_tickers)
                emit(
                    "collecting",
                    "merge",
                    f"Merged {len(new_tickers)} discovered tickers: {', '.join(new_tickers[:15])}",
                    status="ok",
                    data={
                        "original": original_tickers,
                        "added": new_tickers,
                        "total": len(tickers) + len(new_tickers),
                    },
                )
                logger.info(
                    f"[PIPELINE]   [merge] Added {len(new_tickers)} discovered tickers"
                )

                # ── Push newly discovered tickers to analysis queue ──
                # These tickers are NEW (not on watchlist), so per-ticker
                # collection hasn't run for them yet. Push them so analysis
                # workers can start assessing them with whatever DB data exists.
                # The V2 pipeline's data_completeness check will fill gaps.
                if analysis_queue is not None:
                    # Apply cap before pushing
                    all_tickers = list(tickers) + new_tickers
                    if len(all_tickers) > _effective_cap:
                        # Protect positions, cap the rest
                        protected_kept = [t for t in all_tickers if t in _protected]
                        unprotected = [t for t in all_tickers if t not in _protected]
                        remaining_slots = max(0, _effective_cap - len(protected_kept))
                        # Prefer watchlist tickers over discovered
                        watchlist_set = set(original_tickers) - _protected
                        wl_from_watchlist = [t for t in unprotected if t in watchlist_set]
                        others = [t for t in unprotected if t not in watchlist_set]
                        capped_wl = wl_from_watchlist[:remaining_slots]
                        remaining_after_wl = remaining_slots - len(capped_wl)
                        capped_unprotected = capped_wl + others[:max(0, remaining_after_wl)]
                        capped_new = [t for t in capped_unprotected if t in new_tickers]
                        dropped = len(new_tickers) - len(capped_new)
                        if dropped:
                            emit(
                                "collecting",
                                "ticker_cap",
                                f"Hard cap enforced at {_effective_cap} total: "
                                f"kept {len(protected_kept)} positions + "
                                f"{len(capped_unprotected)} others, dropped {dropped} discovered",
                                status="ok",
                                data={
                                    "cap": _effective_cap,
                                    "positions_kept": len(protected_kept),
                                    "dropped": dropped,
                                },
                            )
                        new_tickers = capped_new

                    from app.cycle.orchestration.priority_queue import PRIORITY_DISCOVERED
                    for t in new_tickers:
                        # Use hasattr check for PriorityAnalysisQueue vs plain Queue
                        if hasattr(analysis_queue, 'put_nowait') and hasattr(analysis_queue, 'classify'):
                            analysis_queue.put_nowait(t, priority=PRIORITY_DISCOVERED)
                        else:
                            analysis_queue.put_nowait(t)
                    logger.info(
                        "[PIPELINE] [TRACK A] Pushed %d newly discovered tickers "
                        "to analysis queue at priority=%d (dedup filters duplicates)",
                        len(new_tickers), PRIORITY_DISCOVERED,
                    )
                    emit(
                        "analyzing",
                        "discovery_push",
                        f"Pushed {len(new_tickers)} discovered tickers to analysis queue",
                        status="ok",
                        data={"count": len(new_tickers), "tickers": new_tickers[:20]},
                    )

        _discovered_final.extend(discovered_tickers or [])
        logger.info("[PIPELINE] [TRACK A] Global collection + discovery complete")

    async def _track_b_perticker():
        """Track B: Per-ticker collection for watchlist tickers (starts immediately)."""
        logger.info(
            "[PIPELINE] [TRACK B] Starting per-ticker collection for %d watchlist tickers (immediate)",
            len(tickers),
        )
        emit(
            "collecting",
            "track_b_start",
            f"Foreground track: per-ticker collection for {len(tickers)} watchlist tickers (starts NOW)",
            status="running",
        )

        # ── Pass 3.5: GATHER METADATA & INSTITUTIONAL DATA ──
        # Run metadata enrichment for watchlist tickers before per-ticker collection
        _meta_task = None
        if tickers:
            try:
                from app.graph.sector_collector import collect_metadata

                _meta_task = asyncio.create_task(collect_metadata(list(tickers)))
                _meta_task.add_done_callback(
                    lambda t: (
                        logger.error(
                            "[PIPELINE]   [Metadata] Background task FAILED: %s",
                            t.exception(),
                        )
                        if t.exception()
                        else None
                    )
                )
                logger.info(
                    f"[PIPELINE]   [Metadata] Launched background enrichment for {len(tickers)} tickers"
                )
            except Exception as e:
                logger.info(f"[PIPELINE]   [Metadata] Enrichment skipped: {e}")

        # Global institutional holders
        async with pipeline_profiler.phase("pass3_5_institutional"):
            try:
                from app.collectors.sec_collector import collect_all_tickers_institutional

                if tickers and should_collect("institutional"):
                    emit(
                        "collecting",
                        "institutional",
                        "Fetching institutional holders...",
                        status="running",
                    )
                    inst = await collect_all_tickers_institutional(list(tickers))
                    results["collectors"]["institutional_yf"] = inst
                    total = sum(inst.values())
                    record_collection("institutional", rows=total)
                    logger.info(
                        f"[PIPELINE]   [Institutional] {total} holders across {len(tickers)} tickers"
                    )
                    emit(
                        "collecting",
                        "institutional",
                        f"Got {total} institutional holders across {len(tickers)} tickers",
                        status="ok",
                    )
                elif tickers:
                    emit(
                        "collecting",
                        "institutional",
                        "Institutional: fresh, skipped",
                        status="skipped",
                    )
                    logger.debug("[PIPELINE]   [Institutional] fresh, skipping")
            except Exception as e:
                logger.info(f"[PIPELINE]   [Institutional] skipped: {e}")
                emit(
                    "collecting",
                    "institutional",
                    f"Institutional skipped: {e}",
                    status="error",
                )

        # Await metadata enrichment before starting per-ticker collection
        if _meta_task is not None:
            try:
                await asyncio.wait_for(_meta_task, timeout=30.0)
                logger.info("[PIPELINE]   [Metadata] Enrichment complete before Pass 4")
            except asyncio.TimeoutError:
                logger.warning(
                    "[PIPELINE]   [Metadata] Enrichment timed out (30s), proceeding"
                )
            except Exception as e:
                logger.warning("[PIPELINE]   [Metadata] Enrichment failed: %s", e)

        # ── Per-ticker collection ──
        from app.pipeline.data.data_perticker_collection import run_perticker_collection

        await run_perticker_collection(
            tickers=list(tickers),
            _glance_set=_glance_set,
            _deep_set=_deep_set,
            emit=emit,
            results=results,
            _summary=_summary,
            analysis_queue=analysis_queue,
        )
        logger.info("[PIPELINE] [TRACK B] Per-ticker collection complete")

    # NOTE: Per-ticker summarization + consensus is handled by
    # run_ticker_processors() in data_perticker_collection.py (lines 23-45)
    # BEFORE each ticker is pushed to the analysis queue.
    # No global background loop needed — this was causing DGX Spark to sit
    # idle for 30+ minutes waiting for redundant summarization to finish.

    # ═══════════════════════════════════════════════════════════
    # RUN BOTH TRACKS IN PARALLEL
    # Track A: Global collection → Discovery → Merge (background)
    # Track B: Per-ticker collection (starts immediately)
    # ═══════════════════════════════════════════════════════════
    emit(
        "collecting",
        "parallel_start",
        f"Starting parallel tracks: global+discovery (background) + "
        f"per-ticker collection for {len(tickers)} watchlist tickers (immediate)",
        status="running",
    )

    if len(tickers) == 1:
        logger.info("[PIPELINE] [DATA PHASE] Single-ticker mode detected: skipping background global collection and discovery (Track A)")
        await _track_b_perticker()
    else:
        await asyncio.gather(
            _track_a_global_and_discovery(),
            _track_b_perticker(),
        )

    # Merge discovered tickers into the main ticker list for downstream phases
    if _discovery_merged_tickers:
        merged_set = set(tickers)
        for t in _discovery_merged_tickers:
            if t not in merged_set:
                tickers = list(tickers) + [t]
                merged_set.add(t)

    # ── Safety net: guarantee TOTAL invariant regardless of upstream bugs ──
    if len(tickers) > _effective_cap:
        logger.warning(
            "[PIPELINE]   [SAFETY NET] Total ticker count %d exceeds hard cap %d — "
            "hard-truncating to enforce invariant",
            len(tickers),
            _effective_cap,
        )
        _kept_positions = [t for t in tickers if t in _protected][:_effective_cap]
        _remaining = _effective_cap - len(_kept_positions)
        _kept_non_pos = [t for t in tickers if t not in _protected][:_remaining]
        tickers = _kept_positions + _kept_non_pos
    # ═══════════════════════════════════════════════════════════
    # Wait for background tasks before Deduplication
    # ═══════════════════════════════════════════════════════════
    bg_tasks = [t for t in [curation_task, janitor_task] if t is not None]
    if bg_tasks:
        logger.info("[PIPELINE]   Waiting for background curation/janitor tasks...")
        await asyncio.gather(*bg_tasks)

    # ═══════════════════════════════════════════════════════════
    # PASS 5: DEDUPLICATION
    # ═══════════════════════════════════════════════════════════
    logger.info("[PIPELINE] \n--- Pass 5: Global Deduplication ---")
    await asyncio.sleep(0)  # Cancellation checkpoint
    t0 = time.monotonic()
    try:
        from app.processors.deduplicator import deduplicate_news

        removed, highly_redundant_tickers = deduplicate_news()
        ms = elapsed_ms(t0)
        results["processors"]["deduplicated_news"] = removed
        results["processors"]["highly_redundant_tickers"] = highly_redundant_tickers
        emit(
            "collecting",
            "dedup",
            f"Removed {removed} duplicate news articles",
            status="ok",
            data={"removed": removed, "highly_redundant": len(highly_redundant_tickers)},
            elapsed_ms=ms,
        )
    except Exception as e:
        ms = elapsed_ms(t0)
        emit(
            "collecting", "dedup", f"Dedup failed — {e}", status="error", elapsed_ms=ms
        )
        logger.info(f"[PIPELINE]   [dedup] FAILED: {e}")

    # NOTE: Pass 5.5 (background summarization) and Pass 5.6 (consensus engine)
    # have been removed. Both are already handled per-ticker by
    # run_ticker_processors() in data_perticker_collection.py before each
    # ticker is pushed to the analysis queue. Running them again globally here
    # was purely redundant and caused the DGX Spark to sit idle for 30+ minutes.

    # ═══════════════════════════════════════════════════════════
    # PASS 6: PROCESSORS FINALIZE
    # ═══════════════════════════════════════════════════════════
    logger.info("[PIPELINE] \n--- Pass 6: Processors Finalize ---")
    # Technicals and institutional data are now streamed per-ticker.
    # Nothing globally blocking here anymore!

    _total_ms = elapsed_ms(start)
    results["total_ms"] = _total_ms
    results["tickers"] = tickers  # Update with merged list

    # Merge observability summary into results for PipelineService
    results["collector_ok"] = _summary["collector_ok"]
    results["collector_skipped"] = _summary["collector_skipped"]
    results["collector_error"] = _summary["collector_error"]
    results["failed_collectors"] = _summary["failed_collectors"]

    emit(
        "collecting",
        "complete",
        f"Data collection complete: {_total_ms / 1000:.1f}s total, {len(tickers)} tickers",
        data={
            "total_ms": _total_ms,
            "collector_count": len(results["collectors"]),
            "processor_count": len(results["processors"]),
            "tickers": tickers,
        },
    )

    logger.info(
        "[PIPELINE] DATA PHASE COMPLETE | %d tickers | OK=%d Skip=%d Fail=%d | %dms",
        len(tickers),
        _summary["collector_ok"],
        _summary["collector_skipped"],
        _summary["collector_error"],
        _total_ms,
    )

    return results

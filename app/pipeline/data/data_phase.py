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

    curation_task = asyncio.create_task(_run_curation_bg())
    janitor_task = asyncio.create_task(_run_janitor_bg())

    from app.pipeline.data.data_global_collection import run_global_collection

    await run_global_collection(
        tickers=tickers,
        force_global=force_global,
        intensity=intensity,
        emit=emit,
        results=results,
        _summary=_summary,
    )

    # ═══════════════════════════════════════════════════════════
    # PASS 2: TICKER DISCOVERY
    # ═══════════════════════════════════════════════════════════
    from app.config import settings

    _effective_cap = max_tickers if max_tickers is not None else settings.MAX_ANALYSIS_TICKERS
    _protected = set(position_tickers) if position_tickers else set()

    from app.pipeline.data.data_ticker_discovery import run_ticker_discovery_and_gates

    discovered_tickers = await run_ticker_discovery_and_gates(
        tickers=list(tickers),
        discovered_tickers=[],
        emit=emit,
        results=results,
        _summary=_summary,
    )
    # ═══════════════════════════════════════════════════════════
    # PASS 3: MERGE — combine watchlist + discovered tickers
    # Only merge discovered tickers if we have room under the cap.
    # When max_tickers is small (e.g. 1), the selector already filled
    # the cap, so we skip merging to avoid processing 50-100 extra.
    # Discovery still RUNS (populates discovered_tickers DB table for
    # future cycles) — we just don't merge them into THIS cycle.
    #
    # NOTE: _effective_cap is a HARD TOTAL ceiling (positions + non-positions),
    # not just a non-position cap. This matches the ticker_selector behavior.
    # ═══════════════════════════════════════════════════════════

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
            tickers = list(tickers) + new_tickers
            emit(
                "collecting",
                "merge",
                f"Merged {len(new_tickers)} discovered tickers: {', '.join(new_tickers[:15])}",
                status="ok",
                data={
                    "original": original_tickers,
                    "added": new_tickers,
                    "total": len(tickers),
                },
            )
            logger.info(
                f"[PIPELINE]   [merge] Added {len(new_tickers)} discovered tickers"
            )
            logger.info(
                f"[PIPELINE]   [merge] Full ticker list ({len(tickers)}): {', '.join(tickers)}"
            )

    # Apply ticker cap: protect positions, cap total including positions.
    # IMPORTANT: Open portfolio positions get priority but still count against the cap.
    if len(tickers) > _effective_cap:
        # Split into protected (positions) and unprotected
        protected_kept = [t for t in tickers if t in _protected]
        unprotected = [t for t in tickers if t not in _protected]

        # Positions fill the cap first; remaining slots go to non-position tickers
        remaining_slots = max(0, _effective_cap - len(protected_kept))

        # Among unprotected, prefer original watchlist over discovered.
        watchlist_set = set(original_tickers) - _protected
        wl_from_watchlist = [t for t in unprotected if t in watchlist_set]
        others = [t for t in unprotected if t not in watchlist_set]

        # Cap watchlist tickers first, then fill remaining slots with discovered
        capped_wl = wl_from_watchlist[:remaining_slots]
        remaining_after_wl = remaining_slots - len(capped_wl)
        capped_unprotected = capped_wl + others[: max(0, remaining_after_wl)]

        before = len(tickers)
        tickers = protected_kept + capped_unprotected
        dropped = before - len(tickers)
        emit(
            "collecting",
            "ticker_cap",
            f"Hard cap enforced at {_effective_cap} total: "
            f"kept {len(protected_kept)} positions + "
            f"{len(capped_unprotected)} others, dropped {dropped}",
            status="ok",
            data={
                "cap": _effective_cap,
                "positions_kept": len(protected_kept),
                "kept": len(tickers),
                "dropped": dropped,
            },
        )
        logger.info(
            "[PIPELINE]   [cap] Hard cap at %d total tickers "
            "(positions: %d, non-position: %d, dropped %d)",
            _effective_cap,
            len(protected_kept),
            len(capped_unprotected),
            dropped,
        )

    # ── Safety net: guarantee TOTAL invariant regardless of upstream bugs ──
    if len(tickers) > _effective_cap:
        logger.warning(
            "[PIPELINE]   [SAFETY NET] Total ticker count %d exceeds hard cap %d — "
            "hard-truncating to enforce invariant",
            len(tickers),
            _effective_cap,
        )
        # Positions first, then non-position, but total never exceeds cap
        _kept_positions = [t for t in tickers if t in _protected][:_effective_cap]
        _remaining = _effective_cap - len(_kept_positions)
        _kept_non_pos = [t for t in tickers if t not in _protected][:_remaining]
        tickers = _kept_positions + _kept_non_pos

    # ═══════════════════════════════════════════════════════════
    # PASS 3.5: GATHER METADATA & INSTITUTIONAL DATA
    # ═══════════════════════════════════════════════════════════
    # 1. Ensure all tickers have sector/industry metadata for peer comparison.
    #    Awaited before Pass 4 so downstream analysis has sector data (Fix #7).
    #    Task reference stored for cleanup on shutdown (Fix #5).
    _meta_task = None
    if tickers:
        try:
            from app.graph.sector_collector import collect_metadata

            _meta_task = asyncio.create_task(collect_metadata(tickers))
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

    # 2. Global institutional holders (Fix #6: freshness gate + profiler wrapping)
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
                inst = await collect_all_tickers_institutional(tickers)
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

    # 3. Await metadata enrichment before starting per-ticker collection (Fix #7)
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

    # ═══════════════════════════════════════════════════════════
    # Start Continuous Summarization Background Task
    # ═══════════════════════════════════════════════════════════
    _collection_done = False
    
    async def _continuous_summarization():
        logger.info("[PIPELINE] Started continuous background summarization...")
        summ_stats = {"youtube": 0, "reddit": 0, "news": 0, "tokens": 0, "ms": 0}
        from app.processors.summarizer import summarize_unsummarized
        
        # Scale chunk size to match adaptive concurrency limit
        # (no point querying 30 articles if we can only process 8 at once)
        try:
            from app.services.adaptive_concurrency import concurrency_controller
            chunk_size = concurrency_controller.current_limit
        except Exception:
            chunk_size = 5 if intensity == "micro" else 15
        
        first_run = True
        while not _collection_done or first_run:
            first_run = False
            try:
                stats = await summarize_unsummarized(emit=emit, max_items=chunk_size)
                total_done = stats.get("youtube", 0) + stats.get("reddit", 0) + stats.get("news", 0)
                
                if total_done > 0:
                    for k in summ_stats:
                        summ_stats[k] += stats.get(k, 0)
                else:
                    # Nothing to summarize right now, yield control
                    await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"[PIPELINE] [Continuous Summarizer] loop error: {e}")
                await asyncio.sleep(2)
                
        # Do one final sweep after collection completes just in case
        try:
            stats = await summarize_unsummarized(emit=emit, max_items=50)
            for k in summ_stats:
                summ_stats[k] += stats.get(k, 0)
        except Exception:
            pass
            
        return summ_stats

    summarizer_bg_task = asyncio.create_task(_continuous_summarization())

    from app.pipeline.data.data_perticker_collection import run_perticker_collection

    await run_perticker_collection(
        tickers=tickers,
        _glance_set=_glance_set,
        _deep_set=_deep_set,
        emit=emit,
        results=results,
        _summary=_summary,
        analysis_queue=analysis_queue,
    )
    
    # Signal summarizer to finish up
    _collection_done = True
    # ═══════════════════════════════════════════════════════════
    # Wait for background tasks before Deduplication
    # ═══════════════════════════════════════════════════════════
    logger.info("[PIPELINE]   Waiting for background Curation & Janitor tasks...")
    await asyncio.gather(curation_task, janitor_task)

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

    # ═══════════════════════════════════════════════════════════
    # PASS 5.5: DATA SUMMARIZATION (Await Background Task)
    # ═══════════════════════════════════════════════════════════
    logger.info("[PIPELINE] \n--- Pass 5.5: Await Continuous Summarization ---")
    await asyncio.sleep(0)  # Cancellation checkpoint
    try:
        summ_stats = await summarizer_bg_task
        results["processors"]["summarization"] = summ_stats
    except Exception as e:
        logger.info(f"[PIPELINE]   [summarizer] FAILED: {e}")
        emit("collecting", "summarize", f"Summarization failed — {e}", status="error")

    # ═══════════════════════════════════════════════════════════
    # PASS 5.6: CONSENSUS ENGINE (LLM-powered)
    # Extracts consensus across articles and flags outliers.
    # ═══════════════════════════════════════════════════════════
    logger.info("[PIPELINE] \n--- Pass 5.6: Consensus Engine ---")
    await asyncio.sleep(0)  # Cancellation checkpoint
    try:
        from app.processors.consensus_engine import run_consensus_engine

        consensus_stats = await run_consensus_engine(emit=emit)
        results["processors"]["consensus"] = consensus_stats
    except Exception as e:
        logger.info(f"[PIPELINE]   [consensus] FAILED: {e}")
        emit("collecting", "consensus", f"Consensus failed — {e}", status="error")

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

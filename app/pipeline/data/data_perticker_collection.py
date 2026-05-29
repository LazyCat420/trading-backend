import time
import asyncio
import datetime
import logging
from typing import Callable
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from app.pipeline.orchestration.cycle_control import cycle_control
from app.monitoring.pipeline_profiler import profiler as pipeline_profiler
from app.pipeline.data.collection_scheduler import record_collection
from app.utils.pipeline_utils import elapsed_ms
from app.config import settings

logger = logging.getLogger(__name__)

SOURCE_TIMEOUT = 120.0


async def run_smart_janitor_on_ticker_data(ticker: str) -> None:
    """Find all un-janited news and reddit posts for this ticker and process them."""
    ticker = ticker.upper()
    try:
        from app.processors.smart_janitor import run_smart_janitor_for_article, run_smart_janitor_for_reddit
        from app.db.connection import get_db
        
        # 1. Fetch un-janited news articles
        with get_db() as db:
            news_rows = db.execute(
                "SELECT id FROM news_articles WHERE ticker = %s AND qualitative_draft IS NULL ORDER BY published_at DESC LIMIT 50",
                [ticker]
            ).fetchall()
            article_ids = [r[0] for r in news_rows]
            
        if article_ids:
            logger.info("[PIPELINE] Running Smart Janitor on %d news articles for %s", len(article_ids), ticker)
            tasks = [run_smart_janitor_for_article(aid) for aid in article_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # 2. Fetch un-janited Reddit posts
        with get_db() as db:
            reddit_rows = db.execute(
                "SELECT id FROM reddit_posts WHERE ticker = %s AND qualitative_draft IS NULL ORDER BY created_utc DESC LIMIT 30",
                [ticker]
            ).fetchall()
            post_ids = [r[0] for r in reddit_rows]
            
        if post_ids:
            logger.info("[PIPELINE] Running Smart Janitor on %d Reddit posts for %s", len(post_ids), ticker)
            tasks = [run_smart_janitor_for_reddit(pid) for pid in post_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            
    except Exception as e:
        logger.warning("[PIPELINE] Smart Janitor batch run failed for %s: %s", ticker, e)


async def run_ticker_processors(ticker: str, emit) -> None:
    """Run per-ticker deduplication, summarization, and consensus."""
    # -- Pre-process raw text with Smart Janitor --
    await run_smart_janitor_on_ticker_data(ticker)

    # ── Per-ticker deduplication ──
    try:
        from app.processors.deduplicator import deduplicate_news
        # Run in thread pool to not block
        await asyncio.to_thread(deduplicate_news, ticker)
    except Exception as e:
        logger.warning("[PIPELINE] Per-ticker deduplication failed for %s: %s", ticker, e)

    # ── Per-ticker summarization ──
    try:
        from app.processors.summarizer import summarize_unsummarized
        await summarize_unsummarized(emit=emit, max_items=50, ticker=ticker)
    except Exception as e:
        logger.warning("[PIPELINE] Per-ticker summarization failed for %s: %s", ticker, e)

    # ── Per-ticker consensus ──
    try:
        from app.processors.consensus_engine import run_consensus_engine
        await run_consensus_engine(emit=emit, ticker=ticker)
    except Exception as e:
        logger.warning("[PIPELINE] Per-ticker consensus failed for %s: %s", ticker, e)

    # ── Per-ticker narrative curation ──
    try:
        from app.processors.narrative_curator import update_company_narrative
        await update_company_narrative(ticker=ticker)
    except Exception as e:
        logger.warning("[PIPELINE] Per-ticker narrative curation failed for %s: %s", ticker, e)


async def run_perticker_collection(
    tickers: list[str],
    _glance_set: set[str],
    _deep_set: set[str],
    emit: Callable,
    results: dict,
    _summary: dict,
    analysis_queue: asyncio.Queue | None = None,
):
    # ═══════════════════════════════════════════════════════════
    # PASS 4: PER-TICKER COLLECTION (watchlist + discovered)
    #   Now runs up to COLLECTION_MAX_CONCURRENT tickers in parallel.
    #   If analysis_queue is provided, tickers are pushed there
    #   as they finish so analysis starts immediately (pipelining).
    # ═══════════════════════════════════════════════════════════
    await cycle_control.wait_if_paused()

    concurrency = settings.COLLECTION_MAX_CONCURRENT
    logger.info(
        f"[PIPELINE] \n--- Pass 4: Per-Ticker Collection ({len(tickers)} tickers, {concurrency}x parallel) ---"
    )
    emit(
        "collecting",
        "pass4_perticker",
        f"Collecting data for {len(tickers)} tickers ({concurrency}x parallel): {', '.join(tickers)}",
        status="running",
    )

    # ── Final safety gate: remove any FALSE_TICKERS that leaked through ──
    from app.processors.ticker_extractor import (
        _is_hard_blocked,
        get_registry as _get_reg,
    )

    _reg = _get_reg()
    pre_gate = len(tickers)
    tickers = [
        t for t in tickers if not _is_hard_blocked(t, _reg) and not _reg.is_rejected(t)
    ]
    if len(tickers) < pre_gate:
        dropped = pre_gate - len(tickers)
        logger.info(
            f"[PIPELINE]   [safety] Dropped {dropped} FALSE_TICKERS/rejected before per-ticker collection"
        )
        emit(
            "collecting",
            "safety_gate",
            f"Dropped {dropped} false-positive tickers before collection",
            status="ok",
            data={"dropped": dropped},
        )

    # Semaphore limits concurrent per-ticker scrapers
    sem = asyncio.Semaphore(concurrency)
    results_lock = asyncio.Lock()
    utility_lock = asyncio.Lock()

    async def _collect_single_ticker(ticker: str) -> None:
        # ── Queue Watermark & Utility Mode Check ──
        if analysis_queue is not None:
            high_wm = getattr(settings, "PIPELINE_QUEUE_HIGH_WATERMARK", 200)
            low_wm = getattr(settings, "PIPELINE_QUEUE_LOW_WATERMARK", 100)

            while analysis_queue.qsize() >= high_wm:
                if not utility_lock.locked():
                    async with utility_lock:
                        logger.info(
                            f"[PIPELINE] Queue high watermark ({analysis_queue.qsize()} >= {high_wm}). Running Utility Mode."
                        )
                        try:
                            from app.pipeline.data.utility_worker import (
                                run_utility_cycle,
                            )

                            await run_utility_cycle(emit)
                        except Exception as e:
                            logger.error(f"[PIPELINE] Utility error: {e}")
                            await asyncio.sleep(5)

                        logger.info(
                            f"[PIPELINE] Waiting for queue to drain below {low_wm}..."
                        )
                        while analysis_queue.qsize() > low_wm:
                            await asyncio.sleep(2)
                else:
                    await asyncio.sleep(2)

        """Collect all data sources for a single ticker (semaphore-guarded).

        Sources run in PARALLEL within each ticker, gated by per-API
        rate limiters so we never exceed safe limits for any service.

        Fix #1: Each source wrapped in asyncio.wait_for(timeout=SOURCE_TIMEOUT).
        Fix #2: _src_yfinance uses tenacity retry for transient errors.
        FELL fix: If yfinance auto-rejects, sibling sources are cancelled.
        """
        from app.services.api_rate_limiter import rate_limiter

        def _log_err(src: str, err: Exception, t: str):
            import traceback

            try:
                from app.pipeline.orchestration.state_manager import PipelineStateDB

                PipelineStateDB.log_execution_error(
                    _summary.get("cycle_id", "unknown"),
                    f"collection_{src}",
                    t,
                    type(err).__name__,
                    str(err),
                    traceback.format_exc(),
                )
            except Exception:
                pass

        async with sem:
            await cycle_control.wait_if_paused()
            logger.info(
                f"[PIPELINE] \n  --- Collecting: {ticker} (parallel sources) ---"
            )
            ticker_start = time.monotonic()

            # ── TRIAGE: GLANCE TIER SKIP ──
            # Glance-tier tickers skip per-ticker collection entirely.
            # They'll get a lightweight change-detection check in the analysis phase.
            if ticker in _glance_set:
                # Emit per-source "skipped" events so frontend shows
                # data exists for each source (Glance = all data fresh)
                for _src_key in [
                    "yfinance",
                    "finnhub",
                    "reddit",
                    "youtube",
                    "yfnews",
                ]:
                    emit(
                        "collecting",
                        f"{_src_key}_{ticker}",
                        f"{ticker}: Glance tier (data fresh)",
                        status="skipped",
                    )
                emit(
                    "collecting",
                    f"glance_skip_{ticker}",
                    f"{ticker}: Glance tier — skipping collection (data fresh)",
                    status="skipped",
                )
                logger.info(
                    "[PIPELINE]   [triage] %s skipped collection (Glance tier)", ticker
                )
                # Still compute technicals from cached price data
                try:
                    from app.processors.technical_processor import compute_technicals

                    compute_technicals(ticker)
                except Exception:
                    pass
                if analysis_queue is not None:
                    await analysis_queue.put(ticker)
                    logger.info("[PIPELINE] %s (Glance) queued for analysis immediately", ticker)
                return ticker
            # ── END TRIAGE SKIP ──

            # ── DATA SUFFICIENCY GATE (Smart Pipeline Phase 2) ──
            from app.pipeline.data.data_sufficiency import check_data_sufficiency
            
            _is_deep = ticker in _deep_set
            _is_sufficient = False
            if not _is_deep:
                _is_sufficient = check_data_sufficiency(ticker, hours=48, threshold=5)
                if _is_sufficient:
                    logger.info(f"[PIPELINE]   [Sufficiency] {ticker} has sufficient high-quality data. Bypassing news/social scraping.")
                    emit(
                        "collecting",
                        f"sufficiency_{ticker}",
                        f"{ticker}: Sufficient high-quality data found. Bypassing scraping.",
                        status="skipped",
                    )
            
            # ── FAST PATH: DATA COMPLETENESS GATE (Aggressive Caching) ──
            # Deep-tier tickers bypass the cache gate to force fresh collection
            from app.pipeline.data.collection_scheduler import should_collect

            if not _is_deep and (
                not should_collect("fundamentals", ticker)
                and (_is_sufficient or not should_collect("news_finnhub", ticker))
                and (_is_sufficient or not should_collect("news_yfinance", ticker))
                and (_is_sufficient or not should_collect("reddit", ticker))
                and (_is_sufficient or not should_collect("youtube", ticker))
            ):
                from app.db.connection import get_db as _get_db_cg

                with _get_db_cg() as _db_cg:
                    _p_count = _db_cg.execute(
                        "SELECT COUNT(*) FROM price_history WHERE ticker = %s", [ticker]
                    ).fetchone()[0]

                if _p_count >= 20:
                    # Emit per-source "skipped" events so frontend shows
                    # data exists for each source (otherwise x/6 reads 0/6)
                    for _src_key in [
                        "yfinance",
                        "finnhub",
                        "reddit",
                        "youtube",
                        "yfnews",
                    ]:
                        emit(
                            "collecting",
                            f"{_src_key}_{ticker}",
                            f"{ticker}: cached (data fresh)",
                            status="skipped",
                        )
                    emit(
                        "collecting",
                        f"cache_bypass_{ticker}",
                        f"{ticker}: Full cache hit. Bypassing APIs.",
                        status="skipped",
                    )
                    logger.info(
                        f"[PIPELINE]   [cache] {ticker} fully cached! Early bypass."
                    )

                    try:
                        from app.processors.technical_processor import (
                            compute_technicals,
                        )

                        compute_technicals(ticker)
                    except Exception:
                        pass

                    # Record attention even on cache bypass
                    try:
                        from app.pipeline.attention_tracker import (
                            record_collection as record_attention,
                        )

                        record_attention(ticker)
                    except Exception:
                        pass

                    if analysis_queue is not None:
                        await analysis_queue.put(ticker)
                        logger.info(
                            "[PIPELINE] %s queued directly from cache bypass", ticker
                        )
                    return ticker

            if _is_deep:
                logger.info(
                    "[PIPELINE]   [triage] %s in Deep tier — forcing full re-collection",
                    ticker,
                )
            # ── END FAST PATH ──

            # Local results for this ticker (each source writes independently)
            local = {}
            # Cancellation signal: set when yfinance detects a delisted ticker
            _ticker_rejected = asyncio.Event()

            # ── Source 1: yfinance (prices + fundamentals + financials + balance sheet) ──
            async def _src_yfinance():
                t0 = time.monotonic()
                try:
                    # Force collection when ticker has no price data yet,
                    # regardless of freshness gating.
                    from app.db.connection import get_db as _get_db_yf

                    with _get_db_yf() as _db_yf:
                        existing_prices = _db_yf.execute(
                            "SELECT COUNT(*) FROM price_history WHERE ticker = %s",
                            [ticker],
                        ).fetchone()[0]
                    needs_collection = existing_prices < 20 or should_collect(
                        "fundamentals", ticker
                    )

                    if needs_collection:
                        emit(
                            "collecting",
                            f"yfinance_{ticker}",
                            f"{ticker}: Fetching prices, fundamentals, financials...",
                            status="running",
                        )
                        from app.collectors.data_rotator import (
                            fetch_price_history,
                            fetch_fundamentals,
                            fetch_financials,
                            fetch_balance_sheet,
                        )

                        # Retry wrapper for transient network errors (Fix #2)
                        @retry(
                            stop=stop_after_attempt(3),
                            wait=wait_exponential(multiplier=2, min=2, max=30),
                            retry=retry_if_exception_type(
                                (ConnectionError, OSError, TimeoutError)
                            ),
                            reraise=True,
                        )
                        async def _fetch_yf_with_retry():
                            # Note: data_rotator handles its own limits inside the smart clients,
                            # but we still acquire the yfinance semaphore here as the primary target
                            async with rate_limiter.acquire("yfinance"):
                                p = await fetch_price_history(ticker)
                                f = await fetch_fundamentals(ticker)
                                fi = await fetch_financials(ticker)
                                b = await fetch_balance_sheet(ticker)
                            return p, f, fi, b

                        prices, fundies, fins, bs = await asyncio.wait_for(
                            _fetch_yf_with_retry(), timeout=SOURCE_TIMEOUT
                        )

                        # If yfinance returned 0 price rows AND we had
                        # very few existing rows, this ticker is likely delisted
                        # or untradeable. Auto-reject and cancel sibling sources.
                        if prices == 0 and existing_prices < 5:
                            logger.warning(
                                "[PIPELINE] [yfinance] %s: 0 price rows — "
                                "likely delisted/untradeable. Auto-rejecting.",
                                ticker,
                            )
                            emit(
                                "collecting",
                                f"yfinance_{ticker}",
                                f"{ticker}: NO PRICE DATA — likely delisted or untradeable",
                                status="error",
                            )
                            _ticker_rejected.set()  # Signal sibling tasks to stop
                            try:
                                from app.processors.ticker_extractor import (
                                    get_registry as _get_reg_yf,
                                    _save_rejected_to_db as _reject_db,
                                    FALSE_TICKERS as _FT,
                                )

                                _reg_yf = _get_reg_yf()
                                _reg_yf.add_rejected(ticker)
                                _FT.add(ticker)
                                _reject_db(ticker)
                            except Exception as rej_err:
                                logger.debug(
                                    "[PIPELINE] [yfinance] auto-reject write failed for %s: %s",
                                    ticker,
                                    rej_err,
                                )
                            return  # Skip all other collection for this ticker

                        record_collection(
                            "fundamentals", ticker, rows=prices + fundies + fins + bs
                        )
                        ms = elapsed_ms(t0)
                        local[f"{ticker}_yfinance"] = {
                            "prices": prices,
                            "fundamentals": fundies,
                            "financials": fins,
                            "balance_sheet": bs,
                            "ms": ms,
                        }
                        detail = (
                            f"{ticker}: {prices} prices, "
                            f"{fundies} fundamentals, {fins} financials, "
                            f"{bs} balance sheet rows"
                        )
                        emit(
                            "collecting",
                            f"yfinance_{ticker}",
                            detail,
                            status="ok",
                            data={
                                "prices": prices,
                                "fundamentals": fundies,
                                "financials": fins,
                                "balance_sheet": bs,
                            },
                            elapsed_ms=ms,
                        )
                        logger.info(
                            f"[PIPELINE]   [yfinance] {ticker}: {ms}ms -- prices={prices}, fins={fins}"
                        )
                    else:
                        ms = elapsed_ms(t0)
                        emit(
                            "collecting",
                            f"yfinance_{ticker}",
                            f"{ticker}: fresh, skipping",
                            status="skipped",
                            elapsed_ms=ms,
                        )
                        logger.info(f"[PIPELINE]   [yfinance] {ticker} fresh, skipping")
                except asyncio.TimeoutError:
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"yfinance_{ticker}",
                        f"{ticker}: yfinance TIMEOUT ({SOURCE_TIMEOUT}s)",
                        status="timeout",
                        elapsed_ms=ms,
                    )
                    logger.error(
                        f"[PIPELINE]   [yfinance] {ticker} TIMEOUT after {SOURCE_TIMEOUT}s — removing from cycle"
                    )
                    _ticker_rejected.set()
                except Exception as e:
                    _log_err("yfinance", e, ticker)
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"yfinance_{ticker}",
                        f"{ticker}: Failed -- {e}",
                        status="error",
                        elapsed_ms=ms,
                    )
                    logger.info(
                        f"[PIPELINE]   [yfinance] {ticker} FAILED: {e} — removing from cycle"
                    )
                    _ticker_rejected.set()

            # ── Source 2: Finnhub news ──
            async def _src_finnhub():
                if _ticker_rejected.is_set():
                    return  # Ticker already rejected by yfinance
                t0 = time.monotonic()
                try:
                    if not _is_sufficient and should_collect("news_finnhub", ticker):
                        from app.collectors.finnhub_collector import collect_news

                        async with rate_limiter.acquire("finnhub"):
                            news = await asyncio.wait_for(
                                collect_news(ticker), timeout=SOURCE_TIMEOUT
                            )
                        record_collection("news_finnhub", ticker, rows=news)
                        ms = elapsed_ms(t0)
                        local[f"{ticker}_finnhub"] = {"news": news}
                        emit(
                            "collecting",
                            f"finnhub_{ticker}",
                            f"{ticker}: {news} articles from Finnhub",
                            status="ok",
                            data={"articles": news},
                            elapsed_ms=ms,
                        )
                        logger.info(
                            f"[PIPELINE]   [finnhub] {ticker}: {news} articles ({ms}ms)"
                        )
                    else:
                        ms = elapsed_ms(t0)
                        emit(
                            "collecting",
                            f"finnhub_{ticker}",
                            f"{ticker}: fresh, skipping",
                            status="skipped",
                            elapsed_ms=ms,
                        )
                        logger.info(f"[PIPELINE]   [finnhub] {ticker} fresh, skipping")
                except asyncio.TimeoutError:
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"finnhub_{ticker}",
                        f"{ticker}: Finnhub TIMEOUT ({SOURCE_TIMEOUT}s)",
                        status="timeout",
                        elapsed_ms=ms,
                    )
                    logger.error(f"[PIPELINE]   [finnhub] {ticker} TIMEOUT")
                except Exception as e:
                    _log_err("finnhub", e, ticker)
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"finnhub_{ticker}",
                        f"{ticker}: Finnhub skipped -- {e}",
                        status="skipped",
                        elapsed_ms=ms,
                    )
                    logger.info(f"[PIPELINE]   [finnhub] {ticker} skipped: {e}")

            # ── Source 3: Reddit search ──
            async def _src_reddit():
                if _ticker_rejected.is_set():
                    return  # Ticker already rejected by yfinance
                t0 = time.monotonic()
                try:
                    if not _is_sufficient and should_collect("reddit", ticker):
                        from app.collectors.reddit_collector import (
                            collect_for_ticker as reddit_snipe,
                        )

                        # Fix 3: Apply timeout INSIDE the rate limiter to prevent
                        # the semaphore acquire from inflating the timeout duration.
                        # Previously, wait_for wrapped both acquire+fetch, so if
                        # acquire waited 120s the actual fetch got 0s budget.
                        async with rate_limiter.acquire("reddit"):
                            reddit_t = await asyncio.wait_for(
                                reddit_snipe(ticker), timeout=SOURCE_TIMEOUT
                            )
                        record_collection("reddit", ticker, rows=reddit_t)
                        ms = elapsed_ms(t0)
                        local[f"{ticker}_reddit_search"] = {"posts": reddit_t}
                        emit(
                            "collecting",
                            f"reddit_{ticker}",
                            f"{ticker}: {reddit_t} Reddit posts via search",
                            status="ok",
                            data={"posts": reddit_t},
                            elapsed_ms=ms,
                        )
                        logger.info(
                            f"[PIPELINE]   [Reddit] {ticker}: {reddit_t} posts ({ms}ms)"
                        )
                    else:
                        ms = elapsed_ms(t0)
                        emit(
                            "collecting",
                            f"reddit_{ticker}",
                            f"{ticker}: fresh, skipping",
                            status="skipped",
                            elapsed_ms=ms,
                        )
                        logger.info(f"[PIPELINE]   [Reddit] {ticker} fresh, skipping")
                except asyncio.TimeoutError:
                    ms = elapsed_ms(t0)
                    actual_s = (time.monotonic() - t0)
                    emit(
                        "collecting",
                        f"reddit_{ticker}",
                        f"{ticker}: Reddit TIMEOUT ({SOURCE_TIMEOUT}s configured, {actual_s:.0f}s actual)",
                        status="timeout",
                        elapsed_ms=ms,
                    )
                    logger.error(
                        "[PIPELINE]   [Reddit] %s TIMEOUT (configured=%ss, actual=%.0fs)",
                        ticker, SOURCE_TIMEOUT, actual_s,
                    )
                except Exception as e:
                    _log_err("reddit", e, ticker)
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"reddit_{ticker}",
                        f"{ticker}: Reddit search skipped -- {e}",
                        status="skipped",
                        elapsed_ms=ms,
                    )
                    logger.info(f"[PIPELINE]   [Reddit] {ticker} search skipped: {e}")

            # ── Source 4: YouTube search + transcript ──
            async def _src_youtube():
                if _ticker_rejected.is_set():
                    return  # Ticker already rejected by yfinance
                t0 = time.monotonic()
                try:
                    if not _is_sufficient and should_collect("youtube", ticker):
                        from app.collectors.youtube_collector import (
                            collect_for_ticker as youtube_snipe,
                        )

                        @retry(
                            stop=stop_after_attempt(3),
                            wait=wait_exponential(multiplier=2, min=2, max=30),
                            retry=retry_if_exception_type(
                                (ConnectionError, OSError, TimeoutError, asyncio.TimeoutError)
                            ),
                            reraise=True,
                        )
                        async def _fetch():
                            async with rate_limiter.acquire("youtube"):
                                seven_days_ago = datetime.datetime.now(
                                    datetime.UTC
                                ) - datetime.timedelta(days=7)
                                return await youtube_snipe(
                                    ticker, max_results=5, since=seven_days_ago
                                )

                        yt_stats = await asyncio.wait_for(
                            _fetch(), timeout=180.0
                        )  # Extended 180s hard timeout to prevent yt-dlp stalls

                        yt_t = yt_stats.get("stored", 0)
                        record_collection("youtube", ticker, rows=yt_t)
                        ms = elapsed_ms(t0)
                        local[f"{ticker}_youtube_search"] = {"transcripts": yt_t}
                        emit(
                            "collecting",
                            f"youtube_{ticker}",
                            f"{ticker}: {yt_t} YouTube transcripts via search",
                            status="ok",
                            data={"transcripts": yt_t},
                            elapsed_ms=ms,
                        )
                        logger.info(
                            f"[PIPELINE]   [YouTube] {ticker}: {yt_t} transcripts ({ms}ms)"
                        )
                    else:
                        ms = elapsed_ms(t0)
                        emit(
                            "collecting",
                            f"youtube_{ticker}",
                            f"{ticker}: fresh, skipping",
                            status="skipped",
                            elapsed_ms=ms,
                        )
                        logger.info(f"[PIPELINE]   [YouTube] {ticker} fresh, skipping")
                except asyncio.TimeoutError:
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"youtube_{ticker}",
                        f"{ticker}: YouTube TIMEOUT ({SOURCE_TIMEOUT}s)",
                        status="timeout",
                        elapsed_ms=ms,
                    )
                    logger.error(f"[PIPELINE]   [YouTube] {ticker} TIMEOUT")
                except Exception as e:
                    _log_err("youtube", e, ticker)
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"youtube_{ticker}",
                        f"{ticker}: YouTube search skipped -- {e}",
                        status="skipped",
                        elapsed_ms=ms,
                    )
                    logger.info(f"[PIPELINE]   [YouTube] {ticker} search skipped: {e}")

            # ── Source 5: yfinance curated news ──
            async def _src_yf_news():
                if _ticker_rejected.is_set():
                    return  # Ticker already rejected by yfinance
                t0 = time.monotonic()
                try:
                    if not _is_sufficient and should_collect("news_yfinance", ticker):
                        from app.collectors.yfinance_collector import (
                            collect_news as yf_news_collector,
                        )

                        async with rate_limiter.acquire("yf_news"):
                            yf_n = await asyncio.wait_for(
                                yf_news_collector(ticker), timeout=SOURCE_TIMEOUT
                            )
                        record_collection("news_yfinance", ticker, rows=yf_n)
                        ms = elapsed_ms(t0)
                        local[f"{ticker}_yfinance_news"] = {"articles": yf_n}
                        emit(
                            "collecting",
                            f"yfnews_{ticker}",
                            f"{ticker}: {yf_n} curated Yahoo Finance articles",
                            status="ok",
                            data={"articles": yf_n},
                            elapsed_ms=ms,
                        )
                        logger.info(
                            f"[PIPELINE]   [yfinance] {ticker}: {yf_n} curated news ({ms}ms)"
                        )
                    else:
                        ms = elapsed_ms(t0)
                        emit(
                            "collecting",
                            f"yfnews_{ticker}",
                            f"{ticker}: fresh, skipping",
                            status="skipped",
                            elapsed_ms=ms,
                        )
                        logger.info(
                            f"[PIPELINE]   [yfinance] {ticker} news fresh, skipping"
                        )
                except asyncio.TimeoutError:
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"yfnews_{ticker}",
                        f"{ticker}: yfinance news TIMEOUT ({SOURCE_TIMEOUT}s)",
                        status="timeout",
                        elapsed_ms=ms,
                    )
                    logger.error(f"[PIPELINE]   [yfinance] {ticker} news TIMEOUT")
                except Exception as e:
                    _log_err("yf_news", e, ticker)
                    ms = elapsed_ms(t0)
                    emit(
                        "collecting",
                        f"yfnews_{ticker}",
                        f"{ticker}: yfinance news skipped -- {e}",
                        status="skipped",
                        elapsed_ms=ms,
                    )
                    logger.info(f"[PIPELINE]   [yfinance] {ticker} news skipped: {e}")

            # ── Fire all 5 sources in parallel (rate limiters prevent overloading) ──
            # NOTE: CancelledError propagates from gather() automatically.
            # yfinance runs first and can signal rejection via _ticker_rejected.
            # Due to the gather, sibling sources check the event at their start.
            # For already-running siblings, they complete but the results are
            # discarded below when _ticker_rejected is checked.
            await asyncio.gather(
                _src_yfinance(),
                _src_finnhub(),
                _src_reddit(),
                _src_youtube(),
                _src_yf_news(),
            )

            # ── FELL gap fix: if ticker was auto-rejected, skip everything ──
            if _ticker_rejected.is_set():
                ticker_ms = elapsed_ms(ticker_start)
                logger.info(
                    "[PIPELINE]   --- %s REJECTED (delisted/untradeable): %dms ---",
                    ticker,
                    ticker_ms,
                )
                return None  # Excluded from analysis

            ticker_ms = elapsed_ms(ticker_start)
            logger.info(
                f"[PIPELINE]   --- {ticker} complete: {ticker_ms}ms ({ticker_ms / 1000:.1f}s) ---"
            )

            # Merge local results into shared dict under lock
            async with results_lock:
                results["collectors"].update(local)

            # ── Update watchlist health signals for this ticker ──
            try:
                from app.pipeline.watchlist_health import update_signals_from_collection

                finnhub_news = local.get(f"{ticker}_finnhub", {}).get("news", 0)
                yf_news = local.get(f"{ticker}_yfinance_news", {}).get("articles", 0)
                reddit_posts = local.get(f"{ticker}_reddit_search", {}).get("posts", 0)
                yt_transcripts = local.get(f"{ticker}_youtube_search", {}).get(
                    "transcripts", 0
                )
                yf_ok = f"{ticker}_yfinance" in local
                update_signals_from_collection(
                    ticker,
                    {
                        "news": (finnhub_news or 0) + (yf_news or 0),
                        "reddit": reddit_posts or 0,
                        "youtube": yt_transcripts or 0,
                        "yfinance_ok": yf_ok,
                    },
                )
            except Exception as e:
                logger.info(
                    f"[PIPELINE]   [health] {ticker} signal update skipped: {e}"
                )

            # ── Compute Technicals Immediately (Required for pipelining) ──
            try:
                from app.processors.technical_processor import compute_technicals

                tech_t0 = time.monotonic()
                rows = compute_technicals(ticker)
                tech_ms = elapsed_ms(tech_t0)
                async with results_lock:
                    results.setdefault("processors", {})[f"{ticker}_technicals"] = rows
                emit(
                    "collecting",
                    f"technicals_{ticker}",
                    f"{ticker}: {rows} technical indicator rows computed",
                    status="ok",
                    data={"rows": rows},
                    elapsed_ms=tech_ms,
                )
                logger.info(f"[PIPELINE]   [tech] {ticker}: {rows} indicator rows")
            except Exception as e:
                logger.info(f"[PIPELINE]   [tech] {ticker} FAILED: {e}")

            # ── Push to analysis queue FIRST (eliminates queue starvation) ──
            # Analysis workers were starving for 3-5 minutes waiting for
            # smart janitor + summarizer + consensus + narrative to complete.
            # The V2 pipeline's data_completeness check handles any gaps.
            if analysis_queue is not None:
                await analysis_queue.put(ticker)
                logger.info(
                    "[PIPELINE] %s collection + technicals done → queued for analysis (processors running in background)",
                    ticker,
                )

            # ── Run per-ticker processors as background task ──
            # These are LLM-heavy (dedup, summarize, consensus, narrative)
            # and should NOT block the analysis queue push.
            async def _bg_processors(t: str):
                try:
                    await run_ticker_processors(t, emit)
                except Exception as proc_err:
                    logger.warning("[PIPELINE] Background processors failed for %s: %s", t, proc_err)

            asyncio.create_task(_bg_processors(ticker))

            return ticker

    # ── Tool Calling Bypass ──
    if getattr(settings, "USE_TOOL_CALLING", False):
        logger.info("[PIPELINE] \n--- Pass 4: SKIPPED (Tool-Calling enabled) ---")
        emit(
            "collecting",
            "pass4_skip",
            "Skipping scraping per-ticker data (delegate to LLM tools)",
            status="ok",
        )

        # We must push tickers to the analysis queue since we bypassed it
        if analysis_queue is not None:
            for t in tickers:
                await analysis_queue.put(t)

        ticker_res = [t for t in tickers]
    else:
        # Launch all tickers concurrently (semaphore enforces the cap)
        # NOTE: CancelledError propagates from gather() automatically — no post-gather check needed.
        async with pipeline_profiler.phase("pass4_per_ticker_collection"):
            ticker_res = await asyncio.gather(
                *[_collect_single_ticker(t) for t in tickers],
            )

    # Filter out tickers that were rejected (None) or banned
    valid_tickers = [t for t in ticker_res if t is not None]
    if len(valid_tickers) < len(tickers):
        logger.info(
            f"[PIPELINE]   [pass4] Dropped {len(tickers) - len(valid_tickers)} rejected/toxic tickers. Remaining: {len(valid_tickers)}"
        )
    tickers = valid_tickers

    # ── Alpha Decay Pruning (Fix #9: batch post-gather, not per-ticker) ──
    if tickers:
        try:
            from app.pipeline.alpha_decay_purge import run_alpha_decay_purge

            purged = run_alpha_decay_purge(tickers)
            if purged:
                tickers = [t for t in tickers if t not in purged]
                logger.info(
                    f"[PIPELINE]   [alpha_decay] Banned {len(purged)} toxic tickers: {', '.join(purged)}"
                )
                # Remove banned tickers from analysis queue if pipelining
                # (they were already pushed but will be ignored downstream)
        except Exception as e:
            logger.error(
                f"[PIPELINE]   [alpha_decay] Batch check failed (non-fatal): {e}"
            )

    # Update results with final filtered list
    results["tickers"] = tickers

    logger.info(
        "[PIPELINE] ═══ All %d per-ticker collections complete ═══", len(tickers)
    )

    # ═══════════════════════════════════════════════════════════
    # PASS 4.5: FALLBACK COLLECTION (agentic gap-filling via Hermes)
    # Detects tickers with critical data gaps after standard collection
    # and uses Hermes web research to fill them. Non-fatal.
    # ═══════════════════════════════════════════════════════════
    if tickers:
        try:
            from app.pipeline.data.fallback_collector import (
                detect_data_gaps,
                fill_gaps_via_hermes,
            )

            t0 = time.monotonic()
            gaps = detect_data_gaps(tickers)
            if gaps:
                logger.info(
                    "[PIPELINE] Pass 4.5: %d tickers have data gaps — launching fallback",
                    len(gaps),
                )
                emit(
                    "collecting",
                    "fallback_start",
                    f"{len(gaps)} tickers have data gaps — launching agentic fallback...",
                    status="running",
                )
                fallback_results = await fill_gaps_via_hermes(gaps, emit)
                ms = elapsed_ms(t0)
                filled = fallback_results.get("filled", 0)
                emit(
                    "collecting",
                    "fallback_done",
                    f"Fallback: {filled}/{len(gaps)} tickers had gaps filled via Hermes",
                    status="ok",
                    data={"filled": filled, "gaps": len(gaps)},
                    elapsed_ms=ms,
                )
                results["collectors"]["fallback"] = fallback_results
            else:
                logger.info(
                    "[PIPELINE] Pass 4.5: No data gaps detected, skipping fallback"
                )
        except Exception as e:
            logger.warning(
                "[PIPELINE] Pass 4.5: Fallback collector failed (non-fatal): %s", e
            )
            emit(
                "collecting",
                "fallback_error",
                f"Fallback collector failed (non-fatal): {e}",
                status="error",
            )

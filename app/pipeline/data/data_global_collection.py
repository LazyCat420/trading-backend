import time
import asyncio
import logging
from typing import Callable
from app.monitoring.pipeline_profiler import profiler as pipeline_profiler
from app.pipeline.data.collection_scheduler import should_collect, record_collection
from app.utils.pipeline_utils import elapsed_ms

logger = logging.getLogger(__name__)


async def run_global_collection(
    tickers: list[str],
    force_global: bool,
    intensity: str,
    emit: Callable,
    results: dict,
    _summary: dict,
):
    # ═══════════════════════════════════════════════════════════
    # PASS 1: GLOBAL COLLECTORS (all independent — run in parallel)
    # ═══════════════════════════════════════════════════════════
    from app.pipeline.orchestration.cycle_control import cycle_control

    await cycle_control.wait_if_paused()

    logger.info("[PIPELINE] \n--- Pass 1: Global Data Collection (parallel) ---")
    emit(
        "collecting",
        "pass1_global",
        "Gathering global market data in parallel (news, reddit, youtube, macro)..."
        + (" [FORCED — discovery mode]" if force_global else ""),
        status="running",
    )

    # ── Helper wrappers for each global collector ──

    async def _global_fred():
        await cycle_control.wait_if_paused()
        if not force_global and not should_collect("fred"):
            emit("collecting", "fred", "FRED macro: fresh, skipped", status="skipped")
            logger.debug("[PIPELINE]   [FRED] fresh, skipping")
            _summary["collector_skipped"] += 1
            return
        t0 = time.monotonic()
        try:
            emit(
                "collecting",
                "fred",
                f"Fetching macro indicators (GDP, CPI, rates) [{intensity}]...",
                status="running",
            )
            from app.collectors.fred_collector import collect_macro_indicator, SERIES

            # Throttle macro indicators in lower intensity modes
            if intensity == "micro":
                series_to_check = dict(list(SERIES.items())[:3])  # top 3 indicators
            elif intensity == "light":
                series_to_check = dict(list(SERIES.items())[:6])  # top 6 indicators
            else:
                series_to_check = SERIES

            macro_total = 0
            for name, series_id in series_to_check.items():
                count = await collect_macro_indicator(name, series_id, lookback_years=1)
                macro_total += count
            ms = elapsed_ms(t0)
            results["collectors"]["fred"] = {"indicators": macro_total}
            record_collection("fred", rows=macro_total)
            _summary["collector_ok"] += 1
            emit(
                "collecting",
                "fred",
                f"{len(series_to_check)} FRED series updated ({macro_total} data points)",
                status="ok",
                data={
                    "indicators": macro_total,
                    "series": list(series_to_check.keys()),
                },
                elapsed_ms=ms,
            )
            logger.debug(
                f"[PIPELINE]   [FRED] {len(series_to_check)} series, {macro_total} data points ({ms}ms)"
            )
        except Exception as e:
            ms = elapsed_ms(t0)
            _summary["collector_error"] += 1
            _summary["failed_collectors"].append("fred")
            emit(
                "collecting",
                "fred",
                f"FRED failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.error(f"[PIPELINE]   [FRED] FAILED: {e}")

    async def _global_coingecko():
        await cycle_control.wait_if_paused()
        if not force_global and not should_collect("coingecko"):
            emit(
                "collecting", "coingecko", "CoinGecko: fresh, skipped", status="skipped"
            )
            logger.debug("[PIPELINE]   [CoinGecko] fresh, skipping")
            _summary["collector_skipped"] += 1
            return
        t0 = time.monotonic()
        try:
            from app.collectors.coingecko_collector import (
                collect_crypto_prices,
                TRACKED_COINS,
            )

            days = 1 if intensity == "micro" else 3 if intensity == "light" else 7
            crypto = await collect_crypto_prices(days=days)
            coins = [c[1] for c in TRACKED_COINS]
            ms = elapsed_ms(t0)
            results["collectors"]["coingecko"] = {"rows": crypto}
            record_collection("coingecko", rows=crypto)
            _summary["collector_ok"] += 1
            emit(
                "collecting",
                "coingecko",
                f"{len(coins)} coins × {days} days ({', '.join(coins)})",
                status="ok",
                data={"rows": crypto, "coins": coins},
                elapsed_ms=ms,
            )
            logger.debug(
                f"[PIPELINE]   [CoinGecko] {len(coins)} coins, {days} days: {', '.join(coins)} ({ms}ms)"
            )
        except Exception as e:
            ms = elapsed_ms(t0)
            _summary["collector_error"] += 1
            _summary["failed_collectors"].append("coingecko")
            emit(
                "collecting",
                "coingecko",
                f"CoinGecko failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.error(f"[PIPELINE]   [CoinGecko] FAILED: {e}")

    async def _global_congress():
        await cycle_control.wait_if_paused()
        if not force_global and not should_collect("congress"):
            emit(
                "collecting",
                "congress",
                "Congress trades: fresh, skipped",
                status="skipped",
            )
            logger.debug("[PIPELINE]   [Congress] fresh, skipping")
            _summary["collector_skipped"] += 1
            return
        t0 = time.monotonic()
        try:
            from app.collectors.congress_collector import collect_trades

            pages = 1 if intensity == "micro" else 2 if intensity == "light" else 3
            trades = await collect_trades(pages=pages)
            ms = elapsed_ms(t0)
            results["collectors"]["congress"] = {"trades": trades}
            record_collection("congress", rows=trades)
            _summary["collector_ok"] += 1
            emit(
                "collecting",
                "congress",
                f"{trades} congress trades ({pages} pages)",
                status="ok",
                data={"trades": trades},
                elapsed_ms=ms,
            )
            logger.debug(f"[PIPELINE]   [Congress] {trades} trades ({ms}ms)")
        except Exception as e:
            ms = elapsed_ms(t0)
            _summary["collector_error"] += 1
            _summary["failed_collectors"].append("congress")
            emit(
                "collecting",
                "congress",
                f"Congress failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.error(f"[PIPELINE]   [Congress] FAILED: {e}")

    async def _global_sec_13f():
        await cycle_control.wait_if_paused()
        if not force_global and not should_collect("sec_13f"):
            emit("collecting", "sec_13f", "SEC 13F: fresh, skipped", status="skipped")
            logger.debug("[PIPELINE]   [SEC 13F] fresh, skipping")
            _summary["collector_skipped"] += 1
            return
        t0 = time.monotonic()
        try:
            from app.services.sec_13f.sec_13f_service import (
                SEC13FCollector,
                DEFAULT_FILERS,
            )

            collector = SEC13FCollector()
            max_f = 2 if intensity == "micro" else 5 if intensity == "light" else None
            fund_results = await collector.collect_recent_holdings(max_filers=max_f)
            total = len(fund_results)

            ms = elapsed_ms(t0)
            results["collectors"]["sec_13f"] = {"holdings": total}
            record_collection("sec_13f", rows=total)
            _summary["collector_ok"] += 1
            checked_funds = max_f if max_f is not None else len(DEFAULT_FILERS)
            emit(
                "collecting",
                "sec_13f",
                f"Checked {checked_funds} funds, extracted {total} scored tickers",
                status="ok",
                data={"tickers": total},
                elapsed_ms=ms,
            )
            logger.debug(
                f"[PIPELINE]   [SEC 13F] Checked {checked_funds} funds, {total} tickers ({ms}ms)"
            )

        except Exception as e:
            ms = elapsed_ms(t0)
            _summary["collector_error"] += 1
            _summary["failed_collectors"].append("sec_13f")
            emit(
                "collecting",
                "sec_13f",
                f"SEC 13F failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.error(f"[PIPELINE]   [SEC 13F] FAILED: {e}")

    async def _global_news_rss():
        await cycle_control.wait_if_paused()
        if not force_global and not should_collect("news_rss"):
            emit("collecting", "news_rss", "News RSS: fresh, skipped", status="skipped")
            logger.debug("[PIPELINE]   [News RSS] fresh, skipping")
            _summary["collector_skipped"] += 1
            return
        t0 = time.monotonic()
        try:
            emit(
                "collecting",
                "news_rss",
                f"Fetching news from RSS feeds [{intensity}]...",
                status="running",
            )
            from app.collectors.news_collector import collect_all as collect_news_rss

            limit_f = 3 if intensity == "micro" else 6 if intensity == "light" else None
            # Hard timeout: RSS was the dominant bottleneck (10+ min in some cycles).
            # Proceed with whatever was collected before the timeout.
            RSS_TIMEOUT = 300  # 5 minutes max
            try:
                news_rss = await asyncio.wait_for(collect_news_rss(limit_feeds=limit_f), timeout=RSS_TIMEOUT)
            except asyncio.TimeoutError:
                ms = elapsed_ms(t0)
                logger.warning("[PIPELINE]   [News RSS] TIMEOUT after %ds — proceeding with partial results", RSS_TIMEOUT)
                _summary["collector_ok"] += 1  # partial success
                emit(
                    "collecting",
                    "news_rss",
                    f"News RSS TIMEOUT after {RSS_TIMEOUT}s — proceeding with partial data",
                    status="warning",
                    elapsed_ms=ms,
                )
                return
            ms = elapsed_ms(t0)
            results["collectors"]["news_rss"] = {"articles": news_rss}
            record_collection("news_rss", rows=news_rss)
            _summary["collector_ok"] += 1
            emit(
                "collecting",
                "news_rss",
                f"{news_rss} articles from RSS feeds ({limit_f or 'All'} feeds)",
                status="ok",
                data={"articles": news_rss},
                elapsed_ms=ms,
            )
            logger.debug(f"[PIPELINE]   [News RSS] {news_rss} articles ({ms}ms)")
        except Exception as e:
            ms = elapsed_ms(t0)
            _summary["collector_error"] += 1
            _summary["failed_collectors"].append("news_rss")
            emit(
                "collecting",
                "news_rss",
                f"News RSS failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.error(f"[PIPELINE]   [News RSS] FAILED: {e}")

    async def _global_news_api_rotator():
        await cycle_control.wait_if_paused()
        if not force_global and not should_collect("news_api_rotator"):
            emit(
                "collecting",
                "news_rotator",
                "News Rotator: fresh, skipped",
                status="skipped",
            )
            logger.debug("[PIPELINE]   [News Rotator] fresh, skipping")
            _summary["collector_skipped"] += 1
            return
        t0 = time.monotonic()
        try:
            emit(
                "collecting",
                "news_rotator",
                f"Fetching news from free API pool [{intensity}]...",
                status="running",
            )
            from app.collectors.news_api_rotator import collect_from_all_apis

            # Use top watchlist tickers for search query to bias the news
            wl = list(tickers)[:5]
            query = " ".join(wl) + " market earnings" if wl else "stock market earnings"

            count = await collect_from_all_apis(tickers=list(tickers), query=query)
            ms = elapsed_ms(t0)
            results["collectors"]["news_rotator"] = {"articles": count}
            record_collection("news_api_rotator", rows=count)
            _summary["collector_ok"] += 1
            emit(
                "collecting",
                "news_rotator",
                f"{count} articles from free API pool",
                status="ok",
                data={"articles": count},
                elapsed_ms=ms,
            )
            logger.debug(f"[PIPELINE]   [News Rotator] {count} articles ({ms}ms)")
        except Exception as e:
            ms = elapsed_ms(t0)
            _summary["collector_error"] += 1
            _summary["failed_collectors"].append("news_rotator")
            emit(
                "collecting",
                "news_rotator",
                f"News Rotator failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.error(f"[PIPELINE]   [News Rotator] FAILED: {e}")

    async def _global_reddit():
        await cycle_control.wait_if_paused()
        if not force_global and not should_collect("reddit"):
            emit("collecting", "reddit", "Reddit: fresh, skipped", status="skipped")
            logger.debug("[PIPELINE]   [Reddit] fresh, skipping")
            _summary["collector_skipped"] += 1
            return
        t0 = time.monotonic()
        try:
            emit(
                "collecting",
                "reddit",
                f"Scraping financial subreddits [{intensity}]...",
                status="running",
            )
            from app.collectors.reddit_collector import collect_all as collect_reddit

            lim = 2 if intensity == "micro" else 5 if intensity == "light" else 10
            reddit = await collect_reddit(time_filter="day", limit=lim)
            ms = elapsed_ms(t0)
            results["collectors"]["reddit"] = {"posts": reddit}
            record_collection("reddit", rows=reddit)
            _summary["collector_ok"] += 1
            emit(
                "collecting",
                "reddit",
                f"{reddit} posts across financial subreddits (limit {lim})",
                status="ok",
                data={"posts": reddit},
                elapsed_ms=ms,
            )
            logger.debug(f"[PIPELINE]   [Reddit] {reddit} posts ({ms}ms)")
        except Exception as e:
            ms = elapsed_ms(t0)
            _summary["collector_error"] += 1
            _summary["failed_collectors"].append("reddit")
            emit(
                "collecting",
                "reddit",
                f"Reddit failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.error(f"[PIPELINE]   [Reddit] FAILED: {e}")

    async def _global_youtube():
        await cycle_control.wait_if_paused()
        if not force_global and not should_collect("youtube"):
            emit("collecting", "youtube", "YouTube: fresh, skipped", status="skipped")
            logger.debug("[PIPELINE]   [YouTube] fresh, skipping")
            _summary["collector_skipped"] += 1
            return
        t0 = time.monotonic()
        try:
            emit(
                "collecting",
                "youtube",
                f"Downloading YouTube transcripts [{intensity}]...",
                status="running",
            )
            from app.collectors.youtube_collector import collect_all as collect_youtube

            max_v = 1 if intensity == "micro" else 2
            days_b = 3 if intensity == "micro" else 7
            max_q = 3 if intensity == "micro" else 8 if intensity == "light" else None
            yt = await collect_youtube(
                max_videos=max_v, days_back=days_b, max_queries=max_q
            )
            ms = elapsed_ms(t0)
            results["collectors"]["youtube"] = {"transcripts": yt}
            record_collection("youtube", rows=yt)
            _summary["collector_ok"] += 1
            emit(
                "collecting",
                "youtube",
                f"{yt} transcripts via dynamic search ({max_q or 'all'} queries)",
                status="ok",
                data={"transcripts": yt},
                elapsed_ms=ms,
            )
            logger.debug(f"[PIPELINE]   [YouTube] {yt} transcripts ({ms}ms)")
        except Exception as e:
            ms = elapsed_ms(t0)
            _summary["collector_error"] += 1
            _summary["failed_collectors"].append("youtube")
            emit(
                "collecting",
                "youtube",
                f"YouTube failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.error(f"[PIPELINE]   [YouTube] FAILED: {e}")

    # ── Fire all global collectors in parallel ──
    # NOTE: No return_exceptions=True — CancelledError propagates from gather()
    # automatically. No post-gather CancelledError check needed.
    pass1_start = time.monotonic()
    async with pipeline_profiler.phase("pass1_global_collectors"):
        await asyncio.gather(
            _global_fred(),
            _global_coingecko(),
            _global_congress(),
            _global_sec_13f(),
            _global_news_rss(),
            _global_news_api_rotator(),
            _global_reddit(),
            _global_youtube(),
        )
    pass1_ms = elapsed_ms(pass1_start)
    logger.info(
        f"[PIPELINE]   [Pass 1] All global collectors done in {pass1_ms}ms ({pass1_ms / 1000:.1f}s)"
    )

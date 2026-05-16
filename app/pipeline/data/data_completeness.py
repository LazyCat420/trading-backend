"""
Data Completeness Checker -- Ensures all data categories are populated
before agents run. If data is missing, triggers collectors on-demand.

Runs BEFORE the 5 agents in the decision engine.

Categories checked:
  1. Price history (yfinance) -- required
  2. Technicals (computed from price) -- required
  3. Fundamentals (yfinance) -- required
  4. News articles (news_collector) -- fills if missing/stale
  5. Reddit posts (reddit_collector) -- fills if missing/stale
  6. Congress trades (congress_collector) -- fills if missing
  7. YouTube transcripts -- fills if missing/stale
"""

import logging

logger = logging.getLogger(__name__)


from app.config import settings
from app.db.connection import get_db
from app.pipeline.data.scraper_queue import enqueue_request
import datetime


def _is_stale(db, table: str, ticker: str, date_col: str, max_age_hours: float) -> tuple[bool, datetime.datetime | None]:
    """Check if the most recent data for a ticker is older than max_age_hours."""
    try:
        row = db.execute(
            f"SELECT MAX({date_col}) FROM {table} WHERE ticker = %s", [ticker]
        ).fetchone()
        if not row or not row[0]:
            return True, None  # No data at all
        latest = row[0]
        if isinstance(latest, str):
            latest = datetime.datetime.fromisoformat(latest)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=datetime.UTC)
        age = (datetime.datetime.now(datetime.UTC) - latest).total_seconds() / 3600
        return age > max_age_hours, latest
    except Exception:
        return True, None  # If we can't check, assume stale


async def check_and_fill(ticker: str, emit=None, enqueue_only: bool = False) -> dict:
    """Check data completeness for a ticker. Fill gaps where possible.

    Returns dict with data counts per category and what was collected.
    """
    ticker = ticker.upper()
    report = {"ticker": ticker, "filled": [], "available": {}, "missing": []}

    def _emit(detail, status="info", data=None):
        if emit:
            emit("collecting", f"data_{ticker}", detail, status=status, data=data or {})
        logger.info(f"[PIPELINE]   {detail}")

    _emit(f"📋 {ticker}: Checking data completeness...", status="running")

    # Pre-fetch all data counts within a single context manager so we don't leak
    # or hold connections open across async await boundaries.
    with get_db() as db:
        price_count = db.execute(
            "SELECT COUNT(*) FROM price_history WHERE ticker = %s", [ticker]
        ).fetchone()[0]

        tech_count = db.execute(
            "SELECT COUNT(*) FROM technicals WHERE ticker = %s", [ticker]
        ).fetchone()[0]

        fund_count = db.execute(
            "SELECT COUNT(*) FROM fundamentals WHERE ticker = %s", [ticker]
        ).fetchone()[0]

        news_count = db.execute(
            "SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [ticker]
        ).fetchone()[0]
        news_stale, news_latest = _is_stale(db, "news_articles", ticker, "published_at", 1.0)

        reddit_count = db.execute(
            "SELECT COUNT(*) FROM reddit_posts WHERE ticker = %s", [ticker]
        ).fetchone()[0]
        reddit_stale, reddit_latest = _is_stale(db, "reddit_posts", ticker, "created_utc", 6.0)

        congress_count = db.execute(
            "SELECT COUNT(*) FROM congress_trades WHERE ticker = %s", [ticker]
        ).fetchone()[0]

        yt_count = db.execute(
            "SELECT COUNT(*) FROM youtube_transcripts WHERE ticker = %s", [ticker]
        ).fetchone()[0]
        yt_stale, yt_latest = _is_stale(db, "youtube_transcripts", ticker, "published_at", 6.0)

        crypto_count = 0
        crypto_syms = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX"}
        if ticker.upper() in crypto_syms:
            crypto_count = db.execute(
                "SELECT COUNT(*) FROM asset_prices "
                "WHERE symbol = %s AND asset_class = 'crypto'",
                [ticker.upper()],
            ).fetchone()[0]

    # ── 1. Price history ──
    report["available"]["price_history"] = price_count

    if price_count < 20:
        if enqueue_only:
            _emit(f"📈 {ticker}: Enqueueing price history request...", status="info")
            enqueue_request(
                ticker,
                "price_history",
                priority=settings.SCRAPER_JIT_PRIORITY,
                requested_by_lens="data_completeness",
            )
            report["filled"].append("price_history: enqueued")
            report["missing"].append("price_history")
        else:
            _emit(
                f"📈 {ticker}: Collecting price history ({price_count} rows)...",
                status="running",
            )
            try:
                from app.collectors.yfinance_collector import collect_price_history

                rows = await collect_price_history(ticker, period="3mo")
                report["filled"].append(f"price_history: {rows} rows")
                report["available"]["price_history"] = rows
                _emit(f"📈 {ticker}: Collected {rows} price rows ✓")
                if emit: emit("collecting", f"yfinance_{ticker}", f"Collected {rows} prices", status="ok", data={"rows": rows})
            except Exception as e:
                _emit(f"📈 {ticker}: Price collection failed: {e}", status="error")
                report["missing"].append("price_history")
                if emit: emit("collecting", f"yfinance_{ticker}", f"Failed: {e}", status="error")

    # ── 2. Technicals ──
    report["available"]["technicals"] = tech_count

    if tech_count < 10 and report["available"]["price_history"] >= 20:
        _emit(
            f"📉 {ticker}: Computing technicals ({tech_count} rows)...",
            status="running",
        )
        try:
            from app.processors.technical_processor import compute_technicals
            import asyncio

            rows = await asyncio.to_thread(compute_technicals, ticker)
            report["filled"].append(f"technicals: {rows} rows")
            report["available"]["technicals"] = rows
            _emit(f"📉 {ticker}: Computed {rows} technical rows ✓")
        except Exception as e:
            _emit(f"📉 {ticker}: Technicals computation failed: {e}", status="error")

    # ── 3. Fundamentals ──
    report["available"]["fundamentals"] = fund_count

    if fund_count == 0:
        if enqueue_only:
            _emit(f"📊 {ticker}: Enqueueing fundamentals request...", status="info")
            enqueue_request(
                ticker,
                "fundamentals",
                priority=settings.SCRAPER_JIT_PRIORITY,
                requested_by_lens="data_completeness",
            )
            report["filled"].append("fundamentals: enqueued")
            report["missing"].append("fundamentals")
        else:
            _emit(f"📊 {ticker}: Collecting fundamentals...", status="running")
            try:
                from app.collectors.yfinance_collector import collect_fundamentals

                rows = await collect_fundamentals(ticker)
                report["filled"].append(f"fundamentals: {rows} rows")
                report["available"]["fundamentals"] = rows
                _emit(f"📊 {ticker}: Collected fundamentals ✓")
                if emit: emit("collecting", f"yfinance_{ticker}", f"Collected fundamentals", status="ok")
            except Exception as e:
                _emit(f"📊 {ticker}: Fundamentals failed: {e}", status="error")
                report["missing"].append("fundamentals")
                if emit: emit("collecting", f"yfinance_{ticker}", f"Failed: {e}", status="error")

    # ── 4. News articles (re-scrape only if stale > 1 hour) ──
    report["available"]["news"] = news_count

    from app.pipeline.data.collection_scheduler import should_collect, record_collection

    if news_count < 3 or news_stale:
        if not should_collect("news_finnhub", ticker):
            _emit(f"📰 {ticker}: News recently collected, skipping JIT fetch.")
        elif enqueue_only:
            _emit(f"📰 {ticker}: Enqueueing news request...", status="info")
            enqueue_request(
                ticker,
                "news_articles",
                priority=settings.SCRAPER_JIT_PRIORITY,
                requested_by_lens="data_completeness",
            )
            report["filled"].append("news: enqueued")
            report["missing"].append("news")
        else:
            _emit(
                f"📰 {ticker}: Collecting news ({news_count} articles, stale)...",
                status="running",
            )
            try:
                from app.collectors.news_collector import collect_for_ticker

                rows = await collect_for_ticker(ticker, since=news_latest)
                report["filled"].append(f"news: {rows} articles")
                report["available"]["news"] = news_count + (rows or 0)
                record_collection("news_finnhub", ticker, rows=rows or 0)
                record_collection("news_yfinance", ticker, rows=rows or 0)
                _emit(f"📰 {ticker}: Collected {rows} news articles ✓")
                if emit: emit("collecting", f"finnhub_{ticker}", f"Collected {rows} articles", status="ok", data={"rows": rows})
            except Exception as e:
                _emit(f"📰 {ticker}: News collection failed: {e}", status="error")
                if emit: emit("collecting", f"finnhub_{ticker}", f"Failed: {e}", status="error")
    elif news_count >= 3:
        _emit(f"📰 {ticker}: {news_count} articles (up to date, skipping)")
        if emit: emit("collecting", f"finnhub_{ticker}", "fresh, skipping", status="skipped")

    # ── 5. Reddit posts (re-scrape only if stale > 6 hours) ──
    report["available"]["reddit"] = reddit_count

    if reddit_count < 3 or reddit_stale:
        if not should_collect("reddit", ticker):
            _emit(f"🟠 {ticker}: Reddit recently collected, skipping JIT fetch.")
        elif enqueue_only:
            _emit(f"🟠 {ticker}: Enqueueing Reddit request...", status="info")
            enqueue_request(
                ticker,
                "reddit_posts",
                priority=settings.SCRAPER_JIT_PRIORITY,
                requested_by_lens="data_completeness",
            )
            report["filled"].append("reddit: enqueued")
            report["missing"].append("reddit")
        else:
            _emit(
                f"🟠 {ticker}: Collecting Reddit posts ({reddit_count} posts)...",
                status="running",
            )
            try:
                from app.collectors.reddit_collector import collect_for_ticker

                rows = await collect_for_ticker(ticker, since=reddit_latest)
                report["filled"].append(f"reddit: {rows} posts")
                report["available"]["reddit"] = (reddit_count or 0) + (rows or 0)
                record_collection("reddit", ticker, rows=rows or 0)
                _emit(f"🟠 {ticker}: Collected {rows} Reddit posts ✓")
                if emit: emit("collecting", f"reddit_{ticker}", f"Collected {rows} posts", status="ok", data={"rows": rows})
            except Exception as e:
                _emit(f"🟠 {ticker}: Reddit collection failed: {e}", status="error")
                if emit: emit("collecting", f"reddit_{ticker}", f"Failed: {e}", status="error")
    elif reddit_count >= 3:
        _emit(f"🟠 {ticker}: {reddit_count} posts (up to date, skipping)")
        if emit: emit("collecting", f"reddit_{ticker}", "fresh, skipping", status="skipped")

    # ── 6. Congress trades ──
    report["available"]["congress"] = congress_count
    # Congress trades are collected in bulk, not per-ticker, so no fill needed

    # ── 7. YouTube transcripts (re-scrape only if stale > 6 hours) ──
    report["available"]["youtube"] = yt_count

    if yt_count < 2 or yt_stale:
        if not should_collect("youtube", ticker):
            _emit(f"🔴 {ticker}: YouTube recently collected, skipping JIT fetch.")
        elif enqueue_only:
            _emit(f"🔴 {ticker}: Enqueueing YouTube request...", status="info")
            enqueue_request(
                ticker,
                "youtube_transcripts",
                priority=settings.SCRAPER_JIT_PRIORITY,
                requested_by_lens="data_completeness",
            )
            report["filled"].append("youtube: enqueued")
            report["missing"].append("youtube")
        else:
            _emit(
                f"🔴 {ticker}: Searching YouTube transcripts ({yt_count} found)...",
                status="running",
            )
            try:
                from app.collectors.youtube_collector import (
                    collect_for_ticker as yt_collect,
                )

                if yt_latest:
                    seven_days_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=7)
                    if yt_latest > seven_days_ago:
                        yt_latest = seven_days_ago

                stats = await yt_collect(ticker, max_results=3, since=yt_latest)
                rows = stats.get("stored", 0)
                report["filled"].append(f"youtube: {rows} transcripts")
                report["available"]["youtube"] = (yt_count or 0) + rows
                record_collection("youtube", ticker, rows=rows)
                _emit(f"🔴 {ticker}: Collected {rows} YouTube transcripts ✓")
                if emit: emit("collecting", f"youtube_{ticker}", f"Collected {rows} transcripts", status="ok", data={"rows": rows})
            except Exception as e:
                _emit(f"🔴 {ticker}: YouTube collection failed: {e}", status="error")
                if emit: emit("collecting", f"youtube_{ticker}", f"Failed: {e}", status="error")
    elif yt_count >= 2:
        _emit(f"🔴 {ticker}: {yt_count} transcripts (up to date, skipping)")
        if emit: emit("collecting", f"youtube_{ticker}", "fresh, skipping", status="skipped")

    # ── 9. Crypto prices (for crypto tickers only) ──
    if ticker.upper() in crypto_syms:
        report["available"]["crypto_prices"] = crypto_count

        if crypto_count < 5:
            _emit(f"₿ {ticker}: Collecting crypto prices...", status="running")
            try:
                from app.collectors.coingecko_collector import (
                    collect_crypto_prices,
                )

                rows = await collect_crypto_prices(days=30)
                report["filled"].append(f"crypto: {rows} price points")
                report["available"]["crypto_prices"] = rows
                _emit(f"₿ {ticker}: Collected {rows} crypto prices ✓")
                if emit: emit("collecting", f"coingecko_{ticker}", f"Collected {rows} prices", status="ok", data={"rows": rows})
            except Exception as e:
                _emit(f"₿ {ticker}: Crypto collection failed: {e}", status="error")
                if emit: emit("collecting", f"coingecko_{ticker}", f"Failed: {e}", status="error")
        else:
            if emit: emit("collecting", f"coingecko_{ticker}", "fresh, skipping", status="skipped")

    # ── Summary ──
    total = sum(report["available"].values())
    filled_count = len(report["filled"])
    cats = len(report["available"])
    _emit(
        f"✅ {ticker}: {total} data points across {cats} categories"
        f"{f' (filled {filled_count} gaps)' if filled_count else ' (all up to date)'}"
    )

    return report


# ── Critical-data gate: price + technicals + fundamentals + news are required ──
CRITICAL_CATEGORIES = {"price_history": 20, "technicals": 5, "fundamentals": 1, "news": 1}


def check_data_sufficiency(report: dict) -> dict:
    """Evaluate whether critical data categories meet minimum thresholds.

    Returns dict with 'sufficient' bool and 'gaps' list of unmet categories.
    """
    available = report.get("available", {})
    gaps = []
    for cat, min_rows in CRITICAL_CATEGORIES.items():
        count = available.get(cat, 0)
        if count < min_rows:
            gaps.append({"category": cat, "have": count, "need": min_rows})
    return {"sufficient": len(gaps) == 0, "gaps": gaps}


async def check_and_fill_all(tickers: list[str], enqueue_only: bool = False) -> dict:
    """Check data completeness for all tickers. Fill gaps sequentially."""
    logger.info(
        f"[PIPELINE] \n  [DATA CHECK] Checking {len(tickers)} tickers for data gaps..."
    )
    reports = {}
    for ticker in tickers:
        reports[ticker] = await check_and_fill(ticker, enqueue_only=enqueue_only)

    # Summary
    total_filled = sum(len(r["filled"]) for r in reports.values())
    total_missing = sum(len(r["missing"]) for r in reports.values())
    logger.info(
        f"  [DATA CHECK] Complete: filled {total_filled} gaps, "
        f"{total_missing} still missing\n"
    )

    return reports

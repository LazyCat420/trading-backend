import time
import asyncio
import logging
from typing import Callable
from app.db.connection import get_db
from app.pipeline.orchestration.cycle_control import cycle_control
from app.monitoring.pipeline_profiler import profiler as pipeline_profiler
from app.utils.pipeline_utils import elapsed_ms

logger = logging.getLogger(__name__)


async def run_ticker_discovery_and_gates(
    tickers: list[str],
    discovered_tickers: list[str],
    emit: Callable,
    results: dict,
    _summary: dict,
) -> list[str]:
    # ═══════════════════════════════════════════════════════════
    # PASS 2: TICKER DISCOVERY (scan scraped data BEFORE per-ticker)
    # ═══════════════════════════════════════════════════════════
    await cycle_control.wait_if_paused()

    logger.info("[PIPELINE] \n--- Pass 2: Ticker Discovery ---")
    discovered_tickers: list[str] = []
    t0 = time.monotonic()
    async with pipeline_profiler.phase("pass2_ticker_discovery"):
        try:
            emit(
                "collecting",
                "discovery",
                "Scanning scraped data for ticker mentions...",
                status="running",
            )
            discovered_count, discovered_tickers = await _run_ticker_discovery(emit)
            ms = elapsed_ms(t0)
            results["processors"]["ticker_discovery"] = discovered_count
            emit(
                "collecting",
                "discovery",
                f"Discovered {discovered_count} unique tickers: {', '.join(discovered_tickers[:10])}{'...' if len(discovered_tickers) > 10 else ''}",
                status="ok",
                data={
                    "tickers_found": discovered_count,
                    "tickers": discovered_tickers[:20],
                },
                elapsed_ms=ms,
            )
            logger.info(
                f"[PIPELINE]   [discovery] {discovered_count} unique tickers found: {', '.join(discovered_tickers[:10])}"
            )
        except Exception as e:
            ms = elapsed_ms(t0)
            emit(
                "collecting",
                "discovery",
                f"Ticker discovery failed — {e}",
                status="error",
                elapsed_ms=ms,
            )
            logger.info(f"[PIPELINE]   [discovery] FAILED: {e}")

    # ═══════════════════════════════════════════════════════════
    # PASS 2.5: MARKET CAP GATE (filter junk before merge)
    # ═══════════════════════════════════════════════════════════
    if discovered_tickers:
        logger.info("[PIPELINE] \n--- Pass 2.5: Market Cap Validation ---")
        t0 = time.monotonic()
        async with pipeline_profiler.phase("pass2_5_market_cap_gate"):
            try:
                emit(
                    "collecting",
                    "market_cap_gate",
                    f"Validating {len(discovered_tickers)} discovered tickers via yfinance...",
                    status="running",
                )
                from app.processors.ticker_extractor import (
                    validate_unknown_tickers,
                    get_registry,
                )
                from app.config import settings

                # Only validate tickers not already in registry
                registry = get_registry()
                unknowns = [
                    t
                    for t in discovered_tickers
                    if not registry.is_known(t) and not registry.is_rejected(t)
                ]

                if unknowns:
                    validated = await validate_unknown_tickers(unknowns)
                    rejected = [s for s, ok in validated.items() if not ok]
                    verified = [s for s, ok in validated.items() if ok]
                    logger.info(
                        f"[PIPELINE]   [gate] yfinance: {len(verified)} verified, {len(rejected)} rejected"
                    )
                else:
                    rejected = []
                    verified = []

                # Apply market cap floor
                min_cap = settings.MIN_MARKET_CAP
                below_cap = []
                for t in list(discovered_tickers):
                    company = registry.lookup_symbol(t)
                    if company and 0 < company.market_cap < min_cap:
                        below_cap.append(t)
                        discovered_tickers.remove(t)

                # Remove rejected tickers
                discovered_tickers = [
                    t for t in discovered_tickers if t not in rejected
                ]

                ms = elapsed_ms(t0)
                emit(
                    "collecting",
                    "market_cap_gate",
                    f"Gate: {len(rejected)} rejected (yfinance), "
                    f"{len(below_cap)} below ${min_cap / 1e6:.0f}M cap, "
                    f"{len(discovered_tickers)} passed",
                    status="ok",
                    data={
                        "rejected": rejected,
                        "below_cap": below_cap,
                        "passed": len(discovered_tickers),
                    },
                    elapsed_ms=ms,
                )
                if rejected:
                    logger.info(
                        f"[PIPELINE]   [gate] Rejected: {', '.join(rejected[:15])}"
                    )
                if below_cap:
                    logger.info(
                        f"[PIPELINE]   [gate] Below ${min_cap / 1e6:.0f}M cap: {', '.join(below_cap[:15])}"
                    )
                logger.info(
                    f"[PIPELINE]   [gate] {len(discovered_tickers)} tickers passed gate"
                )
            except Exception as e:
                ms = elapsed_ms(t0)
                emit(
                    "collecting",
                    "market_cap_gate",
                    f"Market cap gate error — {e}",
                    status="error",
                    elapsed_ms=ms,
                )
                logger.info(f"[PIPELINE]   [gate] ERROR: {e}")

    # ═══════════════════════════════════════════════════════════
    # PASS 2.6: BAN GATE — filter banned tickers + pattern matches
    # ═══════════════════════════════════════════════════════════
    if discovered_tickers:
        logger.info("[PIPELINE] \n--- Pass 2.6: Ban Gate ---")
        t0 = time.monotonic()
        async with pipeline_profiler.phase("pass2_6_ban_gate"):
            try:
                from app.trading.watchlist import is_banned, check_ban_patterns

                banned_skip = []
                pattern_skip = []
                clean = []
                for t in discovered_tickers:
                    if is_banned(t):
                        banned_skip.append(t)
                    else:
                        pattern = check_ban_patterns(t)
                        if pattern:
                            pattern_skip.append((t, pattern))
                        else:
                            clean.append(t)
                discovered_tickers = clean
                ms = elapsed_ms(t0)
                emit(
                    "collecting",
                    "ban_gate",
                    f"Ban gate: {len(banned_skip)} banned, "
                    f"{len(pattern_skip)} pattern-matched, "
                    f"{len(discovered_tickers)} passed",
                    status="ok",
                    data={
                        "banned": banned_skip,
                        "pattern_matched": [t for t, _ in pattern_skip],
                        "passed": len(discovered_tickers),
                    },
                    elapsed_ms=ms,
                )
                if banned_skip:
                    logger.info(
                        f"[PIPELINE]   [ban] Skipped banned: {', '.join(banned_skip[:15])}"
                    )
                if pattern_skip:
                    logger.info(
                        f"[PIPELINE]   [ban] Pattern filtered: {', '.join(f'{t}({p})' for t, p in pattern_skip[:10])}"
                    )
                logger.info(
                    f"[PIPELINE]   [ban] {len(discovered_tickers)} tickers passed ban gate"
                )
            except Exception as e:
                ms = elapsed_ms(t0)
                emit(
                    "collecting",
                    "ban_gate",
                    f"Ban gate error — {e}",
                    status="error",
                    elapsed_ms=ms,
                )
                logger.info(f"[PIPELINE]   [ban] ERROR: {e}")

    # ═══════════════════════════════════════════════════════════
    # PASS 2.7: LLM CURATION — smart promotion to watchlist
    # ═══════════════════════════════════════════════════════════
    if discovered_tickers:
        logger.info("[PIPELINE] \n--- Pass 2.7: LLM Curation ---")
        t0 = time.monotonic()
        async with pipeline_profiler.phase("pass2_7_llm_curation"):
            try:
                from app.pipeline.analysis.curation_pass import curate_discoveries

                pre_count = len(discovered_tickers)
                promoted = await curate_discoveries(
                    discovered_tickers=discovered_tickers,
                    current_watchlist=list(tickers),
                    emit=emit,
                    cycle_id=getattr(emit, "_cycle_id", ""),
                )
                skipped = [t for t in discovered_tickers if t not in promoted]
                discovered_tickers = promoted  # Only LLM-approved go to merge
                ms = elapsed_ms(t0)
                emit(
                    "collecting",
                    "llm_curation",
                    f"LLM promoted {len(promoted)}/{pre_count}: "
                    f"{', '.join(promoted) if promoted else 'none'}",
                    status="ok",
                    data={
                        "promoted": promoted,
                        "skipped": skipped,
                        "pre_count": pre_count,
                    },
                    elapsed_ms=ms,
                )
                logger.info(
                    f"[PIPELINE]   [curation] LLM promoted: {', '.join(promoted) if promoted else 'none'}"
                )
                if skipped:
                    logger.info(
                        f"[PIPELINE]   [curation] LLM skipped: {', '.join(skipped)}"
                    )
            except Exception as e:
                ms = elapsed_ms(t0)
                emit(
                    "collecting",
                    "llm_curation",
                    f"LLM curation failed — falling back to all: {e}",
                    status="error",
                    elapsed_ms=ms,
                )
                logger.info(f"[PIPELINE]   [curation] FALLBACK (error): {e}")

    return discovered_tickers


async def _run_ticker_discovery(emit: Callable) -> tuple[int, list[str]]:
    """
    Scan recently scraped data for ticker mentions.

    Reads from news_articles, reddit_posts, youtube_transcripts.
    Runs the shared ticker_extractor (Layers 1-3) on each item.
    Inserts high-confidence discoveries into discovered_tickers.
    Returns (count, list_of_tickers) sorted by score descending.

    Fix #4: All three scan blocks run in parallel via asyncio.gather().
    Fix #11: Each scan block has its own local t0 to avoid closure bugs.
    Fix #8: Removed duplicate collect_metadata call (handled in Pass 3.5).
    """
    from app.processors.ticker_extractor import extract_and_validate

    seen: dict[str, float] = {}  # ticker → best score
    all_contexts: dict[str, str] = {}  # ticker → best context snippet
    all_sources: dict[str, str] = {}  # ticker → source type

    # Thread-safe merge helper (all scans write to shared dicts)
    _lock = asyncio.Lock()

    async def _merge_results(new_seen: dict, new_contexts: dict, source: str):
        """Merge scan results into shared dictionaries."""
        async with _lock:
            for sym, score in new_seen.items():
                if sym not in seen or score > seen[sym]:
                    seen[sym] = score
                    all_contexts[sym] = new_contexts.get(sym, "")
                    all_sources[sym] = source

    # ── Scan 1: news_articles (last 7 days) ──
    async def _scan_news():
        t0 = time.monotonic()  # Fix #11: local t0
        try:
            with get_db() as db:
                rows = db.execute("""
                    SELECT title, COALESCE(summary, '') AS summary
                    FROM news_articles
                    WHERE published_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
                    ORDER BY published_at DESC
                    LIMIT 500
                """).fetchall()
            local_seen = {}
            local_ctx = {}
            news_tickers = set()
            for title, summary in rows:
                text = f"{title} {summary}"[:2000]
                matches = extract_and_validate(
                    text, title=title, source="news"
                )
                for m in matches:
                    if m.confidence >= 0.60:
                        news_tickers.add(m.symbol)
                        if (
                            m.symbol not in local_seen
                            or m.confidence > local_seen[m.symbol]
                        ):
                            local_seen[m.symbol] = m.confidence
                            local_ctx[m.symbol] = m.context_snippet[:200]
            await _merge_results(local_seen, local_ctx, "news")
            ms = elapsed_ms(t0)
            emit(
                "collecting",
                "discovery_news",
                f"Scanned {len(rows)} news articles → {len(news_tickers)} tickers",
                status="ok",
                data={"articles_scanned": len(rows), "tickers": len(news_tickers)},
                elapsed_ms=ms,
            )
        except Exception as e:
            emit(
                "collecting",
                "discovery_news",
                f"News scan failed — {e}",
                status="error",
            )
            logger.info(f"[PIPELINE]   [discovery] news scan error: {e}")

    # ── Scan 2: reddit_posts (last 7 days) ──
    async def _scan_reddit():
        t0 = time.monotonic()  # Fix #11: local t0
        try:
            with get_db() as db:
                rows = db.execute("""
                    SELECT title, COALESCE(body, '') AS body
                    FROM reddit_posts
                    WHERE created_utc > CURRENT_TIMESTAMP - INTERVAL '7 days'
                    ORDER BY created_utc DESC
                    LIMIT 500
                """).fetchall()
            local_seen = {}
            local_ctx = {}
            reddit_tickers = set()
            for title, body in rows:
                text = f"{title} {body}"[:2000]
                matches = extract_and_validate(
                    text, title=title, source="reddit"
                )
                for m in matches:
                    if m.confidence >= 0.60:
                        reddit_tickers.add(m.symbol)
                        if (
                            m.symbol not in local_seen
                            or m.confidence > local_seen[m.symbol]
                        ):
                            local_seen[m.symbol] = m.confidence
                            local_ctx[m.symbol] = m.context_snippet[:200]
            await _merge_results(local_seen, local_ctx, "reddit")
            ms = elapsed_ms(t0)
            emit(
                "collecting",
                "discovery_reddit",
                f"Scanned {len(rows)} reddit posts → {len(reddit_tickers)} tickers",
                status="ok",
                data={"posts_scanned": len(rows), "tickers": len(reddit_tickers)},
                elapsed_ms=ms,
            )
        except Exception as e:
            emit(
                "collecting",
                "discovery_reddit",
                f"Reddit scan failed — {e}",
                status="error",
            )
            logger.info(f"[PIPELINE]   [discovery] reddit scan error: {e}")

    # ── Scan 3: youtube_transcripts (last 7 days) ──
    async def _scan_youtube():
        t0 = time.monotonic()  # Fix #11: local t0
        try:
            with get_db() as db:
                rows = db.execute("""
                    SELECT title, COALESCE(SUBSTRING(raw_transcript, 1, 3000), '') AS snippet
                    FROM youtube_transcripts
                    WHERE published_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
                    ORDER BY published_at DESC
                    LIMIT 100
                """).fetchall()
            local_seen = {}
            local_ctx = {}
            yt_tickers = set()
            for title, snippet in rows:
                text = f"{title} {snippet}"[:3000]
                matches = extract_and_validate(
                    text, title=title, source="youtube"
                )
                for m in matches:
                    if m.confidence >= 0.60:
                        yt_tickers.add(m.symbol)
                        if (
                            m.symbol not in local_seen
                            or m.confidence > local_seen[m.symbol]
                        ):
                            local_seen[m.symbol] = m.confidence
                            local_ctx[m.symbol] = m.context_snippet[:200]
            await _merge_results(local_seen, local_ctx, "youtube")
            ms = elapsed_ms(t0)
            emit(
                "collecting",
                "discovery_youtube",
                f"Scanned {len(rows)} transcripts → {len(yt_tickers)} tickers",
                status="ok",
                data={"transcripts_scanned": len(rows), "tickers": len(yt_tickers)},
                elapsed_ms=ms,
            )
        except Exception as e:
            emit(
                "collecting",
                "discovery_youtube",
                f"YouTube scan failed — {e}",
                status="error",
            )
            logger.info(f"[PIPELINE]   [discovery] youtube scan error: {e}")

    # ── Run all 3 scans in parallel (Fix #4) ──
    await asyncio.gather(_scan_news(), _scan_reddit(), _scan_youtube())

    # -- Write to discovered_tickers table --
    if seen:
        # Filter out hard-blocked FALSE_TICKERS before writing to DB
        # (but allow high-cap stocks like OPEN, FAST, RUN through)
        from app.processors.ticker_extractor import (
            _is_hard_blocked,
            get_registry as _get_disc_reg,
        )

        _disc_reg = _get_disc_reg()
        leaked = [t for t in seen if _is_hard_blocked(t, _disc_reg)]
        for t in leaked:
            del seen[t]
            all_contexts.pop(t, None)
            all_sources.pop(t, None)
        if leaked:
            logger.info(
                f"[PIPELINE]   [discovery] Filtered {len(leaked)} hard-blocked tickers before DB write: {', '.join(leaked[:10])}"
            )

        t0 = time.monotonic()
        written = 0
        with get_db() as db:
            for ticker, score in seen.items():
                try:
                    db.execute(
                        """
                        INSERT INTO discovered_tickers
                            (ticker, source, context, score, discovered_at)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (ticker) DO UPDATE
                        SET score = EXCLUDED.score,
                            source = EXCLUDED.source,
                            context = EXCLUDED.context,
                            discovered_at = CURRENT_TIMESTAMP
                    """,
                        [
                            ticker,
                            all_sources.get(ticker, "scan"),
                            all_contexts.get(ticker, ""),
                            score,
                        ],
                    )
                    written += 1
                except Exception:
                    pass
        ms = elapsed_ms(t0)
        emit(
            "collecting",
            "discovery_write",
            f"Wrote {written} tickers to discovered_tickers table",
            status="ok",
            data={"written": written},
            elapsed_ms=ms,
        )
        logger.info(f"[PIPELINE]   [discovery] wrote {written} tickers to DB")

        # Fix #8: Removed duplicate collect_metadata call.
        # Metadata enrichment is handled in Pass 3.5 on the full merged ticker list.

    # Return sorted by score descending
    sorted_tickers = sorted(seen.keys(), key=lambda t: seen[t], reverse=True)
    return len(seen), sorted_tickers

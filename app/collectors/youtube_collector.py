"""
YouTube Collector — Fetches transcripts from financial YouTube channels/search.

Pure data collector. No LLM calls in the base collection.
Writes to: youtube_transcripts, discovered_tickers, discovered_channels

No API key needed — uses scraper-service.
"""

import re
import datetime
import logging
import asyncio
from app.processors.ticker_extractor import (
    get_ticker_symbols,
    FALSE_TICKERS as SHARED_FALSE_TICKERS,
)
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# Default financial YouTube channels (seed list)
DEFAULT_CHANNELS = [
    # -- Institutional / Major News --
    ("markets", "Bloomberg Television"),
    ("CNBCtelevision", "CNBC Television"),
    ("Reuters", "Reuters"),
    ("YahooFinance", "Yahoo Finance"),
    ("FoxBusiness", "Fox Business"),
    # -- Analysis / Macro --
    ("TheCompoundNews", "The Compound and Friends"),
    ("FundstratTomLee", "Fundstrat - Tom Lee"),
    ("ARKInvest", "ARK Invest (Cathie Wood)"),
    ("PatrickBoyle", "Patrick Boyle"),
    ("EverythingMoney", "Everything Money"),
    ("GameofTrades", "Game of Trades"),
    # -- Retail / Popular --
    ("TickerSymbolYou", "Ticker Symbol: YOU"),
    ("InTheMoneyAdam", "InTheMoney"),
    ("ClearValueTax", "ClearValue Tax"),
    ("TheMotleyFool", "Motley Fool"),
    ("MeetKevin", "Meet Kevin"),
    ("GrahamStephan", "Graham Stephan"),
    ("TomNash", "Tom Nash"),
    ("StockMoe", "Stock Moe"),
]

# Regex for ticker mentions in transcripts
TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")


def _ensure_seed_channels():
    """Ensure all default channels exist in youtube_channels table."""
    with get_db() as db:
        added = 0
        for handle, name in DEFAULT_CHANNELS:
            result = db.execute(
                "SELECT 1 FROM youtube_channels WHERE channel_handle = %s", [handle]
            ).fetchone()
            if not result:
                db.execute(
                    """
                    INSERT INTO youtube_channels
                    (channel_handle, display_name, added_by, is_active)
                    VALUES (%s, %s, 'system', TRUE)
                """,
                    [handle, name],
                )
                added += 1
        if added:
            logger.info(
                f"[youtube] Added {added} new seed channels (total {len(DEFAULT_CHANNELS)})"
            )


def _is_channel_blocked(channel_handle: str) -> bool:
    """Check if a channel is on the blocklist."""
    with get_db() as db:
        blocked = db.execute(
            "SELECT 1 FROM discovered_channels WHERE channel_handle = %s AND status = 'blocked'",
            [channel_handle],
        ).fetchone()
        return blocked is not None


def _log_discovered_channel(
    channel_handle: str, display_name: str, view_count: int = 0
):
    """Log a channel found via search to discovered_channels for review."""
    with get_db() as db:
        existing = db.execute(
            "SELECT discovery_count, avg_view_count FROM discovered_channels WHERE channel_handle = %s",
            [channel_handle],
        ).fetchone()

        if existing:
            new_count = existing[0] + 1
            new_avg = ((existing[1] or 0) * existing[0] + view_count) / new_count
            db.execute(
                """
                UPDATE discovered_channels
                SET discovery_count = %s, avg_view_count = %s, last_seen = CURRENT_TIMESTAMP
                WHERE channel_handle = %s
            """,
                [new_count, new_avg, channel_handle],
            )
        else:
            db.execute(
                """
                INSERT INTO discovered_channels
                (channel_handle, display_name, discovery_count, avg_view_count, status)
                VALUES (%s, %s, 1, %s, 'pending')
            """,
                [channel_handle, display_name, view_count],
            )


async def collect_channel(
    channel: str,
    max_videos: int = 3,
    days_back: int = 7,
) -> dict[str, int]:
    """
    Get recent videos from a YouTube channel and extract transcripts via scraper-service.
    Returns dictionary of results.
    """
    from app.services.scraper_client import scraper_client

    stats = {"videos_found": 0, "skipped_old": 0, "skipped_in_db": 0, "no_caption": 0, "stored": 0, "blocked": 0, "missing_id": 0}

    with get_db() as db:
        if _is_channel_blocked(channel):
            logger.info(f"[youtube] {channel}: blocked, skipping")
            stats["blocked"] += 1
            return stats

        try:
            items = await scraper_client.collect(
                source="youtube",
                req_data={
                    "channels": [channel],
                    "limit": max_videos,
                    "days_back": days_back,
                }
            )

            stats["videos_found"] = len(items)

            for video in items:
                status = await _process_video(db, video, channel, days_back)
                if status in stats:
                    stats[status] += 1

            if stats.get("stored", 0) > 0:
                db.execute(
                    """
                    UPDATE youtube_channels SET last_scraped = CURRENT_TIMESTAMP, total_videos = total_videos + %s
                    WHERE channel_handle = %s
                """,
                    [stats["stored"], channel],
                )

        except Exception as e:
            logger.info(f"[youtube] {channel} error: {e}")

        return stats


async def collect_for_ticker(ticker: str, max_results: int = 15, since: datetime.datetime | None = None) -> dict[str, int]:
    """Search YouTube for recent videos about a specific ticker via scraper-service."""
    from app.services.scraper_client import scraper_client

    stats = {"videos_found": 0, "skipped_old": 0, "skipped_in_db": 0, "no_caption": 0, "stored": 0, "blocked": 0, "missing_id": 0}

    with get_db() as db:
        try:
            current_year = datetime.datetime.now().year
            search_queries = [
                f"{ticker} stock analysis {current_year}",
                f"{ticker} stock news today",
            ]

            for query in search_queries:
                await asyncio.sleep(2.0)

                items = await scraper_client.collect(
                    source="youtube",
                    req_data={
                        "query": query,
                        "limit": max_results,
                        "days_back": 90,
                    }
                )

                if not items:
                    logger.info(f"[youtube] search '{query}': 0 results from scraper-service")
                    continue

                logger.info(f"[youtube] search '{query}': got {len(items)} results")
                stats["videos_found"] += len(items)

                # Sort by published date descending
                items.sort(
                    key=lambda v: v.get("published_at") or "",
                    reverse=True,
                )

                for video in items:
                    channel_name = video.get("channel", "unknown")
                    channel_handle = video.get("channel", channel_name)

                    if _is_channel_blocked(channel_handle):
                        logger.info(f"[youtube] {channel_name}: blocked, skipping")
                        stats["blocked"] += 1
                        continue

                    is_seed = db.execute(
                        "SELECT 1 FROM youtube_channels WHERE channel_handle = %s",
                        [channel_handle],
                    ).fetchone()
                    if not is_seed:
                        view_count = video.get("view_count", 0) or 0
                        _log_discovered_channel(
                            channel_handle, channel_name, view_count
                        )

                    status = await _process_video(
                        db,
                        video,
                        channel_name,
                        days_back=90,
                        force_ticker=ticker.upper(),
                        since=since,
                    )
                    if status in stats:
                        stats[status] += 1

                if stats["stored"] >= 2:
                    logger.info(f"[youtube] search short-circuit: stored {stats['stored']} transcripts so far.")
                    break

        except Exception as e:
            logger.info(f"[youtube] search for {ticker} error: {e}")

        logger.info(f"[youtube] search for {ticker} finished. Stats: {stats}")
        return stats


async def _process_video(
    db,
    video: dict,
    channel: str,
    days_back: int = 7,
    force_ticker: str | None = None,
    since: datetime.datetime | None = None,
) -> str:
    """Process a single video: check date, get transcript, store."""
    video_id = video.get("video_id", video.get("id"))
    if not video_id:
        return "missing_id"

    title = video.get("title", "")
    channel_name = video.get("channel", channel)
    raw_transcript = video.get("transcript", "")
    thumbnail_url = video.get("thumbnail_url", video.get("thumbnail", ""))
    duration = video.get("duration_secs", video.get("duration", 0)) or 0

    if not raw_transcript or len(raw_transcript) < 50:
        return "no_caption"

    published_at = None
    pub_val = video.get("published_at")
    if isinstance(pub_val, str):
        try:
            published_at = datetime.datetime.fromisoformat(pub_val)
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=datetime.UTC)
        except ValueError:
            pass
    elif isinstance(pub_val, (int, float)):
        published_at = datetime.datetime.fromtimestamp(pub_val, tz=datetime.UTC)

    if since and published_at and published_at <= since:
        return "skipped_old"

    if published_at and days_back > 0:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days_back)
        if published_at < cutoff:
            return "skipped_old"

    try:
        from app.routers.data import is_blocked
        if is_blocked("youtube", video_id):
            return "blocked"
    except Exception:
        pass

    existing = db.execute(
        "SELECT video_id FROM youtube_transcripts WHERE video_id = %s", [video_id]
    ).fetchone()
    if existing:
        return "skipped_in_db"

    raw_transcript = _strip_promo_content(raw_transcript)

    primary_ticker = force_ticker or _extract_primary_ticker(
        title + " " + raw_transcript[:1000]
    )

    if not thumbnail_url:
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    db.execute(
        """
        INSERT INTO youtube_transcripts
        (video_id, ticker, title, channel, raw_transcript,
         thumbnail_url, published_at, duration_secs, collected_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (video_id) DO NOTHING
    """,
        [
            video_id,
            primary_ticker,
            title,
            channel_name,
            raw_transcript,
            thumbnail_url,
            published_at,
            duration,
        ],
    )

    tickers = _extract_tickers(raw_transcript[:5000])
    for t in tickers:
        if t in SHARED_FALSE_TICKERS or len(t) < 2:
            continue
        db.execute(
            """
            INSERT INTO discovered_tickers
            (ticker, source, context, score, discovered_at)
            VALUES (%s, 'youtube', %s, 0.80, %s)
            ON CONFLICT (ticker, source) DO NOTHING
        """,
            [
                t,
                f"{channel}: {title[:60]}",
                published_at or datetime.datetime.now(datetime.UTC),
            ],
        )

    logger.info(
        f"[youtube]   stored: '{title[:80]}' ({len(raw_transcript)} chars, ticker={primary_ticker})"
    )
    return "stored"


# Promotional content patterns
_PROMO_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"subscribe.*channel",
        r"hit.*bell.*notification",
        r"join.*telegram",
        r"join.*discord",
        r"follow.*(?:twitter|instagram|tiktok)",
        r"link.*(?:description|below|bio)",
        r"use.*code.*(?:discount|percent|off)",
        r"patreon\.com",
        r"(?:not|this is not).*financial.*advice",
        r"sign up.*free",
        r"sponsored.*by",
        r"check out.*(?:course|program|webinar)",
        r"limited.*time.*offer",
        r"affiliate.*link",
    ]
]


def _strip_promo_content(transcript: str) -> str:
    """Remove promotional/spam content from a YouTube transcript."""
    if not transcript or len(transcript) < 100:
        return transcript

    original_len = len(transcript)

    paragraphs = re.split(r"\n{2,}", transcript)
    if not paragraphs:
        return transcript

    cleaned_paragraphs = []
    stripped_sentences_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)

        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            is_promo = False
            for pattern in _PROMO_PATTERNS:
                if pattern.search(sentence):
                    is_promo = True
                    stripped_sentences_count += 1
                    break

            if not is_promo:
                cleaned_sentences.append(sentence)

        if cleaned_sentences:
            cleaned_paragraphs.append(" ".join(cleaned_sentences))

    separator = "\n\n" if len(paragraphs) > 1 else " "
    result = separator.join(cleaned_paragraphs)
    final_len = len(result)

    if stripped_sentences_count > 0:
        if final_len < (0.5 * original_len):
            return transcript
        logger.debug(
            f"[youtube] Cleaned transcript: {original_len} -> {final_len} chars "
            f"(-{stripped_sentences_count} promo sentences)"
        )

    return result


def _extract_tickers(text: str) -> set[str]:
    """Extract tickers from text using shared ticker_extractor."""
    return set(get_ticker_symbols(text))


def _extract_primary_ticker(text: str) -> str | None:
    """Get the most relevant ticker from title + beginning of transcript."""
    syms = get_ticker_symbols(text)
    return syms[0] if syms else None


# General financial search queries
GENERAL_SEARCH_QUERIES = [
    "stock market analysis today {year}",
    "market news earnings this week",
    "best stocks to buy now {year}",
    "market outlook investing this week",
    "stock market crash or rally {year}",
]


async def collect_all(
    max_videos: int = 3,
    days_back: int = 30,
    max_queries: int | None = None,
) -> int:
    """Collect transcripts via direct channel scraping + dynamic search."""
    import random
    from app.services.scraper_client import scraper_client

    _ensure_seed_channels()

    with get_db() as db:
        year = datetime.datetime.now().year
        # Fetch active channels (both handle and display_name)
        channels = db.execute(
            "SELECT channel_handle, display_name FROM youtube_channels WHERE is_active = TRUE"
        ).fetchall()

    total = 0

    # 1. Direct channel collection for all active channels
    logger.info(
        f"[youtube] Starting direct channel scrape for {len(channels)} active channels (max_videos={max_videos}, days_back={days_back})"
    )
    for i, (handle, name) in enumerate(channels):
        try:
            stats = await collect_channel(handle, max_videos=max_videos, days_back=days_back)
            stored_count = stats.get("stored", 0)
            total += stored_count
            logger.info(f"[youtube] Direct scrape '{handle}' ({name}): stored {stored_count} videos (stats={stats})")
        except Exception as e:
            logger.error(f"[youtube] Error scraping channel '{handle}': {e}")
        
        # Throttle between channel scrapes
        if i < len(channels) - 1:
            await asyncio.sleep(2.0)

    # 2. Search-based sweep for discovery
    channel_queries = [f'"{name}" stock analysis' for (_, name) in channels if name]
    general_queries = [q.format(year=year) for q in GENERAL_SEARCH_QUERIES]

    all_queries = channel_queries + general_queries
    random.shuffle(all_queries)

    if max_queries and len(all_queries) > max_queries:
        all_queries = all_queries[:max_queries]

    logger.info(
        f"[youtube] Search-based sweep: {len(all_queries)} queries (max_videos={max_videos}, days_back={days_back})"
    )

    seen_ids: set[str] = set()

    for i, query in enumerate(all_queries):
        try:
            items = await scraper_client.collect(
                source="youtube",
                req_data={
                    "query": query,
                    "limit": max_videos,
                    "days_back": days_back,
                }
            )

            if items:
                with get_db() as db:
                    for video in items:
                        vid = video.get("video_id", video.get("id"))
                        if not vid or vid in seen_ids:
                            continue
                        seen_ids.add(vid)

                        channel_name = video.get("channel", "unknown")
                        stored = await _process_video(db, video, channel_name, days_back)
                        if stored == "stored":
                            total += 1
        except Exception as e:
            logger.info(f"[youtube] Error in sweep query '{query}': {e}")

        if i < len(all_queries) - 1:
            await asyncio.sleep(2.0)

    logger.info(
        f"[youtube] Total collected this run: {total} transcripts"
    )
    return total

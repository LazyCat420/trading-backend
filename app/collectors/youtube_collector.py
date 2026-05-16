"""
YouTube Collector — Fetches transcripts from financial YouTube channels/search.

Pure data collector. No LLM calls in the base collection.
Writes to: youtube_transcripts, discovered_tickers, discovered_channels

Two modes:
  1. Channel mode: Scrape latest from curated seed list channels
  2. Search mode:  Search YouTube for ticker-specific videos (ytsearch:$TICKER)

Libraries: yt-dlp (video search/metadata), youtube-transcript-api (captions)
No API key needed — uses public YouTube data.
"""

import re
import subprocess
import json
import datetime
import logging
from app.processors.ticker_extractor import (
    get_ticker_symbols,
    FALSE_TICKERS as SHARED_FALSE_TICKERS,
)

logger = logging.getLogger(__name__)

# ── yt-dlp version check at import ──
import sys

try:
    _v = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--version"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    _YTDLP_VERSION = _v.stdout.strip() if _v.returncode == 0 else "unknown"
    logger.info(f"[youtube] yt-dlp version: {_YTDLP_VERSION}")
except Exception:
    _YTDLP_VERSION = "not-found"
    logger.info("[youtube] WARNING: yt-dlp not found in PATH")
from app.db.connection import get_db

# Default financial YouTube channels (seed list)
# These get loaded into youtube_channels table on first run
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

# Known good tickers to match in transcripts
TICKER_KEYWORDS = {
    "NVDA",
    "AAPL",
    "TSLA",
    "MSFT",
    "GOOG",
    "GOOGL",
    "AMZN",
    "META",
    "AMD",
    "PLTR",
    "SOFI",
    "SMCI",
    "ARM",
    "AVGO",
    "MRVL",
    "INTC",
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "JPM",
    "BAC",
    "GS",
    "WFC",
    "DIS",
    "NFLX",
    "COST",
    "WMT",
    "TGT",
}

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
    Get recent videos from a YouTube channel and extract transcripts.
    Uses yt-dlp for video metadata and youtube-transcript-api for captions.
    Returns dictionary of results.
    """
    import asyncio

    stats = {"videos_found": 0, "skipped_old": 0, "skipped_in_db": 0, "no_caption": 0, "stored": 0, "blocked": 0, "missing_id": 0}

    with get_db() as db:
        # Check blocklist
        if _is_channel_blocked(channel):
            logger.info(f"[youtube] {channel}: blocked, skipping")
            return stats

        try:
            result = await asyncio.to_thread(_get_channel_videos, channel, max_videos)

            if not result:
                return stats
                
            stats["videos_found"] = len(result)

            for video in result:
                status = await _process_video(db, video, channel, days_back)
                if status in stats:
                    stats[status] += 1
                elif status:
                    stats[status] = stats.get(status, 0) + 1

            # Update last_scraped
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
    """Search YouTube for recent videos about a specific ticker.

    Uses yt-dlp ytsearch: to find videos from ANY channel.
    Checks blocklist before storing. Logs unknown channels to discovered_channels.
    Returns dictionary of results.
    """
    import asyncio

    stats = {"videos_found": 0, "skipped_old": 0, "skipped_in_db": 0, "no_caption": 0, "stored": 0, "blocked": 0, "missing_id": 0}

    with get_db() as db:
        try:
            current_year = datetime.datetime.now().year
            # Broader search queries to increase hit rate
            search_queries = [
                f"{ticker} stock analysis {current_year}",
                f"{ticker} stock news today",
            ]

            for query in search_queries:
                # Rate limit: 2s between search calls to avoid 429
                await asyncio.sleep(2.0)

                videos = await asyncio.to_thread(_search_youtube, query, max_results, since)

                if not videos:
                    logger.info(f"[youtube] search '{query}': 0 results from yt-dlp")
                    continue

                logger.info(f"[youtube] search '{query}': got {len(videos)} results")
                stats["videos_found"] += len(videos)

                # Sort by upload date descending so we prioritize newer videos
                videos.sort(
                    key=lambda v: v.get("upload_date", "00000000"),
                    reverse=True,
                )

                for video in videos:
                    channel_name = video.get(
                        "channel", video.get("uploader", "unknown")
                    )
                    channel_handle = video.get("channel_id", channel_name)

                    # Check blocklist
                    if _is_channel_blocked(channel_handle):
                        logger.info(f"[youtube] {channel_name}: blocked, skipping")
                        stats["blocked"] += 1
                        continue

                    # Log to discovered_channels if not a seed channel
                    is_seed = db.execute(
                        "SELECT 1 FROM youtube_channels WHERE channel_handle = %s",
                        [channel_handle],
                    ).fetchone()
                    if not is_seed:
                        view_count = video.get("view_count", 0) or 0
                        _log_discovered_channel(
                            channel_handle, channel_name, view_count
                        )

                    # Rate limit: 1s between transcript fetches
                    await asyncio.sleep(1.0)

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
                    elif status:
                        stats[status] = stats.get(status, 0) + 1
                
                # Short-circuit logic: if this query alone stored enough, break
                if stats["stored"] >= 2:
                    logger.info(f"[youtube] search short-circuit: stored {stats['stored']} transcripts so far. Skipping remaining queries.")
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
    """Process a single video: check date, get transcript, store.
    Returns status string (e.g. 'stored', 'skipped_in_db', etc.).
    """
    import asyncio

    # Extract video ID — yt-dlp uses 'id' for --dump-json, but --flat-playlist
    # may use 'url' or 'webpage_url' depending on version
    video_id = video.get("id")
    if not video_id:
        # Try extracting from URL fields
        for url_key in ("url", "webpage_url", "original_url"):
            url_val = video.get(url_key, "")
            if "watch?v=" in url_val:
                video_id = url_val.split("watch?v=")[-1].split("&")[0]
                break
            elif url_val and len(url_val) == 11:  # Raw video ID
                video_id = url_val
                break
    if not video_id:
        logger.info(
            f"[youtube]   skip: no video_id in metadata for '{video.get('title', '?')[:60]}'"
        )
        logger.info(f"[youtube]         available keys: {list(video.keys())}")
        return "missing_id"

    title = video.get("title", "")
    upload_date = video.get("upload_date", "")
    duration = video.get("duration", 0)
    channel_name = video.get("channel", channel)

    # Parse upload date
    published_at = None
    if upload_date:
        try:
            published_at = datetime.datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=datetime.UTC)
        except ValueError:
            pass

    if since and published_at and published_at <= since:
        return "skipped_old"

    # Skip old videos
    if published_at and days_back > 0:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days_back)
        if published_at < cutoff:
            # Only log for channel mode (small days_back), not search mode
            if days_back <= 30:
                logger.debug(
                    f"skip: '{title[:50]}' too old ({upload_date}, cutoff={days_back}d)"
                )
            else:
                logger.info(
                    f"[youtube]   skip: '{title[:50]}' too old ({upload_date}, cutoff={days_back}d)"
                )
            return "skipped_old"

    # Check if blocklisted (user previously deleted this video)
    try:
        from app.routers.data import is_blocked

        if is_blocked("youtube", video_id):
            logger.info(
                f"[youtube]   skip: '{title[:50]}' is blocklisted (previously deleted)"
            )
            return "blocked"
    except Exception:
        pass

    # Check if already in DB
    existing = db.execute(
        "SELECT video_id FROM youtube_transcripts WHERE video_id = %s", [video_id]
    ).fetchone()
    if existing:
        logger.info(f"[youtube]   skip: '{title[:50]}' already in DB (id={video_id})")
        return "skipped_in_db"

    # Get transcript
    transcript = await asyncio.to_thread(_get_transcript, video_id)
    if not transcript:
        logger.info(
            f"[youtube]   skip: no transcript for '{title[:50]}' (id={video_id})"
        )
        return "no_caption"

    # Strip promotional spam paragraphs before storage
    raw_transcript = _strip_promo_content(transcript)

    # Determine primary ticker
    primary_ticker = force_ticker or _extract_primary_ticker(
        title + " " + raw_transcript[:1000]
    )

    # Get thumbnail URL from yt-dlp metadata or construct from video ID
    thumbnail_url = (
        video.get("thumbnail") or f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    )

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

    # Extract ALL discovered tickers from transcript (filter through shared FALSE_TICKERS)
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


def _get_channel_videos(channel: str, max_videos: int) -> list[dict]:
    """Use yt-dlp to get recent video metadata from a channel."""
    try:
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            f"https://www.youtube.com/@{channel}/videos",
            "--flat-playlist",
            "--dump-json",
            f"--playlist-end={max_videos}",
            "--no-download",
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            if result.stderr:
                err = result.stderr[:200]
                logger.info(f"[youtube] yt-dlp channel error for {channel}: {err}")

                # Log handle failure to trigger agent (with dedup check)
                if "404" in err or "does not have a videos tab" in err:
                    try:
                        from app.db.connection import get_db

                        with get_db() as db:
                            # Check for existing pending issue for this channel
                            existing = db.execute(
                                "SELECT 1 FROM pending_evolution_fixes "
                                "WHERE target_name = 'youtube_channel_handle' "
                                "AND motivation LIKE %s AND status = 'pending'",
                                (f"%Channel {channel}%",),
                            ).fetchone()
                            if not existing:
                                db.execute(
                                    "INSERT INTO pending_evolution_fixes (id, cycle_id, target_type, target_name, proposed_fix, motivation, status) "
                                    "VALUES (gen_random_uuid()::text, 'auto', 'scraper_issue', 'youtube_channel_handle', '', %s, 'pending')",
                                    (
                                        f"Channel {channel} failed with {err}. Agent should use youtube_search_handle to find new handle and update DB.",
                                    ),
                                )
                            else:
                                logger.info(
                                    f"[youtube] Skipping duplicate scraper alert for channel {channel}"
                                )
                    except Exception as e:
                        logger.info(f"[youtube] Failed to log scraper alert: {e}")

            return []

        videos = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    videos.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return videos

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.info(f"[youtube] yt-dlp error for {channel}: {e}")
        return []


def _search_youtube(query: str, max_results: int = 5, since: datetime.datetime | None = None) -> list[dict]:
    """Use yt-dlp ytsearch to find videos matching a query.
    """
    try:
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            f"ytsearch{max_results}:{query}",
            "--dump-json",
            "--no-download",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--socket-timeout",
            "15",
        ]
        logger.info(
            f"[youtube] Searching: {query} (max {max_results})"
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else ""
            if stderr:
                logger.info(f"[youtube] yt-dlp search error: {stderr[:300]}")
            return []

        if not result.stdout.strip():
            logger.info(f"[youtube] search '{query}': yt-dlp returned empty stdout")
            return []

        videos = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    videos.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        logger.info(f"[youtube] Found {len(videos)} videos for '{query}'")
        return videos

    except subprocess.TimeoutExpired:
        logger.info(f"[youtube] Search timed out for '{query}'")
        return []
    except FileNotFoundError:
        logger.info("[youtube] yt-dlp not found! pip install yt-dlp")
        return []


def _get_transcript(video_id: str) -> str | None:
    """Get transcript for a YouTube video.

    Strategy (in order):
    1. yt-dlp --write-auto-sub (downloads subtitle file, no extra API call)
    2. youtube-transcript-api (fallback, may be IP-blocked)
    3. Playwright browser (network interception + DOM scraping, ultimate fallback)
    """
    errors = []

    # Method 1: yt-dlp subtitle download (most reliable)
    transcript, err1 = _get_transcript_ytdlp(video_id)
    if transcript:
        return transcript
    if err1:
        errors.append(f"ytdlp: {err1}")

    # Method 2: youtube-transcript-api (fallback)
    transcript, err2 = _get_transcript_api(video_id)
    if transcript:
        return transcript
    if err2:
        errors.append(f"api: {err2}")

    # Method 3: Playwright (network interception, adapted from Youtube-News-Extracter)
    logger.info(f"[youtube]   Trying Playwright for {video_id}...")
    try:
        from app.collectors.youtube_playwright import scrape_transcript_async
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            pass

        text = asyncio.run(scrape_transcript_async(video_id, headless=True))
        if text and len(text) > 100:
            logger.info(f"[youtube]   Playwright OK for {video_id}: {len(text)} chars")
            return text
        else:
            errors.append("playwright: no text extracted")
            logger.info(f"[youtube]   Playwright failed for {video_id}")
    except Exception as e:
        errors.append(f"playwright error: {type(e).__name__}")
        logger.info(f"[youtube]   Playwright fallback error for {video_id}: {e}")

    logger.info(f"[youtube]   FAIL: no transcript from any method for {video_id}. Errors: {errors}")

    # Log failure to DB so the autonomous agent can fix it later (with dedup check)
    try:
        from app.db.connection import get_db

        with get_db() as db:
            # Check for existing pending issue for this video
            existing = db.execute(
                "SELECT 1 FROM pending_evolution_fixes "
                "WHERE target_name = 'youtube_transcript' "
                "AND motivation LIKE %s AND status IN ('pending', 'error')",
                (f"%{video_id}%",),
            ).fetchone()
            if not existing:
                db.execute(
                    "INSERT INTO pending_evolution_fixes (id, cycle_id, target_type, target_name, proposed_fix, motivation, status) "
                    "VALUES (gen_random_uuid()::text, 'auto', 'scraper_issue', 'youtube_transcript', '', %s, 'pending')",
                    (
                        f"Video ID {video_id} failed transcript extraction across all methods. Agent should write a Playwright script to fix it.",
                    ),
                )
            else:
                logger.info(
                    f"[youtube]   Skipping duplicate scraper alert for video {video_id}"
                )
    except Exception as e:
        logger.info(f"[youtube]   Failed to log scraper alert: {e}")

    return None


def _get_transcript_ytdlp(video_id: str) -> tuple[str | None, str | None]:
    """Get transcript using yt-dlp subtitle download.

    Downloads auto-generated subs as JSON3 to a temp file, parses text.
    This avoids the youtube-transcript-api which gets IP-blocked.
    """
    import tempfile
    import os

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "sub")
            cmd = [
                sys.executable,
                "-m",
                "yt_dlp",
                f"https://www.youtube.com/watch?v={video_id}",
                "--skip-download",
                "--write-auto-sub",
                "--write-subs",
                "--sub-lang",
                "en.*",
                "--sub-format",
                "json3",
                "--no-warnings",
                "--socket-timeout",
                "15",
                "-o",
                output_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )

            # Look for the subtitle file (yt-dlp adds .en.json3 suffix)
            sub_file = None
            for f in os.listdir(tmpdir):
                if f.endswith(".json3") or f.endswith(".vtt"):
                    sub_file = os.path.join(tmpdir, f)
                    break

            if not sub_file:
                err = result.stderr.strip()[:200] if result.stderr else "no sub file written"
                logger.info(f"[youtube]   yt-dlp subs failed for {video_id}: {err}")
                return None, err

            # Parse JSON3 subtitle format if applicable
            if sub_file.endswith(".json3"):
                with open(sub_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                parts = []
                for event in data.get("events", []):
                    for seg in event.get("segs", []):
                        text = seg.get("utf8", "").strip()
                        if text and text != "\n":
                            parts.append(text)
                transcript = " ".join(parts).strip()
            else:
                # Basic VTT parsing
                with open(sub_file, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
                parts = [line.strip() for line in lines if "-->" not in line and not line.startswith("WEBVTT") and line.strip()]
                transcript = " ".join(parts).strip()

            if len(transcript) > 50:  # Minimum viable transcript
                logger.info(
                    f"[youtube]   yt-dlp subs OK for {video_id}: {len(transcript)} chars"
                )
                return transcript, None
            return None, "transcript too short"

    except subprocess.TimeoutExpired:
        logger.info(f"[youtube]   yt-dlp subs timeout for {video_id}")
        return None, "timeout"
    except Exception as e:
        logger.info(f"[youtube]   yt-dlp subs error for {video_id}: {e}")
        return None, type(e).__name__


def _get_transcript_api(video_id: str) -> tuple[str | None, str | None]:
    """Fallback: Get transcript using youtube-transcript-api."""
    try:
        import os
        from youtube_transcript_api import YouTubeTranscriptApi

        cookies_file = os.environ.get("YOUTUBE_COOKIES_FILE", "")
        if cookies_file and os.path.exists(cookies_file):
            ytt = YouTubeTranscriptApi(cookie_path=cookies_file)
        else:
            ytt = YouTubeTranscriptApi()

        err_en = ""
        # Try English first
        try:
            transcript = ytt.fetch(video_id, languages=["en"])
            text = _transcript_to_text(transcript)
            if text and len(text) > 50:
                logger.info(
                    f"[youtube]   transcript-api OK for {video_id}: {len(text)} chars"
                )
                return text, None
        except Exception as e:
            err_en = type(e).__name__
            logger.info(
                f"[youtube]   transcript-api en failed for {video_id}: {err_en}"
            )

        err_list = ""
        # Try any available language
        try:
            transcript_list = ytt.list(video_id)
            for t in transcript_list:
                try:
                    fetched = t.fetch()
                    text = _transcript_to_text(fetched)
                    if text and len(text) > 50:
                        return text, None
                except Exception:
                    continue
        except Exception as e:
            err_list = type(e).__name__
            logger.info(
                f"[youtube]   transcript-api list failed for {video_id}: {err_list}"
            )

        return None, f"en: {err_en}, list: {err_list}"
    except Exception as e:
        logger.info(f"[youtube]   transcript-api error for {video_id}: {e}")
        return None, type(e).__name__


def _transcript_to_text(transcript) -> str:
    """Convert a transcript object/list to plain text."""
    parts = []
    for snippet in transcript:
        if isinstance(snippet, dict):
            parts.append(snippet.get("text", ""))
        else:
            parts.append(getattr(snippet, "text", str(snippet)))
    return " ".join(parts)


# ── Promotional content patterns ──
# Matches paragraphs containing self-promotion spam
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
    """Remove promotional/spam content from a YouTube transcript.

    Splits paragraphs into sentences and drops only the specific sentences that match promo patterns.
    Safeguard: if cleaning removes >50% of the text, abort to preserve the original.
    """
    if not transcript or len(transcript) < 100:
        return transcript

    original_len = len(transcript)

    # Split into rough paragraphs (2+ newlines)
    paragraphs = re.split(r"\n{2,}", transcript)
    if not paragraphs:
        return transcript

    cleaned_paragraphs = []
    stripped_sentences_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split paragraph into sentences. We keep sentences intact by splitting after punctuation.
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

    # Reassemble with paragraph breaks or single spaces if originally just sentences
    separator = "\n\n" if len(paragraphs) > 1 else " "
    result = separator.join(cleaned_paragraphs)
    final_len = len(result)

    if stripped_sentences_count > 0:
        # Safeguard: if cleaning dropped >50% of the raw chars
        if final_len < (0.5 * original_len):
            logger.info(
                f"[youtube] Promo stripping too destructive "
                f"({original_len} -> {final_len} chars). Reverting to original."
            )
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


# General financial search queries for broad market intelligence
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
    """Collect transcripts via dynamic YouTube search.

    Instead of scraping hardcoded @handle/videos URLs (which 404 when
    channels rename/rebrand), we search YouTube dynamically using:
      1. Channel display names as search keywords (e.g. "Bloomberg Television")
      2. General financial search terms ("stock market analysis today")

    The channel names from youtube_channels are still valuable — only the
    URL handles go stale. This approach is resilient to handle changes.

    Args:
        max_videos: Max results per search query (yt-dlp ytsearch).
        days_back: Only store videos published within this many days.
        max_queries: Cap total search queries (for intensity control).
                     None = no cap (all queries).
    """
    import asyncio
    import random

    _ensure_seed_channels()

    with get_db() as db:
        year = datetime.datetime.now().year

        # ── Build search query pool ──
        # 1. Channel-name-based queries: use display names from DB
        channels = db.execute(
            "SELECT display_name FROM youtube_channels WHERE is_active = TRUE"
        ).fetchall()
        channel_queries = [f'"{name}" stock analysis' for (name,) in channels if name]

        # 2. General financial queries
        general_queries = [q.format(year=year) for q in GENERAL_SEARCH_QUERIES]

        # Combine and shuffle so we get variety across cycles
        all_queries = channel_queries + general_queries
        random.shuffle(all_queries)

        # Apply query cap (driven by pipeline intensity)
        if max_queries and len(all_queries) > max_queries:
            all_queries = all_queries[:max_queries]

        logger.info(
            f"[youtube] Search-based sweep: {len(all_queries)} queries "
            f"(max_videos={max_videos}, days_back={days_back})"
        )

        total = 0
        seen_ids: set[str] = set()  # Cross-query dedup within this cycle

        for i, query in enumerate(all_queries):
            videos = await asyncio.to_thread(_search_youtube, query, max_videos)

            if videos:
                for video in videos:
                    vid = video.get("id")
                    if not vid or vid in seen_ids:
                        continue
                    seen_ids.add(vid)

                    channel_name = video.get(
                        "channel", video.get("uploader", "unknown")
                    )
                    stored = await _process_video(db, video, channel_name, days_back)
                    if stored:
                        total += 1

            # Rate limit between search queries (yt-dlp subprocess is heavy)
            if i < len(all_queries) - 1:
                await asyncio.sleep(3.0)

        logger.info(
            f"[youtube] Total: {total} transcripts from "
            f"{len(all_queries)} search queries ({len(seen_ids)} unique videos seen)"
        )
        return total

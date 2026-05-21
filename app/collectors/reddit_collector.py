"""
Reddit Collector -- Fetches posts from financial subreddits via scraper-service.

Pure data collector. No LLM calls in the base collection.
Writes to: reddit_posts, discovered_tickers

No API key needed -- uses scraper-service.
"""

import logging
import hashlib
import re
import datetime
import asyncio
from app.db.connection import get_db
from app.processors.ticker_extractor import (
    get_ticker_symbols,
    FALSE_TICKERS as SHARED_FALSE_TICKERS,
)

logger = logging.getLogger(__name__)

# Reverse lookup: ticker -> company names (for searching by company name)
_TICKER_TO_NAMES: dict[str, list[str]] = {}


def _get_company_names(ticker: str) -> list[str]:
    """Get company names for a ticker to use as search queries."""
    global _TICKER_TO_NAMES
    if not _TICKER_TO_NAMES:
        try:
            from app.collectors.news_collector import COMPANY_TICKERS

            for name, t in COMPANY_TICKERS.items():
                _TICKER_TO_NAMES.setdefault(t, []).append(name)
        except ImportError:
            pass
    return _TICKER_TO_NAMES.get(ticker.upper(), [])


# Subreddits to scan for financial intel
SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "SecurityAnalysis",
    "StockMarket",
    "Daytrading",
    "algotrading",
    "CryptoCurrency",
    "pennystocks",
]

# High-signal subs to prioritize in ticker-specific search
PRIORITY_SUBS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "SecurityAnalysis",
    "StockMarket",
    "ValueInvesting",
    "Dividends",
    "smallstreetbets",
]

# Common false positives to filter out
FALSE_TICKERS = SHARED_FALSE_TICKERS


def _is_quality_post(post: dict, min_score: int = 3, min_comments: int = 2) -> bool:
    """Fast deterministic filter -- no LLM needed.

    Filters out memes, low-effort, deleted, and below-threshold posts.
    Returns True if the post is worth considering for storage.
    """
    score = post.get("score", 0)
    comments = post.get("num_comments", 0)
    body = post.get("body", post.get("selftext", ""))
    body_len = len(body)

    # Skip removed/deleted posts
    if body in ("[removed]", "[deleted]"):
        return False

    # Skip NSFW
    if post.get("over_18"):
        return False

    # Minimum engagement thresholds
    if score < min_score:
        return False
    if comments < min_comments:
        return False

    # Title-only posts with no substance -- skip unless very high engagement
    if body_len < 50 and score < 50:
        return False

    return True


def _is_relevant_to_ticker(post: dict, ticker: str) -> bool:
    """Check if a post is actually about the ticker, not just incidentally mentioning it."""
    subreddit = post.get("subreddit", "").lower()
    title = post.get("title", "")
    body = post.get("body", post.get("selftext", ""))
    full_text = f"{title} {body}"

    # If it's from a non-financial sub, require very strong ticker presence
    financial_subs = {
        s.lower()
        for s in SUBREDDITS
        + PRIORITY_SUBS
        + [
            "valueinvesting",
            "dividends",
            "stockanalysis",
            "thetagang",
            "superstonk",
            "weedstocks",
            "spacs",
            "smallstreetbets",
        ]
    }

    if subreddit not in financial_subs:
        # Non-financial sub: require $TICKER notation or ticker in title
        if f"${ticker}" not in full_text and ticker not in title.upper().split():
            return False

    # Check ticker appears in meaningful context (not just a random mention)
    ticker_upper = ticker.upper()
    mentions = 0

    # Count $TICKER mentions (strongest signal)
    mentions += full_text.count(f"${ticker_upper}")

    # Count bare ticker in title (strong signal)
    if ticker_upper in title.upper().split():
        mentions += 2

    # Count bare ticker in body (weak signal)
    body_words = body.upper().split()
    mentions += body_words.count(ticker_upper)

    return mentions >= 1


async def collect_subreddit(
    subreddit: str,
    time_filter: str = "day",
    limit: int = 25,
    ticker_filter: str | None = None,
) -> int:
    """
    Fetch top posts from a subreddit via scraper-service and write to reddit_posts.
    Returns number of posts written.
    """
    from app.services.scraper_client import scraper_client

    with get_db() as db:
        count = 0
        try:
            items = await scraper_client.collect(
                source="reddit",
                req_data={
                    "subreddits": [subreddit],
                    "limit": limit,
                    "time_filter": time_filter,
                    "sort": "top",
                }
            )

            for post in items:
                title = post.get("title", "")
                body = post.get("body", post.get("selftext", ""))
                full_text = f"{title} {body}"

                # Extract ticker mentions (shared extractor)
                tickers_found = set(get_ticker_symbols(full_text, title=title))

                # If filtering by ticker, skip posts without it
                if ticker_filter and ticker_filter.upper() not in tickers_found:
                    if ticker_filter.upper() not in full_text.upper():
                        continue

                # Assign primary ticker
                primary_ticker = (
                    ticker_filter.upper()
                    if ticker_filter
                    else (list(tickers_found)[0] if tickers_found else None)
                )

                if not primary_ticker:
                    continue

                count += _store_post(
                    db, post, primary_ticker, subreddit, tickers_found
                )

        except Exception as e:
            logger.info(f"[reddit] r/{subreddit} error: {e}")

        return count


async def search_subreddit_for_ticker(
    subreddit: str,
    ticker: str,
    queries: list[str],
    time_filter: str = "week",
    limit: int = 10,
    since: datetime.datetime | None = None,
) -> int:
    """Search WITHIN a specific subreddit list for a ticker via scraper-service."""
    from app.services.scraper_client import scraper_client

    with get_db() as db:
        count = 0
        seen_ids = set()

        try:
            subreddits = [s.strip() for s in subreddit.split("+") if s.strip()]

            for query in queries:
                items = await scraper_client.collect(
                    source="reddit",
                    req_data={
                        "query": query,
                        "subreddits": subreddits,
                        "limit": limit,
                        "time_filter": time_filter,
                    }
                )

                for post in items:
                    post_id = post.get("id", "")
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)

                    created_val = post.get("created_at", post.get("created_utc"))
                    if isinstance(created_val, (int, float)):
                        created_utc = datetime.datetime.fromtimestamp(created_val, tz=datetime.UTC)
                    elif isinstance(created_val, str):
                        created_utc = datetime.datetime.fromisoformat(created_val)
                        if created_utc.tzinfo is None:
                            created_utc = created_utc.replace(tzinfo=datetime.UTC)
                    else:
                        created_utc = datetime.datetime.now(datetime.UTC)

                    if since and created_utc <= since:
                        continue

                    if not _is_quality_post(post):
                        continue

                    if not _is_relevant_to_ticker(post, ticker):
                        continue

                    title = post.get("title", "")
                    body = post.get("body", post.get("selftext", ""))
                    full_text = f"{title} {body}"
                    tickers_found = set(get_ticker_symbols(full_text, title=title))
                    tickers_found.add(ticker.upper())

                    actual_sub = post.get("subreddit", subreddits[0] if subreddits else "")
                    count += _store_post(
                        db, post, ticker.upper(), actual_sub, tickers_found
                    )

                await asyncio.sleep(1.0)  # Pace between queries

        except Exception as e:
            logger.info(f"[reddit] r/{subreddit} search error: {e}")

        return count


async def collect_for_ticker(ticker: str, since: datetime.datetime | None = None) -> int:
    """Collect Reddit posts about a specific ticker using subreddit-scoped search."""
    ticker_upper = ticker.upper()

    # Core query variants
    queries = [
        f"${ticker_upper}",  # $NVDA (strongest signal)
        f"{ticker_upper} stock",  # NVDA stock
        f"{ticker_upper} earnings",  # NVDA earnings
        f"{ticker_upper} analysis",  # NVDA analysis
        f"{ticker_upper} DD",  # NVDA DD (due diligence)
    ]

    # Add company name queries (e.g. "Nvidia stock", "Apple earnings")
    company_names = _get_company_names(ticker_upper)
    for name in company_names[:2]:  # Top 2 names to avoid spam
        queries.append(f"{name} stock")

    total = 0
    multi_sub = "+".join(PRIORITY_SUBS)
    count = await search_subreddit_for_ticker(
        subreddit=multi_sub,
        ticker=ticker_upper,
        queries=queries,
        time_filter="month",  # Expanded from week for more data
        limit=25,  # Higher limit since we search all combined
        since=since,
    )
    if count > 0:
        logger.info(f"[reddit] multi-sub: {count} posts for {ticker_upper}")
    total += count

    subs = len(PRIORITY_SUBS)
    q = len(queries)
    logger.info(
        f"[reddit] {ticker_upper}: {total} posts ({subs} combined subs, {q} queries)"
    )
    return total


def _store_post(
    db, post: dict, primary_ticker: str, subreddit: str, tickers_found: set
) -> int:
    """Store a Reddit post and its discovered tickers. Returns 1 on success, 0 on skip."""
    title = post.get("title", "")
    body = post.get("body", post.get("selftext", ""))
    post_id = post.get("id", hashlib.md5(title.encode()).hexdigest()[:12])

    created_val = post.get("created_at", post.get("created_utc"))
    if isinstance(created_val, (int, float)):
        created_utc = datetime.datetime.fromtimestamp(created_val, tz=datetime.UTC)
    elif isinstance(created_val, str):
        created_utc = datetime.datetime.fromisoformat(created_val)
        if created_utc.tzinfo is None:
            created_utc = created_utc.replace(tzinfo=datetime.UTC)
    else:
        created_utc = datetime.datetime.now(datetime.UTC)

    score = post.get("score", 0)
    upvote_ratio = post.get("upvote_ratio", 0.0)
    num_comments = post.get("num_comments", 0)
    flair = post.get("flair", post.get("link_flair_text", ""))
    awards = post.get("awards", post.get("total_awards_received", 0))

    # Comment velocity (rough: comments / hours since posted)
    age_hours = max(
        (datetime.datetime.now(datetime.UTC) - created_utc).total_seconds() / 3600, 0.1
    )
    comment_velocity = num_comments / age_hours

    try:
        db.execute(
            """
            INSERT INTO reddit_posts
            (id, ticker, subreddit, title, body, score, upvote_ratio,
             comment_count, flair, sentiment_score, award_count,
             comment_velocity, created_utc, collected_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (id) DO NOTHING
        """,
            [
                post_id,
                primary_ticker,
                subreddit,
                title,
                body,
                score,
                upvote_ratio,
                num_comments,
                flair,
                None,
                awards,
                round(comment_velocity, 2),
                created_utc,
            ],
        )

        # Write discovered tickers (filter through shared FALSE_TICKERS)
        for t in tickers_found:
            if t in FALSE_TICKERS or len(t) < 2:
                continue
            confidence = min(round(upvote_ratio, 2), 1.0) if upvote_ratio else 0.5
            db.execute(
                """
                INSERT INTO discovered_tickers
                (ticker, source, context, score, discovered_at)
                VALUES (%s, 'reddit', %s, %s, %s)
            ON CONFLICT (ticker, source) DO NOTHING
            """,
                [t, f"r/{subreddit}: {title[:80]}", confidence, created_utc],
            )

        return 1
    except Exception as e:
        logger.info(f"[reddit] store error: {e}")
        return 0


async def collect_all(
    time_filter: str = "day",
    limit: int = 25,
    ticker_filter: str | None = None,
) -> int:
    """Scan all financial subreddits (general sweep). Returns total posts written."""
    total = 0
    for sub in SUBREDDITS:
        count = await collect_subreddit(sub, time_filter, limit, ticker_filter)
        total += count
        if count > 0:
            logger.info(f"[reddit] r/{sub}: {count} posts")
        await asyncio.sleep(2.0)
    logger.info(f"[reddit] Total: {total} posts across {len(SUBREDDITS)} subreddits")
    return total

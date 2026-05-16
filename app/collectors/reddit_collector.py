"""
Reddit Collector -- Fetches posts from financial subreddits via Reddit's public JSON API.

Pure data collector. No LLM calls in the base collection.
Writes to: reddit_posts, discovered_tickers

Three search strategies:
  1. General sweep: Top posts from each financial subreddit
  2. Subreddit-scoped search: Search within EACH financial sub for a ticker
  3. Multi-query search: Multiple query variants to catch different post styles

No API key needed -- uses Reddit's public .json endpoint.
"""

import logging

logger = logging.getLogger(__name__)


import hashlib
import re
import datetime
import asyncio
from tenacity import retry, stop_after_attempt, wait_fixed
from app.db.connection import get_db
from app.services.request_utils import SmartClient
from app.processors.ticker_extractor import (
    get_ticker_symbols,
    FALSE_TICKERS as SHARED_FALSE_TICKERS,
)

# Reverse lookup: ticker -> company names (for searching by company name)
# Uses the same map as news_collector to stay consistent
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
# (these are more likely to have quality analysis posts)
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

# Regex for $TICKER mentions
TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
# Also match bare uppercase tickers in financial context
BARE_TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

# Common false positives to filter out
FALSE_TICKERS = SHARED_FALSE_TICKERS

# Known real tickers that might look like common words
KNOWN_TICKERS = {
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
    "V",
    "MA",
    "PYPL",
    "SQ",
    "COIN",
}


def _is_quality_post(post: dict) -> bool:
    """Fast deterministic filter -- no LLM needed.

    Filters out memes, low-effort, deleted, and below-threshold posts.
    Returns True if the post is worth considering for storage.
    """
    score = post.get("score", 0)
    comments = post.get("num_comments", 0)
    body = post.get("selftext", "")
    body_len = len(body)

    # Skip removed/deleted posts
    if body in ("[removed]", "[deleted]"):
        return False

    # Skip NSFW
    if post.get("over_18"):
        return False

    # Minimum engagement thresholds
    if score < 3:
        return False
    if comments < 2:
        return False

    # Title-only posts with no substance -- skip unless very high engagement
    if body_len < 50 and score < 50:
        return False

    return True


def _is_relevant_to_ticker(post: dict, ticker: str) -> bool:
    """Check if a post is actually about the ticker, not just incidentally mentioning it.

    This is the key filter that prevents noise like r/science posts from getting through.
    Uses the subreddit as a strong signal.
    """
    subreddit = post.get("subreddit", "").lower()
    title = post.get("title", "")
    body = post.get("selftext", "")
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


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def collect_subreddit(
    subreddit: str,
    time_filter: str = "day",
    limit: int = 25,
    ticker_filter: str | None = None,
) -> int:
    """
    Fetch top posts from a subreddit and write to reddit_posts.
    Returns number of posts written.
    """
    with get_db() as db:
        url = f"https://www.reddit.com/r/{subreddit}/top.json"
        params = {"t": time_filter, "limit": limit}
        count = 0

        try:
            async with SmartClient(base_delay=1.0) as client:
                r = await client.get(url, params=params, timeout=30.0)

                if r.status_code == 429:
                    logger.info(f"[reddit] r/{subreddit}: rate limited, skipping")
                    return 0

                if r.status_code != 200:
                    logger.info(f"[reddit] r/{subreddit}: HTTP {r.status_code}")
                    return 0

                data = r.json()
                posts = data.get("data", {}).get("children", [])

                for post_wrapper in posts:
                    post = post_wrapper.get("data", {})
                    if not post:
                        continue

                    title = post.get("title", "")
                    body = post.get("selftext", "")
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


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def search_subreddit_for_ticker(
    subreddit: str,
    ticker: str,
    queries: list[str],
    time_filter: str = "week",
    limit: int = 10,
    since: datetime.datetime | None = None,
) -> int:
    """Search WITHIN a specific subreddit for a ticker.

    Scoped search prevents noise from non-financial subs.
    Uses multiple query variants to catch different post styles.
    Returns number of posts written.
    """
    with get_db() as db:
        count = 0
        seen_ids = set()

        try:
            async with SmartClient(base_delay=1.0) as client:
                for query in queries:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        "q": query,
                        "restrict_sr": "on",  # KEY: restrict to this subreddit only
                        "sort": "relevance",
                        "t": time_filter,
                        "limit": limit,
                        "type": "link",
                    }

                    r = await client.get(url, params=params, timeout=30.0)
                    if r.status_code != 200:
                        continue

                    data = r.json()
                    posts = data.get("data", {}).get("children", [])

                    for post_wrapper in posts:
                        post = post_wrapper.get("data", {})
                        if not post:
                            continue

                        post_id = post.get("id", "")
                        if post_id in seen_ids:
                            continue
                        seen_ids.add(post_id)

                        created_utc = datetime.datetime.fromtimestamp(
                            post.get("created_utc", 0), tz=datetime.UTC
                        )
                        if since and created_utc <= since:
                            continue

                        if not _is_quality_post(post):
                            continue

                        if not _is_relevant_to_ticker(post, ticker):
                            continue

                        title = post.get("title", "")
                        body = post.get("selftext", "")
                        full_text = f"{title} {body}"
                        tickers_found = set(get_ticker_symbols(full_text, title=title))
                        tickers_found.add(ticker.upper())

                        actual_sub = post.get("subreddit", subreddit)
                        count += _store_post(
                            db, post, ticker.upper(), actual_sub, tickers_found
                        )

                    await asyncio.sleep(1.0)  # Pace between queries

        except Exception as e:
            logger.info(f"[reddit] r/{subreddit} search error: {e}")

        return count


async def collect_for_ticker(ticker: str, since: datetime.datetime | None = None) -> int:
    """Collect Reddit posts about a specific ticker using subreddit-scoped search.

    Strategy:
    1. Search within each priority financial subreddit (not global)
    2. Use multiple query variants to catch different post styles
    3. Apply relevance filter to reject noise

    This prevents r/science, r/relationship_advice etc from polluting results.
    """
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
    body = post.get("selftext", "")
    post_id = post.get("id", hashlib.md5(title.encode()).hexdigest()[:12])

    created_utc = datetime.datetime.fromtimestamp(
        post.get("created_utc", 0), tz=datetime.UTC
    )

    score = post.get("score", 0)
    upvote_ratio = post.get("upvote_ratio", 0.0)
    num_comments = post.get("num_comments", 0)
    flair = post.get("link_flair_text", "")
    awards = post.get("total_awards_received", 0)

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
                body,  # Full body -- no truncation for debugging
                score,
                upvote_ratio,
                num_comments,
                flair,
                None,  # sentiment_score -- computed by processor later
                awards,
                round(comment_velocity, 2),
                created_utc,
            ],
        )

        # Write discovered tickers (filter through shared FALSE_TICKERS)
        for t in tickers_found:
            if t in FALSE_TICKERS or len(t) < 2:
                continue
            # Normalize score: use upvote_ratio as confidence (0-1)
            # rather than raw upvote count
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
        # Pace between subreddits to avoid rate limiting
        await asyncio.sleep(2.0)
    logger.info(f"[reddit] Total: {total} posts across {len(SUBREDDITS)} subreddits")
    return total

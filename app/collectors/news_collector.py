"""
News Collector -- Fetches financial news from RSS feeds + web sources.

Pure data collector. No LLM calls. No processing.
Writes to: news_articles
Libraries: feedparser, httpx (via SmartClient)

No API key needed -- uses public RSS feeds.
Dedup: hash(title + published_at) as id.

Search strategy:
  - General sweep: Collect ALL articles from RSS feeds
  - Ticker tagging: Post-process articles to detect ticker mentions in title+summary
  - This replaces the old pre-filter approach (which returned 0 results because
    RSS summaries rarely mention exact ticker symbols like "NVDA")

Anti-rate-limiting: exponential backoff, random jitter, pacing.
"""

import logging

logger = logging.getLogger(__name__)


import hashlib
import re
import datetime
import asyncio
import time
import feedparser
import cloudscraper
from app.db.connection import get_db
from app.services.request_utils import SmartClient
from app.processors.ticker_extractor import get_ticker_symbols

# cloudscraper instance for Cloudflare-protected sites
_cloudscraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "desktop": True}
)

# Domains known to block httpx with Cloudflare — use cloudscraper for these
CLOUDFLARE_DOMAINS = {
    "seekingalpha.com",
    "investing.com",
}

# RSS feeds to monitor (zero rate limits, no API key needed)
RSS_FEEDS = {
    # ── Market News (tier 1 — highest volume) ──
    "MarketWatch Top": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Markets": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "CNBC Top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "CNBC Markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Google News Business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
    # ── Analysis / Research ──
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
    "Benzinga": "https://www.benzinga.com/feed",
    "Business Insider": "https://www.businessinsider.com/rss",
    "Kiplinger": "https://www.kiplinger.com/feed/all",
    "Investing.com": "https://www.investing.com/rss/news.rss",
    "Nasdaq News": "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    # ── Wire services / broadsheet ──
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "NPR Business": "https://feeds.npr.org/1006/rss.xml",
    "The Guardian Business": "https://www.theguardian.com/uk/business/rss",
    "FT Markets": "https://www.ft.com/rss/home/uk",
    # ── Government / macro ──
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "US Treasury": "https://home.treasury.gov/rss.xml",
    # ── Crypto ──
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
}


def _extract_text_from_html(html: str, max_chars: int = 15000) -> str:
    """Extract readable text from HTML using Trafilatura."""
    import trafilatura
    
    # Extract main article text, dropping menus, footers, etc.
    text = trafilatura.extract(
        html, 
        include_links=False, 
        include_images=False, 
        include_tables=False,
        no_fallback=False
    )
    
    if not text:
        return ""
        
    # Strip remaining HTML tags just in case
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Phase 2: Dynamic Noise Filtering
    noise_patterns = [
        r"Copy LinkSavePlay\(\d+min\)Comments",
        r"via Getty Images",
        r"Sign up",
        r"Subscribe",
        r"Cookie",
        r"consent",
        r"Advertisement",
        r"Accessibility Menu",
        r"\[Accessibility Menu\]",
        r"Skip to main content",
        r"Share",
        r"Print",
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
    text = re.sub(r"\s+", " ", text).strip()

    # Phase 3: Pre-Janitor Guardrails (Quality Gate)
    failure_signatures = [
        "please enable javascript",
        "please enable js",
        "please enable cookies",
        "subscribe to continue reading",
        "verify you are human",
        "pardon our interruption",
        "are you a robot",
        "to access this content",
    ]
    
    text_lower = text.lower()
    for sig in failure_signatures:
        if sig in text_lower:
            return "" # Drop immediately, it's a paywall/bot-wall

    return text[:max_chars].strip() if text else ""


def _is_cloudflare_domain(url: str) -> bool:
    """Check if URL belongs to a Cloudflare-protected domain."""
    from urllib.parse import urlparse

    try:
        domain = urlparse(url).netloc.lower()
        return any(cf in domain for cf in CLOUDFLARE_DOMAINS)
    except Exception:
        return False


async def _scrape_article_body(
    url: str, client: SmartClient, max_chars: int = 15000
) -> str:
    """Fetch article URL and extract visible text as summary.

    Uses httpx only — this is called for EVERY RSS article (~100+) so it
    must be fast. No browser overhead.
    """
    if _is_cloudflare_domain(url):
        return await asyncio.to_thread(_scrape_with_cloudscraper, url)

    try:
        r = await client.get(url)
        if r.status_code == 200:
            text = _extract_text_from_html(r.text, max_chars)
            if text and len(text) > 50:
                return text
    except Exception:
        pass

    return ""


def _scrape_with_cloudscraper(url: str) -> str:
    """Sync cloudscraper call for Cloudflare-protected sites."""
    try:
        time.sleep(2.0)  # Rate limit: don't hammer protected sites
        r = _cloudscraper.get(url, timeout=15)
        if r.status_code == 200:
            return r.text
        logger.info(f"[smart_client] {url}: cloudscraper HTTP {r.status_code}")
        return ""
    except Exception as e:
        logger.info(f"[smart_client] {url}: cloudscraper error {e}")
        return ""


# Company name -> ticker mapping for detecting tickers from company names
COMPANY_TICKERS = {
    # Tech mega-caps
    "nvidia": "NVDA",
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "google": "GOOG",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "amd": "AMD",
    "palantir": "PLTR",
    "sofi": "SOFI",
    "super micro": "SMCI",
    "broadcom": "AVGO",
    "intel": "INTC",
    # Media/Retail
    "netflix": "NFLX",
    "disney": "DIS",
    "costco": "COST",
    "walmart": "WMT",
    "target": "TGT",
    # Finance
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "chase": "JPM",
    "goldman sachs": "GS",
    "wells fargo": "WFC",
    "bank of america": "BAC",
    "morgan stanley": "MS",
    "citigroup": "C",
    "citi": "C",
    # Industrial / Energy / Defense
    "boeing": "BA",
    "lockheed": "LMT",
    "raytheon": "RTX",
    "exxon": "XOM",
    "exxon mobil": "XOM",
    "exxonmobil": "XOM",
    "chevron": "CVX",
    "conocophillips": "COP",
    "3m": "MMM",
    "honeywell": "HON",
    "caterpillar": "CAT",
    "general electric": "GE",
    "ge aerospace": "GE",
    # Tech/Software
    "coinbase": "COIN",
    "robinhood": "HOOD",
    "uber": "UBER",
    "airbnb": "ABNB",
    "snowflake": "SNOW",
    "crowdstrike": "CRWD",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "oracle": "ORCL",
    "servicenow": "NOW",
    "palo alto": "PANW",
    # Crypto
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    # Indices
    "s&p 500": "SPY",
    "s&p": "SPY",
    "dow jones": "DIA",
    "nasdaq": "QQQ",
    # Healthcare
    "unitedhealth": "UNH",
    "johnson & johnson": "JNJ",
    "j&j": "JNJ",
    "pfizer": "PFE",
    "eli lilly": "LLY",
    "abbvie": "ABBV",
    # Semiconductor
    "arm holdings": "ARM",
    "marvell": "MRVL",
    "micron": "MU",
    "qualcomm": "QCOM",
    "texas instruments": "TXN",
    "taiwan semi": "TSM",
    "tsmc": "TSM",
}

# Direct ticker symbols to match
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
    "MS",
    "C",
    "NFLX",
    "DIS",
    "COST",
    "WMT",
    "TGT",
    "BA",
    "LMT",
    "RTX",
    "COIN",
    "HOOD",
    # Added from battle test gaps
    "XOM",
    "CVX",
    "COP",
    "MMM",
    "HON",
    "CAT",
    "GE",
    "CRM",
    "ADBE",
    "ORCL",
    "NOW",
    "PANW",
    "UNH",
    "JNJ",
    "PFE",
    "LLY",
    "ABBV",
    "MU",
    "QCOM",
    "TXN",
    "TSM",
}

TICKER_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")


def _detect_tickers_in_text(text: str) -> set[str]:
    """Detect stock tickers mentioned in article text.

    UPGRADED: Now uses the shared ticker_extractor module with
    CompanyRegistry (S&P 500 + aliases) and confidence scoring.
    Returns set of detected ticker symbols (>= 0.60 confidence).
    """
    return set(get_ticker_symbols(text))


def _normalize_title(title: str) -> str:
    """Normalize title for cross-source deduplication.

    Strips noise prefixes, punctuation, and whitespace so the same
    article from Yahoo + Finnhub + RSS gets the same normalized key.
    """
    t = title.lower().strip()
    t = re.sub(r"^(breaking|update|exclusive|report|analysis|opinion)[:\s-]+", "", t)
    t = re.sub(r"[^\w\s]", "", t)  # strip punctuation
    t = re.sub(r"\s+", " ", t)  # collapse whitespace
    return t.strip()[:200]  # cap length for DB


def _get_article_id(title: str, ticker: str | None) -> str:
    """Generate a deterministic SHA256 hash ID for cross-source deduplication.

    Replaces the expensive LIKE query by ensuring identical headlines
    from different sources collide on the primary key (ON CONFLICT DO NOTHING).
    """
    norm = _normalize_title(title)
    return hashlib.sha256(f"{norm}_{ticker or 'NONE'}".encode()).hexdigest()


async def collect_feed(feed_name: str, feed_url: str) -> int:
    """
    Fetch and parse a single RSS feed, write articles to news_articles.
    Tags each article with detected tickers from title + summary.
    Returns number of new articles written.
    """
    count = 0
    scrape_count = 0

    try:
        with get_db() as db:
            async with SmartClient(base_delay=1.0, max_retries=3) as client:
                r = await client.get(feed_url)
                if r.status_code != 200:
                    logger.warning(f"[news] {feed_name}: HTTP {r.status_code}")
                    return 0

                feed = feedparser.parse(r.text)

                if not feed.entries:
                    logger.warning(f"[news] {feed_name}: 0 entries parsed from feed")
                    return 0

                for entry in feed.entries:
                    title = entry.get("title", "").strip()
                    if not title:
                        continue

                    # Parse published date
                    published_at = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        p = entry.published_parsed
                        published_at = datetime.datetime(
                            p.tm_year,
                            p.tm_mon,
                            p.tm_mday,
                            p.tm_hour,
                            p.tm_min,
                            p.tm_sec,
                            tzinfo=datetime.UTC,
                        )
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        u = entry.updated_parsed
                        published_at = datetime.datetime(
                            u.tm_year,
                            u.tm_mon,
                            u.tm_mday,
                            u.tm_hour,
                            u.tm_min,
                            u.tm_sec,
                            tzinfo=datetime.UTC,
                        )
                    else:
                        published_at = datetime.datetime.now(datetime.UTC)

                    url = entry.get("link", "")
                    summary = entry.get("summary", "").strip()
                    publisher = feed_name

                    # If RSS has no summary or it's short/cut-off, scrape the article body
                    if url and (not summary or "..." in summary or len(summary) < 150):
                        body = await _scrape_article_body(url, client)
                        if body:
                            summary = body
                            scrape_count += 1
                            
                    # STRICT QUALITY GATE: If we still don't have a real article, drop it
                    if len(summary) < 150:
                        continue

                    # Detect tickers in title + summary
                    full_text = f"{title} {summary}"
                    detected_tickers = _detect_tickers_in_text(full_text)

                    # Build unique ID from title + date
                    id_str = (
                        f"{title}{published_at.isoformat() if published_at else ''}"
                    )
                    article_id = hashlib.md5(id_str.encode()).hexdigest()

                    if detected_tickers:
                        # Store one row per detected ticker for easy querying
                        for ticker in detected_tickers:
                            ticker_article_id = _get_article_id(title, ticker)
                            db.execute(
                                """
                                INSERT INTO news_articles
                                (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, 'rss', CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO NOTHING
                            """,
                                [
                                    ticker_article_id,
                                    ticker,
                                    title[:500],
                                    publisher,
                                    url,
                                    published_at,
                                    summary,
                                ],
                            )
                            count += 1
                    else:
                        # No specific ticker detected -- store as general market news
                        article_id = _get_article_id(title, None)
                        db.execute(
                            """
                            INSERT INTO news_articles
                            (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, 'rss', CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO NOTHING
                        """,
                            [
                                article_id,
                                None,
                                title[:500],
                                publisher,
                                url,
                                published_at,
                                summary,
                            ],
                        )
                        count += 1

            if scrape_count > 0:
                logger.info(
                    f"[news] {feed_name}: scraped {scrape_count} article bodies"
                )

    except Exception as e:
        logger.error(
            f"[news] {feed_name} FAILED: {type(e).__name__}: {e}",
            exc_info=True,
        )

    return count


async def collect_all(limit_feeds: int | None = None) -> int:
    """
    Fetch all RSS feeds. Returns total articles written.
    All articles are auto-tagged with detected tickers.
    Paces requests between feeds to avoid rate limiting.
    """
    total = 0
    failed = 0
    feeds_to_check = list(RSS_FEEDS.items())
    if limit_feeds and limit_feeds > 0 and limit_feeds < len(feeds_to_check):
        feeds_to_check = feeds_to_check[:limit_feeds]

    for name, url in feeds_to_check:
        try:
            count = await collect_feed(name, url)
            if count > 0:
                logger.info(f"[news] {name}: {count} articles")
            total += count
        except Exception as e:
            failed += 1
            logger.error(
                f"[news] {name}: UNCAUGHT: {type(e).__name__}: {e}",
                exc_info=True,
            )
        await asyncio.sleep(2.0)

    logger.info(
        f"[news] Total: {total} articles from {len(feeds_to_check)} feeds"
        + (f" ({failed} failed)" if failed else "")
    )
    return total


async def collect_for_ticker(ticker: str, since: datetime.datetime | None = None) -> int:
    """Collect news articles mentioning a specific ticker.

    Strategy (ticker-specific sources only):
    1. Finnhub API — per-ticker, highest volume (~200 articles/ticker/week)
    2. yfinance — per-ticker headlines

    NOTE: RSS sweep (collect_all) is NOT called here — it scrapes 9 general
    feeds (CNBC, Bloomberg, etc.) which waste time for individual analysis.
    RSS sweep only runs during the full trading cycle.

    Rate limiting: 3s between API layers to avoid hammering.
    """
    total = 0

    # Layer 1: Finnhub (highest volume, most reliable)
    fh_count = await collect_finnhub_news(ticker, since=since)
    total += fh_count
    await asyncio.sleep(3)  # Breathe between API providers

    # Layer 2: yfinance headlines
    yf_count = await collect_yfinance_news(ticker, since=since)
    total += yf_count

    logger.info(
        f"[news] {ticker}: {total} total articles (finnhub={fh_count}, yfinance={yf_count})"
    )
    return total


async def collect_finnhub_news(
    ticker: str, days: int = 7, max_articles: int = 50, since: datetime.datetime | None = None
) -> int:
    """Fetch per-ticker news from Finnhub API.

    Finnhub returns 100-250 articles/ticker/week. We:
    1. Sort newest first
    2. Deduplicate via headline Jaccard similarity (>60% word overlap = skip)
    3. Cap at `max_articles` unique articles

    Finnhub provides summaries — no need to deep-read most articles.
    Use `deep_read_top_articles()` to scrape full text for the top N.

    Free tier: 60 calls/min. Rate limit: 1s delay after API call.
    """
    import os

    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        logger.info("[news] FINNHUB_API_KEY not set, skipping Finnhub")
        return 0

    try:
        import finnhub
    except ImportError:
        logger.info("[news] finnhub-python not installed, skipping")
        return 0

    try:
        client = finnhub.Client(api_key=api_key)
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        news = await asyncio.to_thread(
            client.company_news, ticker, _from=start_str, to=end_str
        )

        if not news:
            return 0

        # Sort newest first
        news.sort(key=lambda a: a.get("datetime", 0), reverse=True)

        # Sniper Strategy: Drop notoriously bad publishers (Phase 3)
        with get_db() as db:
            trusted = db.execute("SELECT source_name, win_rate, total_items FROM source_trust WHERE source_type='publisher'").fetchall()
        bad_publishers = {row[0] for row in trusted if row[2] >= 5 and row[1] < 0.1}

        # Deduplicate: skip articles with >60% headline word overlap
        seen_word_sets: list[set] = []
        unique_articles = []
        skipped = 0

        for article in news:
            source = article.get("source", "")
            if source in bad_publishers:
                skipped += 1
                continue

            headline = article.get("headline", "").strip()
            if not headline:
                continue

            # Normalize headline to word set
            words = set(headline.lower().split())
            words -= {
                "the",
                "a",
                "an",
                "and",
                "or",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "is",
                "are",
                "was",
            }

            # Check Jaccard similarity against seen headlines
            is_duplicate = False
            for seen in seen_word_sets:
                if not words or not seen:
                    continue
                intersection = words & seen
                union = words | seen
                similarity = len(intersection) / len(union)
                if similarity > 0.6:
                    is_duplicate = True
                    skipped += 1
                    break

            if not is_duplicate:
                seen_word_sets.append(words)
                unique_articles.append(article)
                if len(unique_articles) >= max_articles:
                    break

        # Store unique articles
        with get_db() as db:
            count = 0
            dedup_count = 0
            async with SmartClient() as client:
                for article in unique_articles:
                    headline = article.get("headline", "").strip()
                    summary = article.get("summary", "").strip()
                    url = article.get("url", "")
                    source = article.get("source", "finnhub")
                    ts = article.get("datetime", 0)
                    
                    # Finnhub summaries are often cut off. Scrape the real text!
                    if url and (len(summary) < 150 or "..." in summary):
                        body = await _scrape_article_body(url, client)
                        if body:
                            summary = body
                            
                    # STRICT QUALITY GATE: If we don't have substantial text, drop it entirely.
                    if len(summary) < 150:
                        continue
                        
                    published_at = (
                        datetime.datetime.fromtimestamp(ts, tz=datetime.UTC) if ts else None
                    )

                    if since and published_at and published_at <= since:
                        continue

                    article_id = _get_article_id(headline, ticker.upper())

                    db.execute(
                        """
                        INSERT INTO news_articles
                        (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'finnhub', CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        [
                            article_id,
                            ticker.upper(),
                            headline[:500],
                            source,
                            url,
                            published_at,
                            summary,
                        ],
                    )
                    count += 1

            dedup_msg = f", {dedup_count} cross-source dupes" if dedup_count else ""
            logger.info(
                f"[news] Finnhub {ticker}: {count} unique articles (skipped {skipped} duplicates{dedup_msg})"
            )
            await asyncio.sleep(1)  # Rate limit: stay within 60 calls/min
            return count

    except Exception as e:
        logger.info(f"[news] Finnhub {ticker} error: {e}")
        return 0


async def collect_yfinance_news(ticker: str, since: datetime.datetime | None = None) -> int:
    """Fetch per-ticker news from yfinance.

    Returns ~8-15 recent headlines per ticker.
    Free, no API key. Uses .news property (not method).
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.info("[news] yfinance not installed, skipping")
        return 0

    try:
        t = yf.Ticker(ticker)
        # .news is a property, not a method — returns list of dicts
        news = await asyncio.to_thread(lambda: t.news)

        if not news:
            return 0

        with get_db() as db:
            count = 0
            dedup_count = 0
            async with SmartClient() as client:
                for article in news:
                    # yfinance v2 format: nested under 'content'
                    content = article.get("content", article)
                    title = content.get("title", "").strip()
                    if not title:
                        continue

                    # Extract URL from canonical or clickThroughUrl
                    url = ""
                    if "canonicalUrl" in content:
                        url_obj = content["canonicalUrl"]
                        url = (
                            url_obj.get("url", "")
                            if isinstance(url_obj, dict)
                            else str(url_obj)
                        )
                    elif "clickThroughUrl" in content:
                        url_obj = content["clickThroughUrl"]
                        url = (
                            url_obj.get("url", "")
                            if isinstance(url_obj, dict)
                            else str(url_obj)
                        )
                    elif "link" in article:
                        url = article["link"]

                    # Publisher
                    provider = content.get("provider", {})
                    publisher = (
                        provider.get("displayName", "yfinance")
                        if isinstance(provider, dict)
                        else "yfinance"
                    )

                    # Published date
                    pub_date = content.get("pubDate", "")
                    published_at = None
                    if pub_date:
                        try:
                            published_at = datetime.datetime.fromisoformat(
                                pub_date.replace("Z", "+00:00")
                            )
                        except Exception:
                            pass
                        
                    # yfinance returns NO summary natively in this call. Scrape it!
                    summary = ""
                    if url:
                        summary = await _scrape_article_body(url, client)
                        
                    # STRICT QUALITY GATE: If we don't have substantial text, drop it entirely.
                    if len(summary) < 150:
                        continue
                    
                    if since and published_at and published_at <= since:
                        continue

                    article_id = _get_article_id(title, ticker.upper())

                    db.execute(
                        """
                        INSERT INTO news_articles
                        (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'yfinance', CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        [
                            article_id,
                            ticker.upper(),
                            title[:500],
                            publisher,
                            url,
                            published_at,
                            summary,
                        ],
                    )
                    count += 1

            dedup_msg = (
                f" (skipped {dedup_count} cross-source dupes)" if dedup_count else ""
            )
            logger.info(f"[news] yfinance {ticker}: {count} articles{dedup_msg}")
            return count

    except Exception as e:
        logger.info(f"[news] yfinance {ticker} error: {e}")
        return 0


_GARBAGE_STRINGS = [
    "Accessibility Menu",
    "Skip to main content",
    "Skip to Content",
    "Sign in / Join",
    "Premium Investing Services",
    "Stock Advisor",
    "Rule Breakers",
    "Join Stock Advisor",
    "Subscribe Now",
    "Motley Fool",
    "Accept All Cookies",
    "Cookie Settings",
    "Privacy Policy",
    "We and our partners",
    "consent to the use",
    "strictly necessary",
    "Toggle navigation",
    "Open Navigation",
    "Close Navigation",
    "Full Screen Menu",
    "Site Navigation",
    "Main Navigation",
]


def _clean_deep_read(text: str) -> str | None:
    """Strip known garbage strings from deep-read content.

    Returns cleaned text, or None if the article is mostly garbage.
    """
    if not text:
        return None

    original_len = len(text)
    cleaned = text

    # Strip known garbage strings
    for g in _GARBAGE_STRINGS:
        cleaned = cleaned.replace(g, "")

    # Strip lines that look like nav/footer (very short lines at start/end)
    lines = cleaned.split("\n")
    start_cut = 0
    for line in lines:
        stripped = line.strip()
        # Skip short lines at the beginning (likely nav/header)
        if len(stripped) < 20 and start_cut < 10:
            start_cut += 1
        else:
            break
    lines = lines[start_cut:]
    cleaned = "\n".join(lines).strip()

    # If >30% of original content was garbage, reject it
    if original_len > 0 and len(cleaned) < original_len * 0.5:
        logger.info(
            f"[news] deep-read: rejected — {original_len - len(cleaned)} chars were garbage ({100 - len(cleaned) * 100 // original_len}%)"
        )
        return None

    # Final check: must have meaningful content
    if len(cleaned) < 100:
        return None

    return cleaned


async def deep_read_article(url: str, max_chars: int = 15000) -> str | None:
    """Deep-read a news article URL for full article body.

    Fallback chain:
      0. Adaptive Scraper (vision LLM generated JS scraper via crawl4ai)
      1. crawl4ai (stealth scrape → markdown)
      2. vision_scraper (Playwright overlay removal → screenshot → Qwen OCR)

    All results are cleaned through _clean_deep_read() to filter garbage.
    Returns full article text or None.
    """
    # Method 0: Adaptive Scraper
    try:
        from app.collectors.adaptive_scraper import run_adaptive
        
        adaptive_text = await run_adaptive(url)
        if adaptive_text and len(adaptive_text) > 100:
            cleaned = _clean_deep_read(adaptive_text[:max_chars])
            if cleaned:
                logger.info(
                    f"[news] adaptive-read: {len(cleaned)} chars from {url[:50]}"
                )
                return cleaned
        logger.info("[news] deep-read: adaptive scraper failed or returned placeholder, trying crawl4ai...")
    except ImportError:
        logger.info("[news] adaptive_scraper not installed")
    except Exception as e:
        logger.info(f"[news] adaptive-read error for {url[:50]}: {e}")

    # Method 1: crawl4ai
    try:
        from app.collectors.crawl4ai_config import scrape_url

        result = await scrape_url(url, max_chars=max_chars, rate_limit_delay=3.0)
        if result["success"] and result["text"]:
            text = result["text"]
            if len(text) > 100 and "oops" not in text.lower()[:50]:
                cleaned = _clean_deep_read(text)
                if cleaned:
                    logger.info(
                        f"[news] deep-read: {len(cleaned)} chars from {url[:50]}"
                    )
                    return cleaned
            logger.info("[news] deep-read: crawl4ai got placeholder, trying vision...")
    except ImportError:
        logger.info("[news] crawl4ai not installed for deep-read")
    except Exception as e:
        logger.info(f"[news] deep-read crawl4ai error for {url[:50]}: {e}")

    # Method 2: Vision pipeline (screenshot → Qwen OCR)
    try:
        from app.collectors.vision_scraper import vision_deep_read

        text = await vision_deep_read(url)
        if text and len(text) > 100:
            cleaned = _clean_deep_read(text[:max_chars])
            if cleaned:
                logger.info(
                    f"[news] vision deep-read: {len(cleaned)} chars from {url[:50]}"
                )
                return cleaned
    except ImportError:
        pass  # vision_scraper or playwright not available
    except Exception as e:
        logger.info(f"[news] vision deep-read error for {url[:50]}: {e}")

    return None


async def deep_read_top_articles(
    ticker: str, limit: int = 3, max_chars: int = 15000
) -> list[dict]:
    """Deep-read the top N most recent articles for a ticker.

    Fetches full article body via crawl4ai (primary) or Playwright (fallback)
    for the most recent articles that don't already have substantial summaries.

    Rate limit: 5s between each article deep-read to avoid bans.

    Returns list of {title, url, full_text} dicts.
    Use this to feed detailed article content to the LLM for deep analysis.
    """
    with get_db() as db:
        articles = db.execute(
            """
            SELECT id, title, url, summary FROM news_articles
            WHERE ticker = %s AND url != '' AND url IS NOT NULL
            ORDER BY published_at DESC
            LIMIT %s
        """,
            [ticker.upper(), limit * 2],
        ).fetchall()  # Fetch extra in case some fail

        results = []
        for row in articles:
            if len(results) >= limit:
                break

            article_id, title, url, summary = row

            # Skip if we already have a good summary (>200 chars)
            if summary and len(summary) > 200:
                results.append({"title": title, "url": url, "full_text": summary})
                continue

            # Deep-read
            full_text = await deep_read_article(url, max_chars)
            if full_text:
                # Update the DB with the full text as summary
                db.execute(
                    "UPDATE news_articles SET summary = %s WHERE id = %s",
                    [full_text, article_id],
                )
                results.append({"title": title, "url": url, "full_text": full_text})

            await asyncio.sleep(5)  # Rate limit: go slow to avoid bans

        logger.info(
            f"[news] Deep-read {ticker}: {len(results)} articles with full text"
        )
        return results

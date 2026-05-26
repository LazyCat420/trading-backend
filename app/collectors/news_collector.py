"""
News Collector -- Fetches financial news from RSS feeds + web sources.

Pure data collector. No LLM calls. No processing.
Writes to: news_articles
No API key needed -- uses scraper-service.
Dedup: hash(title + published_at) as id.
"""

import logging
import hashlib
import re
import datetime
import asyncio
import time
from app.db.connection import get_db
from app.processors.ticker_extractor import get_ticker_symbols

logger = logging.getLogger(__name__)

# RSS feeds to monitor
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

# Company name -> ticker mapping
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


async def _scrape_article_body_via_service(url: str, max_chars: int = 15000) -> str:
    """Scrape article body using the http engine on scraper-service."""
    from app.services.scraper_client import scraper_client

    res = await scraper_client.scrape(url, engine="http", options={"max_chars": max_chars})
    if res and res.get("success") and res.get("content"):
        return res["content"]
    return ""


def _extract_text_from_html(html: str, max_chars: int = 15000) -> str:
    """Extract readable text from HTML using BeautifulSoup with a regex fallback."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)
        return text[:max_chars]
    except Exception:
        # Simple regex fallback
        text = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]


def _detect_tickers_in_text(text: str) -> set[str]:
    """Detect stock tickers mentioned in article text."""
    return set(get_ticker_symbols(text))


def _is_article_relevant_to_ticker(ticker: str, text: str) -> bool:
    """Check if an article actually discusses a ticker as a financial instrument.

    For short tickers (2-3 chars) that are also common English words (TV, HD, PC, etc.),
    we require additional evidence that the article is actually about the STOCK,
    not just using the letters as an abbreviation.

    Returns True if the article passes the relevance check.
    """
    # Long tickers (4+ chars) and $TICKER syntax are inherently less ambiguous
    if len(ticker) >= 4:
        return True

    # If the article text contains the $TICKER pattern, it's explicitly financial
    if re.search(rf"\${re.escape(ticker)}\b", text):
        return True

    # Check if the company name is mentioned (not just the ticker letters)
    from app.processors.ticker_extractor import get_registry
    registry = get_registry()
    company = registry.lookup_symbol(ticker)
    if company:
        # Check for company name (e.g., "Grupo Televisa" for TV, "Home Depot" for HD)
        name_lower = company.name.lower()
        text_lower = text.lower()
        if name_lower in text_lower:
            return True
        # Check aliases
        for alias in company.aliases:
            if len(alias) > 3 and alias.lower() in text_lower:
                return True

    # For 2-3 letter tickers without company name: require strong financial context
    # near the ticker mention (at least 2 financial keywords within 150 chars)
    financial_kw = {
        "stock", "shares", "price", "earnings", "revenue", "profit",
        "bullish", "bearish", "analyst", "upgrade", "downgrade", "rating",
        "dividend", "ipo", "merger", "acquisition", "guidance", "forecast",
        "quarterly", "eps", "valuation", "rally", "surge", "plunge",
        "overweight", "underweight", "outperform", "underperform",
        "market cap", "pe ratio", "share price", "ticker",
    }
    for m in re.finditer(rf"\b{re.escape(ticker)}\b", text):
        start_idx = max(0, m.start() - 150)
        end_idx = min(len(text), m.end() + 150)
        window = text[start_idx:end_idx].lower()
        hits = sum(1 for kw in financial_kw if kw in window)
        if hits >= 2:
            return True

    return False


def _normalize_title(title: str) -> str:
    """Normalize title for cross-source deduplication."""
    t = title.lower().strip()
    t = re.sub(r"^(breaking|update|exclusive|report|analysis|opinion)[:\s-]+", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()[:200]


def _get_article_id(title: str, ticker: str | None) -> str:
    """Generate a deterministic SHA256 hash ID for cross-source deduplication."""
    norm = _normalize_title(title)
    return hashlib.sha256(f"{norm}_{ticker or 'NONE'}".encode()).hexdigest()


async def collect_feed(feed_name: str, feed_url: str) -> int:
    """
    Fetch and parse a single RSS feed via scraper-service, write articles to news_articles.
    Returns number of new articles written.
    """
    from app.services.scraper_client import scraper_client

    count = 0
    try:
        with get_db() as db:
            items = await scraper_client.collect(
                source="news",
                req_data={
                    "feed_url": feed_url,
                    "query": feed_name,
                }
            )

            for article in items:
                title = article.get("title", "").strip()
                if not title:
                    continue

                url = article.get("url", "")
                summary = article.get("summary", "").strip()
                publisher = article.get("publisher", feed_name)

                pub_val = article.get("published_at")
                if isinstance(pub_val, str):
                    published_at = datetime.datetime.fromisoformat(pub_val)
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=datetime.UTC)
                else:
                    published_at = datetime.datetime.now(datetime.UTC)

                # STRICT QUALITY GATE & BODY SCRAPING
                api_summary = summary
                summary = ""
                if url and (len(api_summary) < 150 or "..." in api_summary):
                    body = await _scrape_article_body_via_service(url)
                    if body:
                        summary = body

                if (not summary or len(summary) < 150) and len(api_summary) >= 150:
                    summary = api_summary

                if len(summary) < 150:
                    continue

                # Detect tickers in title + summary
                full_text = f"{title} {summary}"
                detected_tickers = _detect_tickers_in_text(full_text)

                # Relevance gate: for short/ambiguous tickers, verify the article
                # actually discusses the stock (not just uses the letters as English).
                if detected_tickers:
                    relevant_tickers = {
                        t for t in detected_tickers
                        if _is_article_relevant_to_ticker(t, full_text)
                    }
                    irrelevant = detected_tickers - relevant_tickers
                    if irrelevant:
                        logger.debug(
                            "[news] Relevance gate filtered tickers %s from article: %s",
                            irrelevant, title[:80],
                        )
                    detected_tickers = relevant_tickers

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
    except Exception as e:
        logger.error(f"[news] {feed_name} FAILED: {type(e).__name__}: {e}", exc_info=True)

    return count


async def collect_all(limit_feeds: int | None = None) -> int:
    """Fetch all RSS feeds. Returns total articles written."""
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
    """Collect news articles mentioning a specific ticker."""
    total = 0

    # Layer 1: Finnhub (highest volume, most reliable)
    fh_count = await collect_finnhub_news(ticker, since=since)
    total += fh_count
    await asyncio.sleep(3)

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
    """Fetch per-ticker news from Finnhub API."""
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

        news.sort(key=lambda a: a.get("datetime", 0), reverse=True)

        with get_db() as db:
            trusted = db.execute("SELECT source_name, win_rate, total_items FROM source_trust WHERE source_type='publisher'").fetchall()
        bad_publishers = {row[0] for row in trusted if row[2] >= 5 and row[1] < 0.1}

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

            words = set(headline.lower().split())
            words -= {
                "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "is", "are", "was",
            }

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

        with get_db() as db:
            count = 0
            for article in unique_articles:
                headline = article.get("headline", "").strip()
                summary = article.get("summary", "").strip()
                url = article.get("url", "")
                source = article.get("source", "finnhub")
                ts = article.get("datetime", 0)

                if url and (len(summary) < 150 or "..." in summary):
                    body = await _scrape_article_body_via_service(url)
                    if body:
                        summary = body

                if len(summary) < 150:
                    continue

                published_at = (
                    datetime.datetime.fromtimestamp(ts, tz=datetime.UTC) if ts else None
                )

                if since and published_at and published_at <= since:
                    continue

                full_text = f"{headline} {summary}"
                detected_tickers = _detect_tickers_in_text(full_text)
                if detected_tickers:
                    detected_tickers = {
                        t for t in detected_tickers
                        if _is_article_relevant_to_ticker(t, full_text)
                    }
                tickers_to_insert = list(detected_tickers) if detected_tickers else [ticker.upper()]

                for t in tickers_to_insert:
                    article_id = _get_article_id(headline, t)

                    db.execute(
                        """
                        INSERT INTO news_articles
                        (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'finnhub', CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        [
                            article_id,
                            t,
                            headline[:500],
                            source,
                            url,
                            published_at,
                            summary,
                        ],
                    )
                    count += 1

            logger.info(
                f"[news] Finnhub {ticker}: {count} unique articles (skipped {skipped} duplicates)"
            )
            await asyncio.sleep(1)
            return count

    except Exception as e:
        logger.info(f"[news] Finnhub {ticker} error: {e}")
        return 0


async def collect_yfinance_news(ticker: str, since: datetime.datetime | None = None) -> int:
    """Fetch per-ticker news from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.info("[news] yfinance not installed, skipping")
        return 0

    try:
        t = yf.Ticker(ticker)
        news = await asyncio.to_thread(lambda: t.news)

        if not news:
            return 0

        with get_db() as db:
            trusted = db.execute("SELECT source_name, win_rate, total_items FROM source_trust WHERE source_type='publisher'").fetchall()
        bad_publishers = {row[0] for row in trusted if row[2] >= 5 and row[1] < 0.1}

        with get_db() as db:
            count = 0
            for article in news:
                content = article.get("content", article)
                title = content.get("title", "").strip()
                if not title:
                    continue

                url = ""
                if "canonicalUrl" in content:
                    url_obj = content["canonicalUrl"]
                    url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                elif "clickThroughUrl" in content:
                    url_obj = content["clickThroughUrl"]
                    url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                elif "link" in article:
                    url = article["link"]

                provider = content.get("provider", {})
                publisher = (
                    provider.get("displayName", "yfinance")
                    if isinstance(provider, dict)
                    else "yfinance"
                )

                if publisher in bad_publishers:
                    continue

                pub_date = content.get("pubDate", "")
                published_at = None
                if pub_date:
                    try:
                        published_at = datetime.datetime.fromisoformat(
                            pub_date.replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

                api_summary = content.get("description", "") or content.get("summary", "")
                summary = ""
                if url:
                    summary = await _scrape_article_body_via_service(url)

                # Fallback if scraping failed but the API summary is detailed enough
                if (not summary or len(summary) < 150) and len(api_summary) >= 150:
                    summary = api_summary

                if len(summary) < 150:
                    continue

                if since and published_at and published_at <= since:
                    continue

                full_text = f"{title} {summary}"
                detected_tickers = _detect_tickers_in_text(full_text)
                if detected_tickers:
                    detected_tickers = {
                        t for t in detected_tickers
                        if _is_article_relevant_to_ticker(t, full_text)
                    }
                tickers_to_insert = list(detected_tickers) if detected_tickers else [ticker.upper()]

                for t in tickers_to_insert:
                    article_id = _get_article_id(title, t)

                    db.execute(
                        """
                        INSERT INTO news_articles
                        (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'yfinance', CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        [
                            article_id,
                            t,
                            title[:500],
                            publisher,
                            url,
                            published_at,
                            summary,
                        ],
                    )
                    count += 1

            logger.info(f"[news] yfinance {ticker}: {count} articles")
            return count

    except Exception as e:
        logger.info(f"[news] yfinance {ticker} error: {e}")
        return 0


_GARBAGE_STRINGS = [
    "Accessibility Menu", "Skip to main content", "Skip to Content", "Sign in / Join",
    "Premium Investing Services", "Stock Advisor", "Rule Breakers", "Join Stock Advisor",
    "Subscribe Now", "Motley Fool", "Accept All Cookies", "Cookie Settings", "Privacy Policy",
    "We and our partners", "consent to the use", "strictly necessary", "Toggle navigation",
    "Open Navigation", "Close Navigation", "Full Screen Menu", "Site Navigation", "Main Navigation",
]


def _clean_deep_read(text: str) -> str | None:
    """Strip known garbage strings from deep-read content."""
    if not text:
        return None

    original_len = len(text)
    cleaned = text

    for g in _GARBAGE_STRINGS:
        cleaned = cleaned.replace(g, "")

    lines = cleaned.split("\n")
    start_cut = 0
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 20 and start_cut < 10:
            start_cut += 1
        else:
            break
    lines = lines[start_cut:]
    cleaned = "\n".join(lines).strip()

    if original_len > 0 and len(cleaned) < original_len * 0.5:
        return None

    if len(cleaned) < 100:
        return None

    return cleaned


async def deep_read_article(url: str, max_chars: int = 15000) -> str | None:
    """Deep-read a news article URL for full article body."""
    # Method 0: Adaptive Scraper
    try:
        from app.collectors.adaptive_scraper import run_adaptive

        adaptive_text = await run_adaptive(url)
        if adaptive_text and len(adaptive_text) > 100:
            cleaned = _clean_deep_read(adaptive_text[:max_chars])
            if cleaned:
                logger.info(f"[news] adaptive-read: {len(cleaned)} chars from {url[:50]}")
                return cleaned
        logger.info("[news] deep-read: adaptive scraper failed, trying crawl4ai...")
    except Exception as e:
        logger.info(f"[news] adaptive-read error for {url[:50]}: {e}")

    # Method 1: crawl4ai via scraper-service
    try:
        from app.services.scraper_client import scraper_client
        res = await scraper_client.scrape(url, engine="crawl4ai", options={"max_chars": max_chars})
        if res and res.get("success") and res.get("content"):
            text = res["content"]
            if len(text) > 100 and "oops" not in text.lower()[:50]:
                cleaned = _clean_deep_read(text)
                if cleaned:
                    logger.info(f"[news] deep-read (crawl4ai): {len(cleaned)} chars from {url[:50]}")
                    return cleaned
            logger.info("[news] deep-read: crawl4ai got placeholder, trying vision...")
    except Exception as e:
        logger.info(f"[news] deep-read crawl4ai error for {url[:50]}: {e}")

    # Method 2: Vision pipeline via scraper-service
    try:
        from app.services.scraper_client import scraper_client
        res = await scraper_client.scrape(url, engine="vision", options={"max_chars": max_chars})
        if res and res.get("success") and res.get("content"):
            text = res["content"]
            if text and len(text) > 100:
                cleaned = _clean_deep_read(text[:max_chars])
                if cleaned:
                    logger.info(f"[news] vision deep-read: {len(cleaned)} chars from {url[:50]}")
                    return cleaned
    except Exception as e:
        logger.info(f"[news] vision deep-read error for {url[:50]}: {e}")

    return None


async def deep_read_top_articles(
    ticker: str, limit: int = 3, max_chars: int = 15000
) -> list[dict]:
    """Deep-read the top N most recent articles for a ticker."""
    with get_db() as db:
        articles = db.execute(
            """
            SELECT id, title, url, summary FROM news_articles
            WHERE ticker = %s AND url != '' AND url IS NOT NULL
            ORDER BY published_at DESC
            LIMIT %s
        """,
            [ticker.upper(), limit * 2],
        ).fetchall()

        results = []
        for row in articles:
            if len(results) >= limit:
                break

            article_id, title, url, summary = row

            if summary and len(summary) > 200:
                results.append({"title": title, "url": url, "full_text": summary})
                continue

            full_text = await deep_read_article(url, max_chars)
            if full_text:
                db.execute(
                    "UPDATE news_articles SET summary = %s WHERE id = %s",
                    [full_text, article_id],
                )
                results.append({"title": title, "url": url, "full_text": full_text})

            await asyncio.sleep(5)

        logger.info(f"[news] Deep-read {ticker}: {len(results)} articles with full text")
        return results

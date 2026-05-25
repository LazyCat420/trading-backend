"""
News API Rotator — Cycles across free news API providers + RSS feeds.

Rotates through all configured providers so no single source's rate limit
blocks the pipeline. Providers without an API key are automatically skipped.

Integrates with the existing collectors pattern:
  - Uses SmartClient (app/services/request_utils.py) for HTTP with backoff
  - Uses %s placeholders for DB queries (psycopg compat shim)
  - Tags all articles with tickers via _detect_tickers_in_text()
  - Deduplicates via ON CONFLICT (id) DO NOTHING
  - Delegates to real finnhub_collector.collect_news() (not a fake class)

Install deps (already in requirements.txt):
    feedparser>=6.0.12
    httpx>=0.28.0
    markdownify>=0.13.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


from app.config import settings
from app.db.connection import get_db
from app.services.request_utils import SmartClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class NewsArticle:
    """In-memory article representation before DB write."""

    title: str
    url: str
    summary: str
    source: str
    published_at: datetime
    tickers: list[str] = field(default_factory=list)
    full_text_md: str = ""
    sentiment: float | None = None  # -1.0 to 1.0 if provider supplies it


# ---------------------------------------------------------------------------
# Rate-limit tracker  (in-memory, resets on restart — fine for testing)
# ---------------------------------------------------------------------------


class QuotaTracker:
    """Simple per-provider quota tracker. Thread-safe via asyncio lock."""

    def __init__(self, daily_limit: int, per_minute_limit: int = 999):
        self.daily_limit = daily_limit
        self.per_minute_limit = per_minute_limit
        self._daily_used = 0
        self._minute_used = 0
        self._minute_reset = time.monotonic() + 60
        self._lock = asyncio.Lock()

    async def can_use(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            if now > self._minute_reset:
                self._minute_used = 0
                self._minute_reset = now + 60
            return (
                self._daily_used < self.daily_limit
                and self._minute_used < self.per_minute_limit
            )

    async def consume(self) -> None:
        async with self._lock:
            self._daily_used += 1
            self._minute_used += 1

    def reset_daily(self) -> None:
        self._daily_used = 0


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    daily_limit: int
    per_minute_limit: int = 999
    enabled: bool = True


def build_providers_from_settings() -> list[ProviderConfig]:
    """Build the provider list from app/config.py settings.

    Providers whose API key is empty are marked enabled=False and will
    be silently skipped by the rotator.
    """
    providers = [
        ProviderConfig(
            "finnhub", settings.FINNHUB_API_KEY, daily_limit=800, per_minute_limit=60
        ),
        ProviderConfig("marketaux", settings.MARKETAUX_API_KEY, daily_limit=100),
        ProviderConfig("newsapi", settings.NEWSAPI_API_KEY, daily_limit=100),
        ProviderConfig("alphavantage", settings.ALPHAVANTAGE_API_KEY, daily_limit=25),
        ProviderConfig(
            "polygon",
            settings.POLYGON_API_KEY or settings.MASSIVE_API_KEY,
            daily_limit=999,
            per_minute_limit=5,
        ),
        ProviderConfig("gnews", settings.GNEWS_API_KEY, daily_limit=100),
        ProviderConfig("currentsapi", settings.CURRENTS_API_KEY, daily_limit=600),
        ProviderConfig("thenewsapi", settings.THENEWSAPI_KEY, daily_limit=150),
        ProviderConfig("worldnewsapi", settings.WORLDNEWSAPI_KEY, daily_limit=300),
        ProviderConfig("stockdata", settings.STOCKDATA_API_KEY, daily_limit=100),
    ]
    # Auto-disable providers with no key
    for p in providers:
        if not p.api_key:
            p.enabled = False
    return providers


# ---------------------------------------------------------------------------
# Individual provider fetchers (all use SmartClient)
# ---------------------------------------------------------------------------


async def _fetch_marketaux(
    api_key: str,
    tickers: list[str],
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    symbols = ",".join(tickers[:5])
    url = (
        f"https://api.marketaux.com/v1/news/all"
        f"?symbols={symbols}&filter_entities=true"
        f"&language=en&limit={limit}&api_token={api_key}"
    )
    resp = await client.get(url)
    if resp.status_code != 200:
        logger.warning("[rotator] marketaux HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("data", []):
        try:
            pub = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
        except Exception:
            pub = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                summary=item.get("description", ""),
                source="marketaux",
                published_at=pub,
                tickers=[
                    e["symbol"] for e in item.get("entities", []) if e.get("symbol")
                ],
                sentiment=item.get("entities", [{}])[0].get("sentiment_score")
                if item.get("entities")
                else None,
            )
        )
    return articles


async def _fetch_newsapi(
    api_key: str,
    query: str,
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={query}&language=en&sortBy=publishedAt"
        f"&pageSize={limit}&apiKey={api_key}"
    )
    resp = await client.get(url)
    if resp.status_code != 200:
        logger.warning("[rotator] newsapi HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("articles", []):
        try:
            pub = datetime.fromisoformat(item["publishedAt"].replace("Z", "+00:00"))
        except Exception:
            pub = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                summary=item.get("description", "") or item.get("content", ""),
                source="newsapi",
                published_at=pub,
            )
        )
    return articles


async def _fetch_alphavantage_news(
    api_key: str,
    tickers: list[str],
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    symbols = ",".join(tickers[:5])
    url = (
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&tickers={symbols}&limit={limit}&apikey={api_key}"
    )
    resp = await client.get(url)
    if resp.status_code != 200:
        logger.warning("[rotator] alphavantage HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("feed", []):
        pub_str = item.get("time_published", "")
        try:
            pub_dt = datetime.strptime(pub_str, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            pub_dt = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                summary=item.get("summary", ""),
                source="alphavantage",
                published_at=pub_dt,
                tickers=[ts["ticker"] for ts in item.get("ticker_sentiment", [])],
                sentiment=float(item.get("overall_sentiment_score", 0) or 0),
            )
        )
    return articles


async def _fetch_polygon_news(
    api_key: str,
    ticker: str,
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    url = (
        f"https://api.polygon.io/v2/reference/news"
        f"?ticker={ticker}&limit={limit}&sort=published_utc"
        f"&order=desc&apiKey={api_key}"
    )
    resp = await client.get(url)
    if resp.status_code != 200:
        logger.warning("[rotator] polygon HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("results", []):
        try:
            pub = datetime.fromisoformat(item["published_utc"].replace("Z", "+00:00"))
        except Exception:
            pub = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("article_url", ""),
                summary=item.get("description", ""),
                source="polygon",
                published_at=pub,
                tickers=item.get("tickers", []),
            )
        )
    return articles


async def _fetch_gnews(
    api_key: str,
    query: str,
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    url = (
        f"https://gnews.io/api/v4/search?q={query}&lang=en&max={limit}&token={api_key}"
    )
    resp = await client.get(url)
    if resp.status_code != 200:
        logger.warning("[rotator] gnews HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("articles", []):
        try:
            pub = datetime.fromisoformat(item["publishedAt"].replace("Z", "+00:00"))
        except Exception:
            pub = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                summary=item.get("description", "") or item.get("content", ""),
                source="gnews",
                published_at=pub,
            )
        )
    return articles


async def _fetch_currentsapi(
    api_key: str,
    query: str,
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    url = (
        f"https://api.currentsapi.services/v1/search"
        f"?keywords={query}&language=en&limit={limit}&apiKey={api_key}"
    )
    resp = await client.get(url)
    if resp.status_code != 200:
        logger.warning("[rotator] currentsapi HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("news", []):
        try:
            pub = datetime.fromisoformat(
                item.get("published", datetime.now(UTC).isoformat())
            )
        except Exception:
            pub = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                summary=item.get("description", ""),
                source="currentsapi",
                published_at=pub,
            )
        )
    return articles


async def _fetch_thenewsapi(
    api_key: str,
    query: str,
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    url = (
        f"https://api.thenewsapi.com/v1/news/all"
        f"?search={query}&language=en&limit={limit}&api_token={api_key}"
    )
    resp = await client.get(url)
    if resp.status_code != 200:
        logger.warning("[rotator] thenewsapi HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("data", []):
        try:
            pub = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
        except Exception:
            pub = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                summary=item.get("description", ""),
                source="thenewsapi",
                published_at=pub,
            )
        )
    return articles


async def _fetch_worldnewsapi(
    api_key: str,
    query: str,
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    url = (
        f"https://api.worldnewsapi.com/search-news"
        f"?text={query}&language=en&number={limit}"
    )
    resp = await client.get(url, headers={"x-api-key": api_key})
    if resp.status_code != 200:
        logger.warning("[rotator] worldnewsapi HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("news", []):
        try:
            pub = datetime.fromisoformat(
                item.get("publish_date", datetime.now(UTC).isoformat())
            )
        except Exception:
            pub = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                summary=item.get("text", "")[:500],
                source="worldnewsapi",
                published_at=pub,
            )
        )
    return articles


async def _fetch_stockdata(
    api_key: str,
    tickers: list[str],
    client: SmartClient,
    limit: int = 10,
) -> list[NewsArticle]:
    symbols = ",".join(tickers[:5])
    url = (
        f"https://api.stockdata.org/v1/news/all"
        f"?symbols={symbols}&filter_entities=true"
        f"&language=en&limit={limit}&api_token={api_key}"
    )
    resp = await client.get(url)
    if resp.status_code != 200:
        logger.warning("[rotator] stockdata HTTP %d", resp.status_code)
        return []
    articles = []
    for item in resp.json().get("data", []):
        try:
            pub = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
        except Exception:
            pub = datetime.now(UTC)
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                url=item.get("url", ""),
                summary=item.get("description", "") or item.get("snippet", ""),
                source="stockdata",
                published_at=pub,
                tickers=[
                    e["symbol"] for e in item.get("entities", []) if e.get("symbol")
                ],
            )
        )
    return articles


# ---------------------------------------------------------------------------
# DB persistence — writes NewsArticle objects to the news_articles table
# ---------------------------------------------------------------------------


async def _persist_articles(articles: list[NewsArticle]) -> int:
    """Write articles to DB with ticker tagging and deduplication.

    Uses the same pattern as news_collector.py:
      - Detects tickers via the shared ticker_extractor module
      - One row per detected ticker for easy querying
      - ON CONFLICT (id) DO NOTHING for deduplication
      - Uses %s placeholders (psycopg compatibility shim)
    """
    from app.collectors.news_collector import (
        _detect_tickers_in_text,
        _get_article_id,
        _scrape_article_body_via_service,
    )

    with get_db() as db:
        count = 0

        for article in articles:
            if not article.title:
                continue

            api_summary = article.summary or ""
            summary = ""
            if article.url and (len(api_summary) < 150 or "..." in api_summary):
                try:
                    body = await _scrape_article_body_via_service(article.url)
                    if body:
                        summary = body
                except Exception as e:
                    logger.warning("[rotator] Failed to scrape body for %s: %s", article.url, e)

            if (not summary or len(summary) < 150) and len(api_summary) >= 150:
                summary = api_summary

            if len(summary) < 150:
                continue

            # Use tickers from API if provided, otherwise detect from full text
            if article.tickers:
                detected = set(article.tickers)
            else:
                full_text = f"{article.title} {summary}"
                detected = _detect_tickers_in_text(full_text)

            base_id = hashlib.md5(
                f"{article.title}{article.published_at.isoformat()}".encode()
            ).hexdigest()

            if detected:
                for ticker in detected:
                    ticker_id = _get_article_id(article.title, ticker)
                    db.execute(
                        """
                        INSERT INTO news_articles
                        (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING
                    """,
                        [
                            ticker_id,
                            ticker,
                            article.title[:500],
                            article.source,
                            article.url,
                            article.published_at,
                            summary[:15000],
                            article.source,
                        ],
                    )
                    count += 1
            else:
                # General market news — no specific ticker
                article_id = _get_article_id(article.title, None)
                db.execute(
                    """
                    INSERT INTO news_articles
                    (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO NOTHING
                """,
                    [
                        base_id,
                        None,
                        article.title[:500],
                        article.source,
                        article.url,
                        article.published_at,
                        summary[:15000],
                        article.source,
                    ],
                )
                count += 1

        return count


# ---------------------------------------------------------------------------
# Main Rotator
# ---------------------------------------------------------------------------


class NewsApiRotator:
    """
    Rotates across all free news API providers. Falls back to next provider
    automatically when a quota is hit or a request fails.

    Usage:
        rotator = NewsApiRotator(tickers=["AAPL", "TSLA"])
        async with rotator:
            articles = await rotator.fetch_news(query="AAPL earnings")
    """

    def __init__(
        self,
        providers: list[ProviderConfig] | None = None,
        tickers: list[str] | None = None,
        include_rss: bool = False,  # RSS already runs in news_collector.py
    ):
        self.providers = providers or build_providers_from_settings()
        self.tickers = tickers or []
        self.include_rss = include_rss
        self._quotas: dict[str, QuotaTracker] = {
            p.name: QuotaTracker(p.daily_limit, p.per_minute_limit)
            for p in self.providers
        }
        self._client: SmartClient | None = None

    async def __aenter__(self) -> "NewsApiRotator":
        self._client = SmartClient(base_delay=1.5, max_retries=3, timeout=20.0)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.__aexit__(*_)

    def _get_client(self) -> SmartClient:
        if not self._client:
            raise RuntimeError("Use NewsApiRotator as an async context manager.")
        return self._client

    async def _fetch_from_provider(
        self,
        provider: ProviderConfig,
        query: str,
    ) -> list[NewsArticle]:
        """Route to the correct fetcher based on provider name."""
        c = self._get_client()
        key = provider.api_key

        match provider.name:
            case "finnhub":
                # Delegate to the real finnhub_collector module-level function
                from app.collectors.finnhub_collector import collect_news as fh_collect

                # finnhub_collector writes directly to DB and returns count
                # We call it for each ticker and return empty (already persisted)
                for ticker in self.tickers[:5]:
                    try:
                        await fh_collect(ticker, days_back=3)
                    except Exception as e:
                        logger.warning("[rotator] finnhub failed for %s: %s", ticker, e)
                return []  # Already written to DB by finnhub_collector
            case "marketaux":
                return await _fetch_marketaux(key, self.tickers, c)
            case "newsapi":
                return await _fetch_newsapi(key, query, c)
            case "alphavantage":
                return await _fetch_alphavantage_news(key, self.tickers, c)
            case "polygon":
                return await _fetch_polygon_news(
                    key, self.tickers[0] if self.tickers else "SPY", c
                )
            case "gnews":
                return await _fetch_gnews(key, query, c)
            case "currentsapi":
                return await _fetch_currentsapi(key, query, c)
            case "thenewsapi":
                return await _fetch_thenewsapi(key, query, c)
            case "worldnewsapi":
                return await _fetch_worldnewsapi(key, query, c)
            case "stockdata":
                return await _fetch_stockdata(key, self.tickers, c)
            case _:
                logger.warning("[rotator] Unknown provider: %s", provider.name)
                return []

    async def fetch_news(
        self,
        query: str = "stock market",
        max_per_provider: int = 10,
        persist: bool = True,
    ) -> int:
        """
        Fetch news from all available providers in rotation.
        Skips any provider whose quota is exhausted or has no API key.

        Args:
            query: Search query for keyword-based APIs.
            max_per_provider: Max articles to keep per provider.
            persist: If True, write articles to the DB.

        Returns:
            Total number of new articles written to DB.
        """
        all_articles: list[NewsArticle] = []
        seen_urls: set[str] = set()

        # Randomize provider order to spread load
        providers = list(self.providers)
        random.shuffle(providers)

        exhausted_count = 0
        enabled_count = 0

        for provider in providers:
            if not provider.enabled:
                continue
            enabled_count += 1
            quota = self._quotas[provider.name]
            if not await quota.can_use():
                logger.info(
                    "[rotator] Quota exhausted for %s — skipping", provider.name
                )
                exhausted_count += 1
                continue
            try:
                articles = await self._fetch_from_provider(provider, query)
                await quota.consume()
                for a in articles[:max_per_provider]:
                    if a.url and a.url not in seen_urls:
                        seen_urls.add(a.url)
                        all_articles.append(a)
                if articles:
                    logger.info(
                        "[rotator] Fetched %d articles from %s",
                        len(articles),
                        provider.name,
                    )
            except Exception as exc:
                logger.warning("[rotator] Provider %s failed: %s", provider.name, exc)

        if enabled_count > 0 and exhausted_count == enabled_count:
            raise RuntimeError("All news API keys exhausted")

        # Sort newest-first
        all_articles.sort(key=lambda x: x.published_at, reverse=True)

        # Persist to DB
        if persist and all_articles:
            count = await _persist_articles(all_articles)
            logger.info("[rotator] Persisted %d new articles from API providers", count)
            return count

        return 0

    def reset_daily_quotas(self) -> None:
        """Call this at midnight to reset all daily counters."""
        for tracker in self._quotas.values():
            tracker.reset_daily()
        logger.info("[rotator] Daily quotas reset for all providers")


# ---------------------------------------------------------------------------
# Convenience function for pipeline integration
# ---------------------------------------------------------------------------


async def collect_from_all_apis(
    tickers: list[str],
    query: str = "stock market earnings",
) -> int:
    """
    One-shot convenience function for use in the pipeline.
    Fetches from all configured API providers and persists to DB.

    Usage in data_phase.py:
        from app.collectors.news_api_rotator import collect_from_all_apis
        count = await collect_from_all_apis(tickers)
    """
    async with NewsApiRotator(tickers=tickers) as rotator:
        return await rotator.fetch_news(query=query)

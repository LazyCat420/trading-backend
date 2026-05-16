"""
Web Search Service — Real-Time Multi-Source News Aggregation.

Fetches the FRESHEST possible news from 4 concurrent layers:
  1. DuckDuckGo News API  (ddgs.news, timelimit="d")
  2. Finnhub              (general_news + company_news)
  3. YouTube transcripts  (DB query, last 30 minutes only)
  4. DuckDuckGo Text      (ddgs.text, timelimit="d", fallback)

All layers fire concurrently via asyncio.gather().
Results are merged, deduplicated (Jaccard > 0.6), and sorted newest-first.

Usage:
    from app.services.web_search import searcher

    # Real-time aggregation (preferred for chat)
    articles = await searcher.search_realtime("Iran war latest", ticker="NG")

    # Legacy snippet-only search
    results = await searcher.search("NVDA stock news today")
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# Rate limit: minimum seconds between searches
_MIN_SEARCH_INTERVAL = 3.0
_last_search_time = 0.0


@dataclass
class SearchResult:
    """A single search result (snippet only)."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = "duckduckgo"


@dataclass
class Article:
    """A scraped/fetched article with full text and publish time."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    full_text: str = ""
    source: str = "web"
    published_at: datetime | None = None


from abc import ABC, abstractmethod


class BaseNewsSource(ABC):
    @abstractmethod
    async def fetch_raw(
        self, query: str, ticker: str | None = None, max_results: int = 5
    ) -> list[dict]:
        """Fetch raw data from the provider."""
        pass

    @abstractmethod
    def normalize(self, raw: dict, query: str) -> Article | None:
        """Convert raw data dict into an Article object. Return None if it should be skipped."""
        pass

    async def fetch(
        self, query: str, ticker: str | None = None, max_results: int = 5
    ) -> list[Article]:
        articles = []
        try:
            raw_data = await self.fetch_raw(query, ticker, max_results)
            for item in raw_data:
                article = self.normalize(item, query)
                if article:
                    articles.append(article)
                    if len(articles) >= max_results:
                        break
            logger.info(
                "%s '%s': %d results",
                self.__class__.__name__,
                query[:50],
                len(articles),
            )
        except asyncio.TimeoutError:
            logger.warning("%s timed out for '%s'", self.__class__.__name__, query[:50])
        except Exception as e:
            logger.warning(
                "%s failed for '%s': %s", self.__class__.__name__, query[:50], e
            )
        return articles


class DDGNewsSource(BaseNewsSource):
    async def fetch_raw(
        self, query: str, ticker: str | None = None, max_results: int = 5
    ) -> list[dict]:
        from ddgs import DDGS

        def _do_news():
            with DDGS() as ddgs:
                return list(ddgs.news(query, max_results=max_results, timelimit="d"))

        return await asyncio.to_thread(_do_news)

    def normalize(self, raw: dict, query: str) -> Article | None:
        pub_dt = None
        date_str = raw.get("date", "")
        if date_str:
            try:
                pub_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                pass
        return Article(
            title=raw.get("title", ""),
            url=raw.get("url", raw.get("link", "")),
            snippet=raw.get("body", raw.get("snippet", "")),
            full_text=raw.get("body", raw.get("snippet", "")),
            source="ddg_news",
            published_at=pub_dt,
        )


class FinnhubNewsSource(BaseNewsSource):
    async def fetch_raw(
        self, query: str, ticker: str | None = None, max_results: int = 5
    ) -> list[dict]:
        try:
            from app.config import settings as _settings

            api_key = _settings.FINNHUB_API_KEY
        except Exception:
            api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not api_key:
            return []
        try:
            import finnhub
        except ImportError:
            return []

        client = finnhub.Client(api_key=api_key)
        raw_all = []
        # General market news
        raw_general = await asyncio.wait_for(
            asyncio.to_thread(lambda: client.general_news("general")),
            timeout=5.0,
        )
        if raw_general:
            for r in raw_general:
                r["_source_type"] = "finnhub"
                raw_all.append(r)

        # Ticker-specific company news
        if ticker:
            today_str = datetime.now().strftime("%Y-%m-%d")
            yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            raw_company = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: client.company_news(
                        ticker, _from=yesterday_str, to=today_str
                    )
                ),
                timeout=5.0,
            )
            if raw_company:
                for r in raw_company:
                    r["_source_type"] = f"finnhub_{ticker}"
                    raw_all.append(r)
        return raw_all

    def normalize(self, raw: dict, query: str) -> Article | None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        ts = raw.get("datetime", 0)
        pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        if pub_dt and pub_dt < cutoff:
            return None

        headline = raw.get("headline", "").strip()
        if not headline:
            return None

        search_text = f"{headline} {raw.get('summary', '')}".lower()
        query_words = set(query.lower().split()) - {
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
            "what",
            "how",
            "why",
            "news",
            "today",
            "latest",
        }
        if query_words and not any(w in search_text for w in query_words):
            return None

        return Article(
            title=headline,
            url=raw.get("url", ""),
            snippet=raw.get("summary", "")[:500],
            full_text=raw.get("summary", ""),
            source=raw.get("_source_type", "finnhub"),
            published_at=pub_dt,
        )


class YouTubeRecentSource(BaseNewsSource):
    async def fetch_raw(
        self, query: str, ticker: str | None = None, max_results: int = 5
    ) -> list[dict]:
        from app.db.connection import get_db

        with get_db() as db:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
            if ticker:
                rows = db.execute(
                    """
                    SELECT title, channel, COALESCE(summary, SUBSTRING(raw_transcript, 1, 2000)),
                           published_at, video_id
                    FROM youtube_transcripts
                    WHERE (ticker = %s OR ticker IS NULL) AND published_at > %s
                    ORDER BY published_at DESC LIMIT 1
                """,
                    [ticker, cutoff],
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT title, channel, COALESCE(summary, SUBSTRING(raw_transcript, 1, 2000)),
                           published_at, video_id
                    FROM youtube_transcripts
                    WHERE published_at > %s
                    ORDER BY published_at DESC LIMIT 1
                """,
                    [cutoff],
                ).fetchall()

        raw_list = []
        for r in rows:
            raw_list.append(
                {
                    "title": r[0] or "",
                    "channel": r[1] or "",
                    "content": r[2] or "",
                    "pub_at": r[3],
                    "video_id": r[4] or "",
                }
            )
        return raw_list

    def normalize(self, raw: dict, query: str) -> Article | None:
        pub_at = raw["pub_at"]
        pub_dt = None
        if pub_at:
            try:
                if isinstance(pub_at, datetime):
                    pub_dt = pub_at
                else:
                    pub_dt = datetime.fromisoformat(str(pub_at).replace("Z", "+00:00"))
            except Exception:
                pass
        return Article(
            title=f"{raw['title']} — {raw['channel']}",
            url=f"https://youtube.com/watch?v={raw['video_id']}"
            if raw["video_id"]
            else "",
            snippet=raw["content"][:500],
            full_text=raw["content"],
            source="youtube_recent",
            published_at=pub_dt,
        )


class DDGTextSource(BaseNewsSource):
    async def fetch_raw(
        self, query: str, ticker: str | None = None, max_results: int = 5
    ) -> list[dict]:
        global _last_search_time
        now = time.monotonic()
        wait = _MIN_SEARCH_INTERVAL - (now - _last_search_time)
        if wait > 0:
            await asyncio.sleep(wait)
        from ddgs import DDGS

        def _do_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results, timelimit="d"))

        raw = await asyncio.to_thread(_do_search)
        _last_search_time = time.monotonic()
        return raw

    def normalize(self, raw: dict, query: str) -> Article | None:
        return Article(
            title=raw.get("title", ""),
            url=raw.get("href", raw.get("link", "")),
            snippet=raw.get("body", raw.get("snippet", "")),
            full_text=raw.get("body", raw.get("snippet", "")),
            source="ddg_text",
            published_at=None,
        )


class WebSearchService:
    """Multi-source real-time news aggregation for LLM grounding."""

    def __init__(self, max_results: int = 5, scrape_top_n: int = 3):
        self.max_results = max_results
        self.scrape_top_n = scrape_top_n
        self.sources = [
            DDGNewsSource(),
            FinnhubNewsSource(),
            YouTubeRecentSource(),
            DDGTextSource(),
        ]

    # ── Layer 4: DuckDuckGo Text (fallback) ───────────────────────────

    async def search(
        self, query: str, max_results: int | None = None, timelimit: str | None = "d"
    ) -> list[SearchResult]:
        """Search DuckDuckGo text results with time filtering.

        Returns list of SearchResult with title, url, snippet.
        Rate-limited to avoid bans. Default: last 24h.
        """
        global _last_search_time
        n = max_results or self.max_results

        # Rate limit
        now = time.monotonic()
        wait = _MIN_SEARCH_INTERVAL - (now - _last_search_time)
        if wait > 0:
            await asyncio.sleep(wait)

        results: list[SearchResult] = []
        try:
            from ddgs import DDGS

            def _do_search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=n, timelimit=timelimit))

            raw = await asyncio.to_thread(_do_search)
            _last_search_time = time.monotonic()

            for r in raw:
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", r.get("link", "")),
                        snippet=r.get("body", r.get("snippet", "")),
                    )
                )
            logger.info(
                "DDG text '%s' (timelimit=%s): %d results",
                query[:50],
                timelimit,
                len(results),
            )

        except Exception as e:
            logger.warning("DDG text failed for '%s': %s", query[:50], e)

        return results

    # ── Main Entry Point: Real-Time Aggregation ───────────────────────

    async def search_realtime(
        self,
        query: str,
        ticker: str | None = None,
        max_articles: int = 5,
        scrape_top_n: int = 2,
        max_age_minutes: int = 60,
    ) -> list[Article]:
        """Multi-source real-time news aggregation.

        Fires 4 layers concurrently:
          1. DDG News (last 24h from API, then post-filtered)
          2. Finnhub (last 24h from API, then post-filtered)
          3. YouTube (DB, last 30 min)
          4. DDG Text (last 24h, fallback — only kept if timestamped)

        After aggregation:
          - Deduplicate by Jaccard similarity
          - STRICT FRESHNESS GATE: discard anything older than max_age_minutes
          - Articles without timestamps are discarded (can't verify freshness)
          - Sort newest-first, cap at max_articles
        """
        logger.info(
            "search_realtime START: query='%s', ticker=%s, max_age=%dmin",
            query[:60],
            ticker,
            max_age_minutes,
        )

        # Fire all layers concurrently
        results = await asyncio.gather(
            self.sources[0].fetch(query, ticker=ticker, max_results=8),
            self.sources[1].fetch(query, ticker=ticker, max_results=8),
            self.sources[2].fetch(query, ticker=ticker, max_results=5),
            self.sources[3].fetch(query, ticker=ticker, max_results=3),
            return_exceptions=True,
        )

        # Merge all layers
        all_articles: list[Article] = []
        layer_names = ["DDG News", "Finnhub", "YouTube", "DDG Text"]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Layer %s failed: %s", layer_names[i], result)
                continue
            if isinstance(result, list):
                all_articles.extend(result)
                logger.info("Layer %s: %d articles", layer_names[i], len(result))

        if not all_articles:
            logger.warning(
                "search_realtime: ALL layers returned 0 results for '%s'", query[:50]
            )
            return []

        # Deduplicate by Jaccard title similarity (> 0.6 = duplicate)
        unique = self._deduplicate(all_articles)

        # ── STRICT FRESHNESS GATE ──
        # Discard anything older than max_age_minutes or without a timestamp.
        # The stock market is fast-paced — even 3-6 hour old news is stale.
        now = datetime.now(timezone.utc)
        freshness_cutoff = now - timedelta(minutes=max_age_minutes)
        fresh: list[Article] = []
        stale_count = 0
        no_ts_count = 0

        for a in unique:
            if not a.published_at:
                no_ts_count += 1
                continue  # Can't verify freshness — discard
            if a.published_at < freshness_cutoff:
                stale_count += 1
                continue  # Too old — discard
            fresh.append(a)

        if stale_count or no_ts_count:
            logger.info(
                "Freshness gate (%dmin): kept %d, discarded %d stale + %d no-timestamp",
                max_age_minutes,
                len(fresh),
                stale_count,
                no_ts_count,
            )

        # Sort by publish time (newest first)
        fresh.sort(
            key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        # Cap at max_articles
        final = fresh[:max_articles]

        # Optionally scrape top N for full text (only if they're just snippets)
        if scrape_top_n > 0 and final:
            await self._scrape_top_articles(final, scrape_top_n)

        logger.info(
            "search_realtime DONE: %d fresh articles (from %d total, %d unique) for '%s'",
            len(final),
            len(all_articles),
            len(unique),
            query[:50],
        )
        return final

    def _deduplicate(self, articles: list[Article]) -> list[Article]:
        """Remove duplicate articles by Jaccard title similarity > 0.6."""
        stop_words = {
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
            "by",
            "with",
            "from",
        }
        seen_word_sets: list[set[str]] = []
        unique: list[Article] = []

        for article in articles:
            words = set(article.title.lower().split()) - stop_words
            if not words:
                unique.append(article)
                continue

            is_dup = False
            for seen in seen_word_sets:
                if not seen:
                    continue
                intersection = words & seen
                union = words | seen
                similarity = len(intersection) / len(union)
                if similarity > 0.6:
                    is_dup = True
                    break

            if not is_dup:
                seen_word_sets.append(words)
                unique.append(article)

        if len(articles) != len(unique):
            logger.info("Dedup: %d → %d articles", len(articles), len(unique))

        return unique

    async def _scrape_top_articles(self, articles: list[Article], top_n: int) -> None:
        """Scrape full text for the top N articles that only have snippets."""
        to_scrape = [
            a
            for a in articles[:top_n]
            if a.url and len(a.full_text) < 300  # Only if just a snippet
        ]

        if not to_scrape:
            return

        urls = [a.url for a in to_scrape]
        try:
            from app.collectors.crawl4ai_config import scrape_urls_batch

            logger.info("Scraping %d URLs for full text...", len(urls))
            batch = await asyncio.wait_for(
                scrape_urls_batch(urls, max_chars=4000, fast=True),
                timeout=15.0,
            )
            for i, sr in enumerate(batch):
                if sr.get("success") and sr.get("text") and len(sr["text"]) > 100:
                    to_scrape[i].full_text = sr["text"]
            logger.info(
                "Scraped %d/%d URLs",
                sum(1 for s in batch if s.get("success")),
                len(urls),
            )
        except Exception as e:
            logger.warning("Batch scrape failed: %s: %s", type(e).__name__, e)

    # ── Legacy: search_and_scrape (kept for backward compat) ──────────

    async def search_and_scrape(
        self,
        query: str,
        max_results: int | None = None,
        scrape_top_n: int | None = None,
    ) -> list[Article]:
        """Legacy method. Now delegates to search_realtime for freshness."""
        return await self.search_realtime(
            query, max_articles=max_results or self.max_results
        )

    # ── Context Formatting ────────────────────────────────────────────

    def format_for_context(
        self,
        articles: list[Article],
        max_chars: int = 6000,
        freshness_minutes: int = 60,
    ) -> str:
        """Format articles into a numbered citation context for the LLM prompt.

        Uses [1], [2], [3] numbering. Shows publish time for each source.
        Includes the freshness window so the LLM knows the search scope.
        """
        if not articles:
            return ""

        hrs = freshness_minutes / 60
        window_str = (
            f"{freshness_minutes} minutes"
            if freshness_minutes < 60
            else f"{hrs:.0f} hour{'s' if hrs != 1 else ''}"
        )
        parts = [f"── WEB SEARCH RESULTS (filtered to last {window_str}) ──"]
        parts.append(
            "Cite these sources using [1], [2], [3] in your response. State the age of each source.\n"
        )
        total = 0

        for i, a in enumerate(articles, 1):
            text = a.full_text or a.snippet
            # Truncate individual articles
            if len(text) > 2000:
                text = text[:2000] + "..."

            # Format publish time
            time_str = ""
            if a.published_at:
                try:
                    # Show relative time for freshness context
                    now = datetime.now(timezone.utc)
                    delta = now - a.published_at
                    if delta.total_seconds() < 3600:
                        mins = int(delta.total_seconds() / 60)
                        time_str = f" ({mins}min ago)"
                    elif delta.total_seconds() < 86400:
                        hrs = int(delta.total_seconds() / 3600)
                        time_str = f" ({hrs}h ago)"
                    else:
                        time_str = f" ({a.published_at.strftime('%Y-%m-%d %H:%M')} UTC)"
                except Exception:
                    pass

            source_tag = f"[{a.source}]" if a.source else ""
            entry = f"[{i}] {a.title}{time_str} {source_tag}\nURL: {a.url}\n{text}\n"

            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)

        return "\n".join(parts)


# Singleton
searcher = WebSearchService()

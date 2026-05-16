"""
Finnhub Collector — Fetches news, analyst targets, earnings calendar.

Pure data collector. No LLM calls. No processing.
Writes to: news_articles, fundamentals (analyst price targets only)
Requires: FINNHUB_API_KEY in .env (free tier = 60 calls/min)
"""

import logging

logger = logging.getLogger(__name__)


import hashlib
import datetime
import finnhub
from app.config import settings
from app.db.connection import get_db


def _get_client() -> finnhub.Client:
    """Get Finnhub client. Raises ValueError if no API key."""
    key = settings.FINNHUB_API_KEY
    if not key:
        raise ValueError("FINNHUB_API_KEY not set in .env — get free key at finnhub.io")
    return finnhub.Client(api_key=key)


async def collect_news(ticker: str, days_back: int = 7) -> int:
    """
    Fetch company news from Finnhub via a Smart Delta Sync.
    Queries the newest article in the DB and only requests news
    starting from that date to minimize API payloads.
    Returns the true number of *new* rows inserted.
    """
    with get_db() as db:
        today = datetime.date.today()

        # 1. Check database for newest timestamp
        row = db.execute(
            "SELECT MAX(published_at) FROM news_articles WHERE ticker = %s AND source = 'finnhub'",
            [ticker],
        ).fetchone()
        if row and row[0]:
            try:
                # Parse '2023-10-05 14:00:00' to datetime
                last_dt = datetime.datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                from_date = last_dt.date()
            except Exception:
                from_date = today - datetime.timedelta(days=days_back)
        else:
            # Scenario A: First time / empty
            from_date = today - datetime.timedelta(days=days_back)

        # 2. Fetch from API using the dynamic date window
        try:
            client = _get_client()
            import asyncio

            articles = await asyncio.to_thread(
                lambda: client.company_news(ticker, _from=str(from_date), to=str(today))
            )
        except Exception as e:
            logger.info(f"[finnhub] Error fetching news for {ticker}: {e}")
            return 0

        if not articles:
            logger.info(f"[finnhub] No news for {ticker}")
            return 0

        count = 0
        for a in articles:
            article_id = hashlib.md5(
                f"{a.get('headline', '')}{a.get('datetime', '')}".encode()
            ).hexdigest()

            # 3. True Deduplication (Avoid overwriting AI summaries!)
            exists = db.execute(
                "SELECT 1 FROM news_articles WHERE id = %s", [article_id]
            ).fetchone()
            if exists:
                continue

            published = datetime.datetime.fromtimestamp(
                a.get("datetime", 0), tz=datetime.UTC
            )

            db.execute(
                """
                INSERT INTO news_articles
                (id, ticker, title, publisher, url, published_at, summary, source, collected_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'finnhub', CURRENT_TIMESTAMP)
            """,
                [
                    article_id,
                    ticker,
                    a.get("headline", ""),
                    a.get("source", ""),
                    a.get("url", ""),
                    published,
                    a.get("summary", ""),
                ],
            )
            count += 1

        logger.info(f"[finnhub] {ticker}: {count} NEW articles written (Smart Delta)")
        return count


async def collect_analyst_targets(ticker: str) -> bool:
    """
    Fetch analyst price targets from Finnhub.
    These supplement the yfinance fundamentals with additional data points.
    Returns True if data was written.
    """
    try:
        client = _get_client()
        import asyncio

        target = await asyncio.to_thread(client.price_target, ticker)

        if not target or "targetHigh" not in target:
            logger.info(f"[finnhub] No analyst targets for {ticker}")
            return False

        logger.info(
            f"[finnhub] {ticker}: analyst targets — "
            f"high={target.get('targetHigh')}, "
            f"low={target.get('targetLow')}, "
            f"mean={target.get('targetMean')}"
        )
        return True
    except Exception as e:
        logger.info(f"[finnhub] Error fetching analyst targets for {ticker}: {e}")
        return False


async def collect_earnings_calendar(ticker: str) -> list[dict]:
    """
    Fetch upcoming earnings dates for a ticker.
    Returns list of earnings events (not written to DB — used by alert engine).
    """
    try:
        client = _get_client()
        today = datetime.date.today()
        future = today + datetime.timedelta(days=90)

        import asyncio

        earnings = await asyncio.to_thread(
            client.earnings_calendar, _from=str(today), to=str(future), symbol=ticker
        )

        events = earnings.get("earningsCalendar", [])
        if events:
            logger.info(f"[finnhub] {ticker}: {len(events)} upcoming earnings events")
        else:
            logger.info(f"[finnhub] {ticker}: no upcoming earnings")
        return events
    except Exception as e:
        logger.info(f"[finnhub] Error fetching earnings calendar for {ticker}: {e}")
        return []


async def collect_recommendation_trends(ticker: str) -> list[dict]:
    """
    Fetch analyst recommendation trends (buy/sell/hold/strongBuy/strongSell).
    Returns list of monthly recommendation snapshots.
    """
    try:
        client = _get_client()
        import asyncio

        trends = await asyncio.to_thread(client.recommendation_trends, ticker)

        if trends:
            latest = trends[0]
            logger.info(
                f"[finnhub] {ticker}: recommendations — "
                f"buy={latest.get('buy')}, hold={latest.get('hold')}, "
                f"sell={latest.get('sell')}"
            )
        else:
            logger.info(f"[finnhub] {ticker}: no recommendation trends")
        return trends
    except Exception as e:
        logger.info(f"[finnhub] Error fetching recommendation trends for {ticker}: {e}")
        return []


async def collect_all(ticker: str) -> dict:
    """Run all Finnhub collectors for a single ticker."""
    news_count = await collect_news(ticker)
    targets = await collect_analyst_targets(ticker)
    earnings = await collect_earnings_calendar(ticker)
    recommendations = await collect_recommendation_trends(ticker)

    return {
        "ticker": ticker,
        "news_articles": news_count,
        "analyst_targets": targets,
        "earnings_events": len(earnings),
        "recommendation_snapshots": len(recommendations),
    }

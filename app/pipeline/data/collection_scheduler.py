"""
Collection Scheduler — Smart freshness checks for all data sources.

Prevents re-scraping data that hasn't changed. Each source has a
different refresh cadence based on when new data actually appears:

  Source            Cadence         Why
  ──────────────    ──────────      ─────────────────────────────────
  SEC 13F           Quarterly       Filed 45 days after quarter end
  Congress trades   Daily           New disclosures appear daily
  News (Finnhub)    6 hours         Articles published throughout day
  News (RSS)        6 hours         Feeds update frequently
  YouTube           12 hours        New videos posted daily
  Reddit            6 hours         Posts flow constantly
  Fundamentals      Daily           Price-derived metrics change daily
  Technicals        Daily           Technical indicators recalc daily
  Price history     Daily           Market hours only
  Commodities       6 hours         Trade nearly 24/7

Usage:
    from app.pipeline.data.collection_scheduler import should_collect

    if should_collect("sec_13f"):
        await collect_all_funds()
    else:
        logger.info("[PIPELINE] 13F data is fresh, skipping")
"""

import logging

logger = logging.getLogger(__name__)


import datetime
from app.db.connection import get_db


# Refresh intervals per source (in hours)
REFRESH_INTERVALS = {
    "sec_13f": 24,  # daily (collector handles per-fund skipping)
    "congress": 24,  # daily
    "fred": 12,  # 12 hours (macro data updates infrequently)
    "coingecko": 12,  # 12 hours (crypto prices)
    "news_finnhub": 6,  # 6 hours — articles publish throughout day
    "news_rss": 6,  # 6 hours — feeds update frequently
    "news_yfinance": 6,  # 6 hours — curated headlines refresh often
    "news_api_rotator": 6,  # 6 hours — API news sources
    "youtube": 8,  # 8 hours — new videos posted daily (extraction is slow)
    "reddit": 6,  # 6 hours — posts flow constantly
    "fundamentals": 24,  # daily
    "technicals": 24,  # daily
    "price_history": 24,  # daily
    "commodities": 12,  # 12 hours
    "sec_10k": 24 * 90,  # quarterly
    "balance_sheet": 24 * 90,  # quarterly
    "financials": 24 * 90,  # quarterly
    "institutional": 24,  # daily (yfinance institutional holders)
}

# Content-based freshness queries (fallback if data_source_status
# doesn't have an entry for this source). Uses the actual data
# timestamps from each table.
_FRESHNESS_QUERIES = {
    "sec_13f": """
        SELECT MAX(filing_quarter) FROM sec_13f_holdings
    """,
    "congress": """
        SELECT MAX(disclosure_date) FROM congress_trades
        WHERE party != '' AND party IS NOT NULL
    """,
    "news_finnhub": """
        SELECT MAX(published_at) FROM news_articles
        WHERE source = 'finnhub'
    """,
    "news_rss": """
        SELECT MAX(published_at) FROM news_articles
        WHERE source = 'rss'
    """,
    "news_yfinance": """
        SELECT MAX(published_at) FROM news_articles
        WHERE source = 'yfinance'
    """,
    "news_api_rotator": """
        SELECT MAX(published_at) FROM news_articles
        WHERE source NOT IN ('finnhub', 'rss', 'yfinance')
    """,
    "youtube": """
        SELECT MAX(published_at) FROM youtube_transcripts
    """,
    "reddit": """
        SELECT MAX(created_utc) FROM reddit_posts
    """,
    "fundamentals": """
        SELECT MAX(snapshot_date) FROM fundamentals
    """,
    "technicals": """
        SELECT MAX(date) FROM technicals
    """,
    "price_history": """
        SELECT MAX(date) FROM price_history
    """,
    "commodities": """
        SELECT MAX(date) FROM asset_prices
        WHERE asset_class = 'commodity'
    """,
    "sec_10k": """
        SELECT MAX(extracted_at) FROM sec_10k_extractions
    """,
    "balance_sheet": """
        SELECT MAX(period_end) FROM balance_sheet
    """,
    "financials": """
        SELECT MAX(period_end) FROM financial_history
    """,
}

# Ticker-specific freshness queries (for per-ticker checks)
_TICKER_QUERIES = {
    "news_finnhub": """
        SELECT MAX(published_at) FROM news_articles
        WHERE source = 'finnhub' AND ticker = %s
    """,
    "news_yfinance": """
        SELECT MAX(published_at) FROM news_articles
        WHERE source = 'yfinance' AND ticker = %s
    """,
    "youtube": """
        SELECT MAX(published_at) FROM youtube_transcripts
        WHERE ticker = %s
    """,
    "reddit": """
        SELECT MAX(created_utc) FROM reddit_posts
        WHERE ticker = %s
    """,
    "fundamentals": """
        SELECT MAX(snapshot_date) FROM fundamentals
        WHERE ticker = %s
    """,
    "technicals": """
        SELECT MAX(date) FROM technicals
        WHERE ticker = %s
    """,
    "price_history": """
        SELECT MAX(date) FROM price_history
        WHERE ticker = %s
    """,
}


def _parse_timestamp(val) -> datetime.datetime | None:
    """Parse a DB timestamp value into a datetime."""
    if val is None:
        return None
    if isinstance(val, datetime.datetime):
        return val
    if isinstance(val, datetime.date):
        return datetime.datetime.combine(val, datetime.time(), tzinfo=datetime.UTC)
    if isinstance(val, str):
        # Try common formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.datetime.strptime(val, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.UTC)
                return dt
            except ValueError:
                continue
    return None


def record_collection(
    source: str, ticker: str = "_global_", rows: int = 0, error: str | None = None
):
    """Record that a collection just happened.

    Writes to data_source_status so the scheduler knows when
    each source was last refreshed.
    """
    now = datetime.datetime.now(datetime.UTC).isoformat()
    with get_db() as db:
        if error:
            db.execute(
                """
                INSERT INTO data_source_status
                (source, ticker, last_failure, error_msg, rows_fetched)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source, ticker) DO UPDATE
                SET last_failure = EXCLUDED.last_failure,
                    error_msg = EXCLUDED.error_msg,
                    rows_fetched = EXCLUDED.rows_fetched
            """,
                [source, ticker, now, error, rows],
            )
        else:
            db.execute(
                """
                INSERT INTO data_source_status
                (source, ticker, last_success, rows_fetched)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source, ticker) DO UPDATE
                SET last_success = EXCLUDED.last_success,
                    rows_fetched = EXCLUDED.rows_fetched
            """,
                [source, ticker, now, rows],
            )
    logger.info(
        f"[PIPELINE] [scheduler] Recorded {source}/{ticker}: "
        f"{rows} rows {'(error)' if error else '(ok)'}"
    )


def get_last_collected(
    source: str, ticker: str | None = None
) -> datetime.datetime | None:
    """Get the timestamp of the most recent data for a source.

    Checks data_source_status first (explicit tracking),
    then falls back to content-based queries.

    Args:
        source: One of the keys in REFRESH_INTERVALS
        ticker: Optional ticker for per-ticker freshness

    Returns:
        datetime of last collection, or None if no data exists
    """
    with get_db() as db:
        # Primary: check data_source_status table
        try:
            tk = ticker or "_global_"
            result = db.execute(
                """
                SELECT last_success FROM data_source_status
                WHERE source = %s AND ticker = %s
            """,
                [source, tk],
            ).fetchone()
            if result and result[0]:
                return _parse_timestamp(result[0])
        except Exception:
            pass

        # Fallback: content-based timestamps
        try:
            if ticker and source in _TICKER_QUERIES:
                query = _TICKER_QUERIES[source]
                result = db.execute(query, [ticker]).fetchone()
            elif source in _FRESHNESS_QUERIES:
                query = _FRESHNESS_QUERIES[source]
                result = db.execute(query).fetchone()
            else:
                return None

            if result and result[0]:
                return _parse_timestamp(result[0])
        except Exception:
            pass  # Table may not exist yet (e.g., sec_10k_extractions)
        return None


def hours_since_last(source: str, ticker: str | None = None) -> float | None:
    """Hours since last collection for a source.

    Returns None if no data exists (meaning: always collect).
    """
    last = get_last_collected(source, ticker)
    if last is None:
        return None

    now = datetime.datetime.now(datetime.UTC)
    if last.tzinfo is None:
        last = last.replace(tzinfo=datetime.UTC)

    delta = now - last
    return delta.total_seconds() / 3600


def should_collect(source: str, ticker: str | None = None, force: bool = False) -> bool:
    """Check if a source needs fresh data.

    Args:
        source: Data source key (e.g., "sec_13f", "congress")
        ticker: Optional ticker for per-ticker checks
        force: Override freshness check and always collect

    Returns:
        True if data is stale or missing and should be collected
    """
    if force:
        return True

    if source not in REFRESH_INTERVALS:
        # Unknown source — always collect
        return True

    hours = hours_since_last(source, ticker)

    if hours is None:
        # No data at all — definitely collect
        return True

    interval = REFRESH_INTERVALS[source]
    return hours >= interval


def get_sec_13f_quarter() -> str:
    """Get the current SEC filing quarter (e.g., '2026Q1').

    13F filings are due 45 days after quarter end.
    Returns the quarter that should be currently available.
    """
    now = datetime.datetime.now()
    # Quarter boundary dates and their filing available dates
    # Q4 (Oct-Dec) -> available by Feb 14
    # Q1 (Jan-Mar) -> available by May 14
    # Q2 (Apr-Jun) -> available by Aug 14
    # Q3 (Jul-Sep) -> available by Nov 14
    year = now.year
    month = now.month

    if month >= 11:  # Q3 filings available (Nov+)
        return f"{year}Q3"
    elif month >= 8:  # Q2 filings available (Aug+)
        return f"{year}Q2"
    elif month >= 5:  # Q1 filings available (May+)
        return f"{year}Q1"
    elif month >= 2:  # Q4 filings available (Feb+)
        return f"{year - 1}Q4"
    else:  # Still waiting for Q3 filings (Jan)
        return f"{year - 1}Q3"


def should_collect_congress() -> bool:
    """Smart check for congress trades.

    Only re-scrape if we haven't scraped in the last 24 hours.
    Uses data_source_status for explicit tracking.
    """
    hours = hours_since_last("congress")
    if hours is not None and hours < 24:
        logger.info(
            f"[PIPELINE] [scheduler] Congress: last scraped {hours:.1f}h ago, "
            "skipping (interval=24h)"
        )
        return False
    logger.info("[PIPELINE] [scheduler] Congress: needs refresh")
    return True


def freshness_report() -> str:
    """Generate a human-readable freshness report for all sources."""
    lines = []
    lines.append("DATA FRESHNESS REPORT")
    lines.append("=" * 60)

    for source in sorted(REFRESH_INTERVALS.keys()):
        interval = REFRESH_INTERVALS[source]
        hours = hours_since_last(source)

        if hours is None:
            status = "❌ NO DATA"
            age_str = "never collected"
        elif hours < interval:
            status = "✅ FRESH"
            if hours < 1:
                age_str = f"{hours * 60:.0f}m ago"
            elif hours < 24:
                age_str = f"{hours:.1f}h ago"
            else:
                age_str = f"{hours / 24:.1f}d ago"
        else:
            status = "⚠️  STALE"
            if hours < 24:
                age_str = f"{hours:.1f}h ago"
            else:
                age_str = f"{hours / 24:.1f}d ago"

        interval_str = f"{interval}h" if interval < 24 else f"{interval / 24:.0f}d"
        lines.append(
            f"  {source:<18} {status}  (last: {age_str}, interval: {interval_str})"
        )

    return "\n".join(lines)


def retire_old_scripts():
    """Retire old scraper scripts that haven't been successfully used in 30 days.

    This forces the adaptive scraper to regenerate the script for a site,
    helping to catch up with site redesigns and layout changes.
    """
    try:
        with get_db() as db:
            result = db.execute(
                "UPDATE scraper_scripts SET status = 'retired' "
                "WHERE status = 'active' AND last_success < NOW() - INTERVAL '30 days'"
            )
            count = result.rowcount
            if count > 0:
                logger.info(
                    f"[scheduler] Retired {count} old adaptive scraper scripts."
                )
    except Exception as e:
        logger.error(f"[scheduler] Error retiring old scripts: {e}")

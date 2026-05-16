"""
Watermark Store — Tracks the last collection timestamp per (ticker, source).

Used to ensure collectors only fetch net-new data since the last run,
avoiding re-processing articles/posts/transcripts that have already been
summarized and debated.

Table: ticker_collection_watermarks (created by migrations.py)
"""

import logging
from datetime import datetime, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# Default watermark when no prior collection exists for a (ticker, source)
_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def get_watermark(ticker: str, source: str) -> datetime:
    """Get the last collection timestamp for a (ticker, source) pair.

    Returns _EPOCH (2000-01-01) if no watermark exists, which effectively
    means "collect everything available".
    """
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT last_collected FROM ticker_collection_watermarks "
                "WHERE ticker = %s AND source = %s",
                [ticker, source],
            ).fetchone()
        if row and row[0]:
            return row[0]
    except Exception as e:
        logger.warning(
            "[WATERMARK] get_watermark failed for %s/%s: %s", ticker, source, e
        )
    return _EPOCH


def set_watermark(
    ticker: str, source: str, dt: datetime | None = None
) -> None:
    """Update the watermark for a (ticker, source) pair.

    Args:
        ticker: Stock symbol.
        source: Data source name (e.g., 'news', 'reddit', 'youtube').
        dt: Timestamp to set. Defaults to now() if not provided.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    try:
        with get_db() as db:
            db.execute(
                """
                INSERT INTO ticker_collection_watermarks (ticker, source, last_collected)
                VALUES (%s, %s, %s)
                ON CONFLICT (ticker, source)
                DO UPDATE SET last_collected = EXCLUDED.last_collected
                """,
                [ticker, source, dt],
            )
    except Exception as e:
        logger.warning(
            "[WATERMARK] set_watermark failed for %s/%s: %s", ticker, source, e
        )


def get_all_watermarks(ticker: str) -> dict[str, datetime]:
    """Get all watermarks for a ticker, keyed by source.

    Returns:
        Dict of {source: last_collected} for the ticker.
    """
    result: dict[str, datetime] = {}
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT source, last_collected FROM ticker_collection_watermarks "
                "WHERE ticker = %s",
                [ticker],
            ).fetchall()
        for row in rows:
            result[row[0]] = row[1]
    except Exception as e:
        logger.warning(
            "[WATERMARK] get_all_watermarks failed for %s: %s", ticker, e
        )
    return result

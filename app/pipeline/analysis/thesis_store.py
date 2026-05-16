"""
Thesis Store — Thin DB read/write layer for per-ticker thesis state.

The thesis represents the bot's current standing verdict on a ticker
(BUY / SELL / HOLD + confidence + summary). It persists across cycles
so that Glance-tier skips can return the cached verdict instead of
a hardcoded HOLD, and Standard/Deep runs can inject the existing thesis
as context for delta analysis.

NOTE: There is no save_thesis() here. Thesis fields are written directly
by _log_decision() in decision_engine.py via the INSERT INTO analysis_results
statement. This avoids a race condition where an UPDATE would hit the
*previous* cycle's row instead of the one being inserted.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.db.connection import get_db

logger = logging.getLogger(__name__)


@dataclass
class ThesisRecord:
    """Snapshot of a ticker's current thesis state."""

    ticker: str
    verdict: str  # BUY / SELL / HOLD
    confidence: int
    summary: str
    updated_at: datetime
    unchanged: bool = False


def get_thesis(ticker: str) -> Optional[ThesisRecord]:
    """Fetch the most recent saved thesis for a ticker.

    Returns None if the ticker has never had a thesis persisted.
    """
    try:
        with get_db() as db:
            row = db.execute(
                """
                SELECT ticker, thesis_verdict, thesis_confidence,
                       thesis_summary, thesis_updated_at, thesis_unchanged
                FROM analysis_results
                WHERE ticker = %s AND thesis_verdict IS NOT NULL
                ORDER BY thesis_updated_at DESC LIMIT 1
                """,
                [ticker],
            ).fetchone()
        if not row:
            return None
        return ThesisRecord(
            ticker=row[0],
            verdict=row[1],
            confidence=row[2] or 0,
            summary=row[3] or "",
            updated_at=row[4],
            unchanged=bool(row[5]),
        )
    except Exception as e:
        logger.warning("[THESIS] get_thesis failed for %s: %s", ticker, e)
        return None


def is_thesis_stale(ticker: str, hours: int = 72) -> bool:
    """Check if a ticker's thesis is stale (older than `hours` hours).

    Returns True if no thesis exists (treat as maximally stale).
    """
    thesis = get_thesis(ticker)
    if not thesis:
        return True
    age = (datetime.now(timezone.utc) - thesis.updated_at).total_seconds() / 3600
    return age > hours


def thesis_age_hours(ticker: str) -> float:
    """Return the age of the most recent thesis in hours.

    Returns float('inf') if no thesis exists.
    """
    thesis = get_thesis(ticker)
    if not thesis:
        return float("inf")
    return (datetime.now(timezone.utc) - thesis.updated_at).total_seconds() / 3600


def mark_unchanged(ticker: str) -> None:
    """Mark the latest thesis for a ticker as unchanged (Glance skip).

    Sets thesis_unchanged=TRUE on the most recent analysis_results row.
    This does NOT update thesis_updated_at, so the staleness timer
    keeps ticking until a real Standard/Deep analysis runs.
    """
    try:
        with get_db() as db:
            db.execute(
                """
                UPDATE analysis_results SET thesis_unchanged = TRUE
                WHERE id = (
                    SELECT id FROM analysis_results
                    WHERE ticker = %s ORDER BY created_at DESC LIMIT 1
                )
                """,
                [ticker],
            )
    except Exception as e:
        logger.warning("[THESIS] mark_unchanged failed for %s: %s", ticker, e)

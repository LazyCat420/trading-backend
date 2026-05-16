"""
Ticker Triage — 3-tier research depth classifier.

Classifies each ticker into Glance / Standard / Deep tiers
based on attention data, positions, news volume, and neglect flags.

Runs at the START of each cycle, before data collection.

Tiers:
    Glance   — Analyzed < 24h ago, no catalysts, no position, no neglect flag.
               Skip per-ticker collection, quick change-detection LLM check.
    Standard — Default tier. Normal collection + analysis.
    Deep     — > 72h since analysis, neglect-flagged, high news volume,
               or position with significant loss. Extended collection,
               forced Config C+D escalation, double context window.

Usage:
    from app.pipeline.ticker_triage import classify_tickers, TriageResult
    triage = classify_tickers(tickers, attention_data, positions)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from app.db.connection import get_db

if TYPE_CHECKING:
    from app.pipeline.attention_tracker import AttentionRecord

logger = logging.getLogger(__name__)

# ── Configurable Thresholds ──────────────────────────────────────────
# (hours since last analysis)
GLANCE_THRESHOLD_HOURS = 24
DEEP_THRESHOLD_HOURS = 72


# Time-based heartbeat: force a full Standard review after this many hours
# even if no catalysts fired. Replaces the old cycle-count approach.
HEARTBEAT_HOURS = 48

# High news volume: >= this many NEW articles (since last analysis) triggers Deep.
# This only counts articles the bot hasn't seen yet.
HIGH_NEW_NEWS_DEEP = 5

# Position loss threshold for forced Deep analysis
POSITION_LOSS_DEEP_PCT = 5.0  # >= 5% unrealized loss → Deep

# Max days_since_deep before auto-promote to Deep
MAX_DAYS_SINCE_DEEP = 7


@dataclass
class TriageResult:
    """Result of the triage classification."""

    glance: list[str] = field(default_factory=list)
    standard: list[str] = field(default_factory=list)
    deep: list[str] = field(default_factory=list)

    @property
    def all_tickers(self) -> list[str]:
        """Return all tickers in priority order: deep first, then standard, then glance."""
        return self.deep + self.standard + self.glance

    def get_tier(self, ticker: str) -> str:
        """Get the tier for a specific ticker."""
        if ticker in self.deep:
            return "deep"
        if ticker in self.glance:
            return "glance"
        return "standard"

    def summary(self) -> str:
        """Human-readable summary string."""
        return (
            f"Glance: {len(self.glance)}, "
            f"Standard: {len(self.standard)}, "
            f"Deep: {len(self.deep)}"
        )


def _get_bulk_new_news(
    tickers: list[str],
    attention_data: dict[str, "AttentionRecord"],
) -> dict[str, int]:
    """Count news articles published AFTER each ticker's last analysis.

    This is the core of delta-based triage: we only care about news the bot
    hasn't seen yet. A ticker with 50 articles but 0 new ones since last
    analysis will correctly return 0, staying at Glance.

    Falls back to 24h window if the ticker has never been analyzed.
    """
    if not tickers:
        return {}
    try:
        with get_db() as db:
            # Build per-ticker cutoffs based on last_analyzed_at
            results: dict[str, int] = {}
            fallback_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

            for ticker in tickers:
                attn = attention_data.get(ticker)
                cutoff = (
                    attn.last_analyzed_at
                    if attn and attn.last_analyzed_at
                    else fallback_cutoff
                )
                row = db.execute(
                    "SELECT COUNT(*) FROM news_articles "
                    "WHERE ticker = %s AND published_at > %s",
                    [ticker, cutoff],
                ).fetchone()
                results[ticker] = row[0] if row else 0
            return results
    except Exception:
        return {}


def _get_bulk_upcoming_earnings(
    tickers: list[str], days_ahead: int = 7
) -> dict[str, bool]:
    """Bulk check if tickers have earnings within the next N days."""
    if not tickers:
        return {}
    try:
        with get_db() as db:
            # Check if column exists
            row = db.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'fundamentals' AND column_name = 'next_earnings_date'
                """
            ).fetchone()
            if not row:
                return {}

            cutoff = datetime.now(timezone.utc) + timedelta(days=days_ahead)
            placeholders = ", ".join(["?"] * len(tickers))
            rows = db.execute(
                f"""
                WITH Ranked AS (
                    SELECT ticker, next_earnings_date,
                           ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY snapshot_date DESC) as rn
                    FROM fundamentals
                    WHERE ticker IN ({placeholders}) AND next_earnings_date IS NOT NULL
                )
                SELECT ticker, next_earnings_date FROM Ranked WHERE rn = 1
                """,
                tickers,
            ).fetchall()
            return {r[0]: (r[1] <= cutoff.date()) for r in rows}
    except Exception:
        return {}


def classify_tickers(
    tickers: list[str],
    attention_data: dict[str, "AttentionRecord"],
    positions: list[str],
    user_added: set[str] | None = None,
) -> TriageResult:
    """Classify tickers into Glance / Standard / Deep tiers.

    Pure deterministic classification based on attention state,
    portfolio positions, news volume, and neglect flags.

    Args:
        tickers: All tickers to classify.
        attention_data: Dict mapping ticker -> AttentionRecord.
        positions: List of tickers with open positions.
        user_added: Optional set of user-manually-added tickers
                    (always get at least Standard).

    Returns:
        TriageResult with classified ticker lists.
    """
    result = TriageResult()
    now = datetime.now(timezone.utc)
    user_set = user_added or set()

    # Pre-fetch delta news counts (only articles newer than last analysis)
    bulk_new_news = _get_bulk_new_news(tickers, attention_data)
    bulk_earnings = _get_bulk_upcoming_earnings(tickers)

    for ticker in tickers:
        attn = attention_data.get(ticker)
        new_news_count = bulk_new_news.get(ticker, 0)
        has_earnings = bulk_earnings.get(ticker, False)

        tier = _classify_single(
            ticker, attn, now, positions, user_set, new_news_count, has_earnings
        )
        if tier == "deep":
            result.deep.append(ticker)
        elif tier == "glance":
            result.glance.append(ticker)
        else:
            result.standard.append(ticker)

    logger.info(
        "[TRIAGE] Classification complete: %s | Tickers: %s",
        result.summary(),
        ", ".join(
            f"{t}({'D' if t in result.deep else 'G' if t in result.glance else 'S'})"
            for t in tickers[:20]
        ),
    )

    return result


def _classify_single(
    ticker: str,
    attn: "AttentionRecord | None",
    now: datetime,
    positions: list[str],
    user_added: set[str],
    new_news_count: int,
    has_earnings: bool,
) -> str:
    """Classify a single ticker. Returns 'glance', 'standard', or 'deep'.

    new_news_count: Number of articles published AFTER the ticker's last analysis.
                    This is delta-based — only genuinely new information counts.
    """

    # ── Force Deep conditions (checked first) ──

    # 1. Neglect-flagged tickers always get Deep
    if attn and attn.neglect_flagged:
        logger.debug(
            "[TRIAGE] %s → Deep (neglect-flagged: %s)", ticker, attn.neglect_reason
        )
        return "deep"

    # 2. Never analyzed → Deep (new ticker, needs full research)
    if attn is None or attn.last_analyzed_at is None:
        logger.debug("[TRIAGE] %s → Deep (never analyzed)", ticker)
        return "deep"

    hours_since = (now - attn.last_analyzed_at).total_seconds() / 3600

    # 3. Very stale → Deep (> DEEP_THRESHOLD_HOURS without analysis)
    if hours_since > DEEP_THRESHOLD_HOURS:
        logger.debug(
            "[TRIAGE] %s → Deep (%.0fh since last analysis)", ticker, hours_since
        )
        return "deep"

    # 4. days_since_deep exceeded → Deep (hasn't had deep research recently)
    if attn.days_since_deep >= MAX_DAYS_SINCE_DEEP:
        logger.debug(
            "[TRIAGE] %s → Deep (%d days since last deep research)",
            ticker,
            attn.days_since_deep,
        )
        return "deep"

    # 5. High volume of NEW news → Deep (lots of genuinely new info to process)
    if new_news_count >= HIGH_NEW_NEWS_DEEP:
        logger.debug(
            "[TRIAGE] %s → Deep (%d new articles since last analysis)",
            ticker,
            new_news_count,
        )
        return "deep"

    # 6. Position with significant loss → Deep
    # (Removed to avoid per-ticker queries. Positions always get Standard at least)

    # 7. Upcoming earnings → Deep
    if has_earnings:
        logger.debug("[TRIAGE] %s → Deep (earnings within 7 days)", ticker)
        return "deep"

    # ── Standard conditions ──

    # 8. Has open position → at least Standard (never Glance)
    if ticker in positions:
        return "standard"

    # 9. User-added tickers → at least Standard
    if ticker in user_added:
        return "standard"

    # 10. Moderate staleness → Standard (24-72h since analysis)
    if hours_since > GLANCE_THRESHOLD_HOURS:
        return "standard"

    # 11. Any genuinely new news → Standard
    # Only counts articles published AFTER last analysis, so popular stocks
    # with old-but-numerous articles won't trigger needless re-analysis.
    if new_news_count > 0:
        logger.debug(
            "[TRIAGE] %s → Standard (%d new article(s) since last analysis)",
            ticker,
            new_news_count,
        )
        return "standard"

    # ── Glance conditions ──

    # 12. Fresh data, no catalysts → Glance
    # But check heartbeat timer first: force a full review every HEARTBEAT_HOURS
    # even if no catalysts fired (replaces old MAX_CONSECUTIVE_GLANCE cycle count)
    if hasattr(attn, "last_full_review_at") and attn.last_full_review_at is not None:
        hours_since_full = (now - attn.last_full_review_at).total_seconds() / 3600
        if hours_since_full > HEARTBEAT_HOURS:
            logger.debug(
                "[TRIAGE] %s → Standard (heartbeat: %.0fh since full review)",
                ticker,
                hours_since_full,
            )
            return "standard"

    return "glance"



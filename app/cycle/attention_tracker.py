"""
Attention Tracker — tracks per-ticker engagement across pipeline cycles.

Prevents:
  - Duplicate data collection (content hash comparison)
  - Ticker neglect (flags tickers unreviewed for too many days)
  - Redundant analysis (consecutive skip counting)

Database table: ticker_attention (auto-created via migration)
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)


@dataclass
class AttentionRecord:
    """Snapshot of a ticker's attention state."""

    ticker: str
    last_collected_at: datetime | None
    last_analyzed_at: datetime | None
    last_traded_at: datetime | None
    consecutive_skips: int
    consecutive_holds: int
    days_since_deep: int
    neglect_flagged: bool
    neglect_reason: str | None
    data_hash: str | None
    last_full_review_at: datetime | None = None


def _ensure_table() -> None:
    """Ensure the ticker_attention table exists (called lazily)."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS ticker_attention (
                ticker              TEXT PRIMARY KEY,
                last_collected_at   TIMESTAMPTZ,
                last_analyzed_at    TIMESTAMPTZ,
                last_traded_at      TIMESTAMPTZ,
                consecutive_skips   INTEGER DEFAULT 0,
                consecutive_holds   INTEGER DEFAULT 0,
                days_since_deep     INTEGER DEFAULT 0,
                neglect_flagged     BOOLEAN DEFAULT FALSE,
                neglect_reason      TEXT,
                data_hash           TEXT,
                created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)


def _upsert_field(ticker: str, **kwargs) -> None:
    """Update specific fields on a ticker's attention record (upsert)."""
    with get_db() as db:
        now = datetime.now(timezone.utc)

        # Build SET clause dynamically
        set_parts = ["updated_at = %s"]
        values = [now]
        for key, val in kwargs.items():
            set_parts.append(f"{key} = %s")
            values.append(val)

        set_clause = ", ".join(set_parts)

        # Build INSERT columns/values for ON CONFLICT
        insert_cols = ["ticker", "updated_at"] + list(kwargs.keys())
        insert_placeholders = ["%s", "%s"] + ["%s"] * len(kwargs)
        insert_values = [ticker, now] + list(kwargs.values())

        sql = f"""
            INSERT INTO ticker_attention ({", ".join(insert_cols)})
            VALUES ({", ".join(insert_placeholders)})
            ON CONFLICT (ticker) DO UPDATE SET {set_clause}
        """
        params = insert_values + values

        try:
            db.execute(sql, params)
        except Exception as e:
            logger.warning(
                "[ATTENTION] _upsert_field failed for %s: %s — retrying after rollback",
                ticker,
                e,
            )
            # Connection may be in error state from a prior failed query;
            # rollback to clear it, then retry once.
            try:
                db._conn.rollback()
                db.execute(sql, params)
                logger.info("[ATTENTION] _upsert_field retry succeeded for %s", ticker)
            except Exception as retry_err:
                logger.error(
                    "[ATTENTION] _upsert_field retry ALSO failed for %s: %s",
                    ticker,
                    retry_err,
                )


# ── Recording Functions ──────────────────────────────────────────────


def record_collection(ticker: str, data_hash: str | None = None) -> None:
    """Record that a ticker's data was collected this cycle.

    Args:
        ticker: Stock symbol.
        data_hash: Optional hash of collected content for dedup detection.
    """
    try:
        _ensure_table()
        fields = {"last_collected_at": datetime.now(timezone.utc)}
        if data_hash:
            fields["data_hash"] = data_hash
        _upsert_field(ticker, **fields)
        logger.debug("[ATTENTION] Recorded collection for %s", ticker)
    except Exception as e:
        logger.warning("[ATTENTION] record_collection failed for %s: %s", ticker, e)


def record_analysis(
    ticker: str, action: str, confidence: int, was_deep: bool = False
) -> None:
    """Record that a ticker was analyzed this cycle.

    Args:
        ticker: Stock symbol.
        action: BUY/SELL/HOLD from decision engine.
        confidence: 0-100 confidence score.
        was_deep: Whether this was a Deep-tier analysis.
    """
    try:
        _ensure_table()
        now = datetime.now(timezone.utc)
        with get_db() as db:
            # Fetch current state for incrementing counters
            row = db.execute(
                "SELECT consecutive_skips, consecutive_holds, days_since_deep "
                "FROM ticker_attention WHERE ticker = %s",
                [ticker],
            ).fetchone()

            consecutive_holds = 0
            days_since_deep = 0

            if row:
                if action == "HOLD":
                    consecutive_holds = (row[1] or 0) + 1
                # else: reset consecutive_holds
                days_since_deep = 0 if was_deep else (row[2] or 0)
            elif action == "HOLD":
                consecutive_holds = 1

            _upsert_field(
                ticker,
                last_analyzed_at=now,
                consecutive_skips=0,  # Reset skip counter on analysis
                consecutive_holds=consecutive_holds,
                days_since_deep=days_since_deep,
                neglect_flagged=False,  # Clear neglect flag on analysis
                neglect_reason=None,
                last_full_review_at=now,  # Record heartbeat for non-glance runs
            )
            logger.debug(
                "[ATTENTION] Recorded analysis for %s: %s (%d%%)",
                ticker,
                action,
                confidence,
            )
    except Exception as e:
        logger.warning("[ATTENTION] record_analysis failed for %s: %s", ticker, e)


def record_skip(ticker: str) -> None:
    """Record that a ticker was skipped (Glance tier, no material change).

    Increments the consecutive_skips counter.
    """
    try:
        _ensure_table()
        with get_db() as db:
            row = db.execute(
                "SELECT consecutive_skips FROM ticker_attention WHERE ticker = %s",
                [ticker],
            ).fetchone()
            current = (row[0] or 0) if row else 0
            _upsert_field(ticker, consecutive_skips=current + 1)
            logger.debug(
                "[ATTENTION] Recorded skip for %s (streak: %d)", ticker, current + 1
            )
    except Exception as e:
        logger.warning("[ATTENTION] record_skip failed for %s: %s", ticker, e)


def record_trade(ticker: str) -> None:
    """Record that a trade was executed (BUY or SELL) for this ticker."""
    try:
        _ensure_table()
        _upsert_field(
            ticker,
            last_traded_at=datetime.now(timezone.utc),
            consecutive_holds=0,  # Reset hold streak on actual trade
        )
        logger.debug("[ATTENTION] Recorded trade for %s", ticker)
    except Exception as e:
        logger.warning("[ATTENTION] record_trade failed for %s: %s", ticker, e)


def increment_days_since_deep(tickers: list[str]) -> None:
    """Increment days_since_deep for all given tickers by 1.

    Called once per cycle start to advance the deep-research counter.
    """
    try:
        _ensure_table()
        if not tickers:
            return
        with get_db() as db:
            placeholders = ", ".join(["%s"] * len(tickers))
            db.execute(
                f"""
                UPDATE ticker_attention
                SET days_since_deep = COALESCE(days_since_deep, 0) + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE ticker IN ({placeholders})
                """,
                tickers,
            )
    except Exception as e:
        logger.warning("[ATTENTION] increment_days_since_deep failed: %s", e)


# ── Query Functions ───────────────────────────────────────────────────


def is_data_duplicate(ticker: str, new_hash: str) -> bool:
    """Check if the new data hash matches the stored hash (duplicate detection).

    Args:
        ticker: Stock symbol.
        new_hash: Hash of new collected content.

    Returns:
        True if the data is identical to last collection (duplicate).
    """
    try:
        _ensure_table()
        with get_db() as db:
            row = db.execute(
                "SELECT data_hash FROM ticker_attention WHERE ticker = %s",
                [ticker],
            ).fetchone()
            if row and row[0]:
                return row[0] == new_hash
    except Exception as e:
        logger.warning(
            "[ATTENTION] is_data_duplicate check failed for %s: %s", ticker, e
        )
    return False


def compute_data_hash(content: str) -> str:
    """Compute a lightweight hash of data content for dedup comparison."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def get_attention_summary(tickers: list[str]) -> dict[str, AttentionRecord]:
    """Batch fetch attention records for a list of tickers.

    Returns a dict mapping ticker -> AttentionRecord.
    Tickers with no record get a default (never-seen) record.
    """
    result: dict[str, AttentionRecord] = {}
    if not tickers:
        return result

    try:
        _ensure_table()
        with get_db() as db:
            placeholders = ", ".join(["%s"] * len(tickers))
            rows = db.execute(
                f"""
                SELECT ticker, last_collected_at, last_analyzed_at, last_traded_at,
                       consecutive_skips, consecutive_holds, days_since_deep,
                       neglect_flagged, neglect_reason, data_hash,
                       last_full_review_at
                FROM ticker_attention
                WHERE ticker IN ({placeholders})
                """,
                tickers,
            ).fetchall()

            for row in rows:
                result[row[0]] = AttentionRecord(
                    ticker=row[0],
                    last_collected_at=row[1],
                    last_analyzed_at=row[2],
                    last_traded_at=row[3],
                    consecutive_skips=row[4] or 0,
                    consecutive_holds=row[5] or 0,
                    days_since_deep=row[6] or 0,
                    neglect_flagged=bool(row[7]),
                    neglect_reason=row[8],
                    data_hash=row[9],
                    last_full_review_at=row[10],
                )
    except Exception as e:
        logger.warning(
            "[ATTENTION] get_attention_summary failed (tickers=%d): %s",
            len(tickers),
            e,
        )

    # Fill in defaults for tickers not yet tracked
    for t in tickers:
        if t not in result:
            result[t] = AttentionRecord(
                ticker=t,
                last_collected_at=None,
                last_analyzed_at=None,
                last_traded_at=None,
                consecutive_skips=0,
                consecutive_holds=0,
                days_since_deep=0,
                neglect_flagged=False,
                neglect_reason=None,
                data_hash=None,
            )

    return result


# ── Neglect Detection ─────────────────────────────────────────────────


def flag_neglected_tickers(max_days: int = 5) -> list[str]:
    """Scan all active watchlist tickers and flag any not analyzed in max_days.

    Tickers that HAVE been recently analyzed get their flag cleared.

    Args:
        max_days: Number of days without analysis before flagging.

    Returns:
        List of ticker symbols that were flagged as neglected.
    """
    flagged: list[str] = []
    try:
        _ensure_table()
        with get_db() as db:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)

            # Get all active watchlist tickers
            wl_rows = db.execute(
                "SELECT ticker FROM watchlist WHERE status = 'active'"
            ).fetchall()
            active_tickers = {r[0] for r in wl_rows}

            if not active_tickers:
                return []

            # Ensure all active tickers have attention records
            for t in active_tickers:
                db.execute(
                    """
                    INSERT INTO ticker_attention (ticker, created_at, updated_at)
                    VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (ticker) DO NOTHING
                    """,
                    [t],
                )

            # Flag tickers that haven't been analyzed in max_days
            # (last_analyzed_at is NULL or older than cutoff)
            # Excludes tickers rejected in company_registry — they can never be
            # analyzed so flagging them as neglected is noise.
            rows = db.execute(
                """
                SELECT ta.ticker
                FROM ticker_attention ta
                JOIN watchlist w ON ta.ticker = w.ticker
                LEFT JOIN company_registry cr ON ta.ticker = cr.symbol
                WHERE w.status = 'active'
                  AND (ta.last_analyzed_at IS NULL OR ta.last_analyzed_at < %s)
                  AND ta.neglect_flagged = FALSE
                  AND (cr.rejected IS NULL OR cr.rejected = FALSE)
                """,
                [cutoff],
            ).fetchall()

            for row in rows:
                ticker = row[0]
                reason = f"Not analyzed in {max_days}+ days"
                db.execute(
                    """
                    UPDATE ticker_attention
                    SET neglect_flagged = TRUE,
                        neglect_reason = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE ticker = %s
                    """,
                    [reason, ticker],
                )
                flagged.append(ticker)

            # Clear flags for tickers that have been recently analyzed
            db.execute(
                """
                UPDATE ticker_attention
                SET neglect_flagged = FALSE,
                    neglect_reason = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE neglect_flagged = TRUE
                  AND last_analyzed_at >= %s
                """,
                [cutoff],
            )

            if flagged:
                logger.info(
                    "[ATTENTION] Flagged %d neglected tickers: %s",
                    len(flagged),
                    ", ".join(flagged[:15]),
                )
    except Exception as e:
        logger.warning("[ATTENTION] flag_neglected_tickers failed: %s", e)

    return flagged


def get_neglect_flags() -> list[dict]:
    """Return all currently neglect-flagged tickers.

    Returns:
        List of dicts with ticker, reason, and days_since fields.
    """
    try:
        _ensure_table()
        with get_db() as db:
            rows = db.execute(
                """
                SELECT ticker, neglect_reason, last_analyzed_at
                FROM ticker_attention
                WHERE neglect_flagged = TRUE
                ORDER BY last_analyzed_at ASC NULLS FIRST
                """
            ).fetchall()

            now = datetime.now(timezone.utc)
            result = []
            for row in rows:
                days_since = None
                if row[2]:
                    days_since = (now - row[2]).days
                result.append(
                    {
                        "ticker": row[0],
                        "reason": row[1] or "Unknown",
                        "days_since_analysis": days_since,
                    }
                )
            return result
    except Exception as e:
        logger.warning("[ATTENTION] get_neglect_flags failed: %s", e)
        return []

"""
Data quality flag service — flag bad data items and manage source trust.

Tables used:
  - data_flags    — user flags on individual data items
  - source_trust  — publisher/subreddit/channel trust scores

When a user flags an item:
  1. Record the flag in data_flags
  2. Increment the source's total_flags count
  3. Recalculate flag_rate
  4. If flag_rate > 0.5, auto-downgrade trust to 0.3

Flagged items are:
  - Excluded from context blobs (context_builder checks data_flags)
  - Shown to user in a "Filtered" panel (transparency)
"""

import logging
import uuid
from datetime import datetime, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# Source table → source field mapping for trust tracking
_SOURCE_FIELD_MAP = {
    "news_articles": ("publisher", "publisher"),
    "reddit_posts": ("subreddit", "subreddit"),
    "youtube_transcripts": ("youtube_channel", "channel"),
}


# ── Flag CRUD ────────────────────────────────────────────────────────


def flag_item(
    source_table: str,
    source_id: str,
    flag_type: str,
    reason: str = "",
    ticker: str = "",
) -> dict:
    """Flag a data item as bad. Updates source trust score.

    Args:
        source_table: 'news_articles' | 'reddit_posts' | 'youtube_transcripts'
        source_id: primary key of the flagged row
        flag_type: 'spam' | 'clickbait' | 'irrelevant' | 'outdated' | 'fake'
        reason: optional user explanation
        ticker: optional ticker association

    Returns:
        dict with flag_id and updated trust info
    """
    with get_db() as db:
        flag_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Validate source_table
        valid_tables = {"news_articles", "reddit_posts", "youtube_transcripts"}
        if source_table not in valid_tables:
            raise ValueError(f"source_table must be one of {valid_tables}")

        # Check if already flagged
        existing = db.execute(
            "SELECT id FROM data_flags WHERE source_table = %s AND source_id = %s",
            [source_table, source_id],
        ).fetchone()
        if existing:
            return {"flag_id": existing[0], "already_flagged": True}

        # Insert flag
        db.execute(
            "INSERT INTO data_flags "
            "(id, source_table, source_id, ticker, flag_type, reason, "
            "flagged_by, flagged_at, auto_action) "
            "VALUES (%s, %s, %s, %s, %s, %s, 'user', %s, 'excluded')",
            [flag_id, source_table, source_id, ticker, flag_type, reason, now],
        )

        # Update source trust
        trust_info = _update_source_trust(source_table, source_id)

        logger.info(
            "data_flag: flagged %s.%s as %s (reason: %s)",
            source_table,
            source_id,
            flag_type,
            reason[:50],
        )

        return {
            "flag_id": flag_id,
            "source_table": source_table,
            "source_id": source_id,
            "flag_type": flag_type,
            "trust_updated": trust_info,
        }


def unflag_item(flag_id: str) -> bool:
    """Remove a flag. Returns True if flag existed."""
    with get_db() as db:
        row = db.execute(
            "SELECT id, source_table, source_id FROM data_flags WHERE id = %s",
            [flag_id],
        ).fetchone()
        if not row:
            return False

        source_table, source_id = row[1], row[2]
        db.execute("DELETE FROM data_flags WHERE id = %s", [flag_id])

        # Recalculate trust after removing flag
        _update_source_trust(source_table, source_id)

        logger.info("data_flag: unflagged %s", flag_id)
        return True


def get_flags(
    ticker: str | None = None,
    source_table: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """List flagged items. Optional filters."""
    with get_db() as db:
        query = (
            "SELECT id, source_table, source_id, ticker, flag_type, "
            "reason, flagged_at, auto_action FROM data_flags"
        )
        params: list = []
        clauses = []

        if ticker:
            clauses.append("ticker = %s")
            params.append(ticker.upper().strip())
        if source_table:
            clauses.append("source_table = %s")
            params.append(source_table)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY flagged_at DESC LIMIT %s"
        params.append(limit)

        rows = db.execute(query, params).fetchall()
        return [
            {
                "id": r[0],
                "source_table": r[1],
                "source_id": r[2],
                "ticker": r[3],
                "flag_type": r[4],
                "reason": r[5],
                "flagged_at": r[6].isoformat() if r[6] else None,
                "auto_action": r[7],
            }
            for r in rows
        ]


def get_flagged_source_ids(source_table: str, ticker: str | None = None) -> set:
    """Get set of source_ids that are flagged for a given table.

    Used by context_builder to exclude flagged items efficiently.
    """
    with get_db() as db:
        try:
            if ticker:
                rows = db.execute(
                    "SELECT source_id FROM data_flags "
                    "WHERE source_table = %s AND ticker = %s",
                    [source_table, ticker.upper().strip()],
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT source_id FROM data_flags WHERE source_table = %s",
                    [source_table],
                ).fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()


def get_filtered_report(ticker: str) -> dict:
    """Get a transparency report of what was filtered for a ticker.

    Returns flagged items + blocked sources so user can verify.
    """
    with get_db() as db:
        t = ticker.upper().strip()
        report: dict = {"ticker": t, "flagged_items": [], "blocked_sources": []}

        # Flagged items
        try:
            rows = db.execute(
                "SELECT df.source_table, df.source_id, df.flag_type, "
                "df.reason, df.flagged_at "
                "FROM data_flags df WHERE df.ticker = %s "
                "ORDER BY df.flagged_at DESC",
                [t],
            ).fetchall()
            report["flagged_items"] = [
                {
                    "source_table": r[0],
                    "source_id": r[1],
                    "flag_type": r[2],
                    "reason": r[3],
                    "flagged_at": r[4].isoformat() if r[4] else None,
                }
                for r in rows
            ]
        except Exception:
            pass

        # Blocked sources (trust_score < 0.5)
        try:
            rows = db.execute(
                "SELECT source_type, source_name, trust_score, "
                "total_flags, total_items, flag_rate "
                "FROM source_trust WHERE trust_score < 0.5 "
                "ORDER BY trust_score ASC"
            ).fetchall()
            report["blocked_sources"] = [
                {
                    "source_type": r[0],
                    "source_name": r[1],
                    "trust_score": r[2],
                    "total_flags": r[3],
                    "total_items": r[4],
                    "flag_rate": r[5],
                }
                for r in rows
            ]
        except Exception:
            pass

        return report


# ── Source Trust ──────────────────────────────────────────────────────


def get_source_trust(source_type: str, source_name: str) -> float:
    """Get trust score for a source. Returns 1.0 (trusted) if unknown."""
    with get_db() as db:
        try:
            row = db.execute(
                "SELECT trust_score FROM source_trust "
                "WHERE source_type = %s AND source_name = %s",
                [source_type, source_name],
            ).fetchone()
            return row[0] if row else 1.0
        except Exception:
            return 1.0


def get_untrusted_sources(threshold: float = 0.5) -> list[dict]:
    """Get all sources below trust threshold."""
    with get_db() as db:
        try:
            rows = db.execute(
                "SELECT source_type, source_name, trust_score, "
                "total_flags, total_items, flag_rate "
                "FROM source_trust WHERE trust_score < %s "
                "ORDER BY trust_score ASC",
                [threshold],
            ).fetchall()
            return [
                {
                    "source_type": r[0],
                    "source_name": r[1],
                    "trust_score": r[2],
                    "total_flags": r[3],
                    "total_items": r[4],
                    "flag_rate": r[5],
                }
                for r in rows
            ]
        except Exception:
            return []


# ── Helpers ──────────────────────────────────────────────────────────


def _update_source_trust(source_table: str, source_id: str) -> dict | None:
    """Recalculate trust score for the source of a flagged item."""
    with get_db() as db:
        mapping = _SOURCE_FIELD_MAP.get(source_table)
        if not mapping:
            return None

        source_type, source_column = mapping

        # Get the source name from the original table
        try:
            row = db.execute(
                f"SELECT {source_column} FROM {source_table} WHERE id = %s",
                [source_id],
            ).fetchone()
            if not row or not row[0]:
                return None
            source_name = row[0]
        except Exception:
            return None

        # Count total items from this source
        try:
            total_row = db.execute(
                f"SELECT COUNT(*) FROM {source_table} WHERE {source_column} = %s",
                [source_name],
            ).fetchone()
            total_items = total_row[0] if total_row else 0
        except Exception:
            total_items = 0

        # Count flags for this source
        try:
            flag_row = db.execute(
                "SELECT COUNT(*) FROM data_flags df "
                f"JOIN {source_table} st ON df.source_id = st.id "
                f"WHERE st.{source_column} = %s AND df.source_table = %s",
                [source_name, source_table],
            ).fetchone()
            total_flags = flag_row[0] if flag_row else 0
        except Exception:
            total_flags = 0

        # Calculate trust
        flag_rate = total_flags / max(total_items, 1)
        trust_score = max(0.0, 1.0 - flag_rate)

        # Auto-downgrade if more than half flagged
        if flag_rate > 0.5:
            trust_score = min(trust_score, 0.3)

        now = datetime.now(timezone.utc)
        db.execute(
            "INSERT INTO source_trust "
            "(source_type, source_name, trust_score, total_flags, "
            "total_items, flag_rate, last_updated) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (source_type, source_name) DO NOTHING",
            [
                source_type,
                source_name,
                trust_score,
                total_flags,
                total_items,
                flag_rate,
                now,
            ],
        )

        logger.info(
            "source_trust: %s/%s = %.2f (flags=%d/%d)",
            source_type,
            source_name,
            trust_score,
            total_flags,
            total_items,
        )

        return {
            "source_type": source_type,
            "source_name": source_name,
            "trust_score": trust_score,
            "flag_rate": flag_rate,
        }

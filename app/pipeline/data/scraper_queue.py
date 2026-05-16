"""
JIT Scraper Queue — demand-driven data request system.

When analysis detects missing data, it enqueues a request here
instead of blocking inline. Scraper workers consume by priority.

Priority levels:
    1  = JIT (blocking analysis — highest urgency)
    5  = Routine sweep (background maintenance)

Status lifecycle:
    PENDING → PROCESSING → RESOLVED | FAILED

Usage:
    from app.pipeline.data.scraper_queue import enqueue_request, get_pending_requests

    # Analysis discovers missing news for AAPL:
    enqueue_request("AAPL", "news", priority=1, requested_by_lens="fundamental")

    # Worker picks up the highest-priority pending request:
    requests = get_pending_requests(limit=5)
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta

from app.db.connection import get_db
from app.config import settings

logger = logging.getLogger(__name__)


def enqueue_request(
    ticker: str,
    data_type: str,
    priority: int | None = None,
    requested_by_lens: str | None = None,
    max_retries: int | None = None,
) -> str | None:
    """Enqueue a data scraping request with deduplication.

    Skips if an identical PENDING/PROCESSING request already exists,
    or if the ticker+data_type is in cooldown.

    Args:
        ticker: Stock ticker symbol
        data_type: One of 'news', 'reddit', 'youtube', 'price', 'fundamentals', 'options'
        priority: 1=JIT (blocking), 5=routine. Defaults to settings.SCRAPER_ROUTINE_PRIORITY
        requested_by_lens: Which analytical lens requested this data
        max_retries: Max retry attempts before permanent failure

    Returns:
        Request ID if enqueued, None if deduplicated or in cooldown
    """
    if priority is None:
        priority = settings.SCRAPER_ROUTINE_PRIORITY
    if max_retries is None:
        max_retries = settings.SCRAPER_MAX_RETRIES

    with get_db() as db:
        now = datetime.now(timezone.utc)

        # Dedup: skip if identical request already pending/processing
        existing = db.execute(
            """
            SELECT id FROM scraper_queue
            WHERE ticker = %s AND data_type_requested = %s
              AND status IN ('PENDING', 'PROCESSING')
            LIMIT 1
            """,
            [ticker, data_type],
        ).fetchone()

        if existing:
            logger.debug(
                "[SCRAPER_QUEUE] Dedup: %s/%s already queued (id=%s)",
                ticker,
                data_type,
                existing[0][:8],
            )
            return None

        # Cooldown check: skip if recently scraped or failed and in cooldown
        cooldown_row = db.execute(
            """
            SELECT cooldown_until FROM scraper_queue
            WHERE ticker = %s AND data_type_requested = %s
              AND cooldown_until IS NOT NULL
              AND cooldown_until > %s
            ORDER BY created_at DESC LIMIT 1
            """,
            [ticker, data_type, now],
        ).fetchone()

        if cooldown_row:
            logger.debug(
                "[SCRAPER_QUEUE] Cooldown active: %s/%s until %s",
                ticker,
                data_type,
                cooldown_row[0],
            )
            return None

        # Queue size guard
        queue_size = db.execute(
            "SELECT COUNT(*) FROM scraper_queue WHERE status = 'PENDING'"
        ).fetchone()[0]

        if queue_size >= settings.SCRAPER_MAX_QUEUE_SIZE:
            logger.warning(
                "[SCRAPER_QUEUE] Queue full (%d/%d), dropping %s/%s request",
                queue_size,
                settings.SCRAPER_MAX_QUEUE_SIZE,
                ticker,
                data_type,
            )
            return None

        request_id = str(uuid.uuid4())
        db.execute(
            """
            INSERT INTO scraper_queue
            (id, ticker, data_type_requested, priority, status,
             requested_by_lens, max_retries, created_at)
            VALUES (%s, %s, %s, %s, 'PENDING', %s, %s, %s)
            """,
            [
                request_id,
                ticker,
                data_type,
                priority,
                requested_by_lens,
                max_retries,
                now,
            ],
        )

        logger.info(
            "[SCRAPER_QUEUE] Enqueued: %s/%s priority=%d lens=%s (id=%s)",
            ticker,
            data_type,
            priority,
            requested_by_lens or "none",
            request_id[:8],
        )
        return request_id


def get_pending_requests(limit: int = 10) -> list[dict]:
    """Get highest-priority pending requests.

    Returns requests ordered by priority ASC (1=highest), then created_at ASC.
    """
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, ticker, data_type_requested, priority,
                   requested_by_lens, retry_count, max_retries, created_at
            FROM scraper_queue
            WHERE status = 'PENDING'
            ORDER BY priority ASC, created_at ASC
            LIMIT %s
            """,
            [limit],
        ).fetchall()

        return [
            {
                "id": r[0],
                "ticker": r[1],
                "data_type": r[2],
                "priority": r[3],
                "requested_by_lens": r[4],
                "retry_count": r[5],
                "max_retries": r[6],
                "created_at": str(r[7]),
            }
            for r in rows
        ]


def mark_processing(request_id: str) -> bool:
    """Atomically claim a request for processing.

    Returns True if successfully claimed (status was PENDING),
    False if already claimed by another worker.
    """
    with get_db() as db:
        now = datetime.now(timezone.utc)

        # Atomic check-and-set: only update if still PENDING
        result = db.execute(
            """
            UPDATE scraper_queue
            SET status = 'PROCESSING', started_at = %s
            WHERE id = %s AND status = 'PENDING'
            """,
            [now, request_id],
        )

        # Database driver doesn't expose rowcount on UPDATE in all versions,
        # so verify the claim succeeded
        row = db.execute(
            "SELECT status FROM scraper_queue WHERE id = %s",
            [request_id],
        ).fetchone()

        if row and row[0] == "PROCESSING":
            logger.debug("[SCRAPER_QUEUE] Claimed request %s", request_id[:8])
            return True

        logger.debug(
            "[SCRAPER_QUEUE] Failed to claim %s (already taken)", request_id[:8]
        )
        return False


def mark_resolved(request_id: str) -> None:
    """Mark a request as successfully resolved.
    Sets a 6-hour cooldown to prevent immediate JIT re-scraping.
    """
    with get_db() as db:
        now = datetime.now(timezone.utc)
        cooldown = now + timedelta(hours=6)
        db.execute(
            """
            UPDATE scraper_queue
            SET status = 'RESOLVED', resolved_at = %s, cooldown_until = %s
            WHERE id = %s
            """,
            [now, cooldown, request_id],
        )
        logger.debug("[SCRAPER_QUEUE] Resolved request %s", request_id[:8])


def mark_failed(request_id: str, error: str) -> None:
    """Mark a request as failed. Re-queues if retries remain.

    On permanent failure (retries exhausted), sets a 24h cooldown
    to prevent infinite request loops.
    """
    with get_db() as db:
        now = datetime.now(timezone.utc)

        row = db.execute(
            "SELECT retry_count, max_retries FROM scraper_queue WHERE id = %s",
            [request_id],
        ).fetchone()

        if not row:
            return

        retry_count, max_retries = row[0], row[1]
        new_retry_count = retry_count + 1

        if new_retry_count < max_retries:
            # Re-queue for retry
            db.execute(
                """
                UPDATE scraper_queue
                SET status = 'PENDING', retry_count = %s, error_message = %s,
                    started_at = NULL
                WHERE id = %s
                """,
                [new_retry_count, error, request_id],
            )
            logger.info(
                "[SCRAPER_QUEUE] Retry %d/%d for request %s: %s",
                new_retry_count,
                max_retries,
                request_id[:8],
                error[:100],
            )
        else:
            # Permanent failure — set 24h cooldown
            cooldown = now + timedelta(hours=24)
            db.execute(
                """
                UPDATE scraper_queue
                SET status = 'FAILED', retry_count = %s, error_message = %s,
                    resolved_at = %s, cooldown_until = %s
                WHERE id = %s
                """,
                [new_retry_count, error, now, cooldown, request_id],
            )
            logger.warning(
                "[SCRAPER_QUEUE] Permanently failed request %s after %d retries: %s",
                request_id[:8],
                new_retry_count,
                error[:100],
            )


def cleanup_stale(timeout_minutes: int = 10) -> int:
    """Reset requests stuck in PROCESSING for longer than timeout.

    Returns the number of requests reset to PENDING.
    """
    with get_db() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)

        stale = db.execute(
            """
            UPDATE scraper_queue
            SET status = 'PENDING', started_at = NULL
            WHERE status = 'PROCESSING' AND started_at < %s
            RETURNING id
            """,
            [cutoff],
        ).fetchall()

        count = len(stale)

        if count:
            logger.info(
                "[SCRAPER_QUEUE] Reset %d stale requests back to PENDING", count
            )
        return count


def get_queue_stats() -> dict:
    """Get queue statistics for monitoring."""
    with get_db() as db:
        stats = db.execute(
            """
            SELECT
                status,
                COUNT(*) as count,
                AVG(priority) as avg_priority
            FROM scraper_queue
            GROUP BY status
            """,
        ).fetchall()

        result = {
            "total": 0,
            "pending": 0,
            "processing": 0,
            "resolved": 0,
            "failed": 0,
            "avg_priority": 0.0,
        }

        for row in stats:
            status_lower = row[0].lower()
            result[status_lower] = row[1]
            result["total"] += row[1]
            if status_lower == "pending":
                result["avg_priority"] = round(row[2] or 0, 1)

        return result


def purge_resolved(older_than_hours: int = 48) -> int:
    """Remove resolved requests older than the specified age.

    Keeps the queue table from growing indefinitely.
    Returns the number of rows deleted.
    """
    with get_db() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)

        count_row = db.execute(
            """
            SELECT COUNT(*) FROM scraper_queue
            WHERE status = 'RESOLVED' AND resolved_at < %s
            """,
            [cutoff],
        ).fetchone()
        count = count_row[0] if count_row else 0

        if count > 0:
            db.execute(
                """
                DELETE FROM scraper_queue
                WHERE status = 'RESOLVED' AND resolved_at < %s
                """,
                [cutoff],
            )
            logger.info(
                "[SCRAPER_QUEUE] Purged %d resolved requests older than %dh",
                count,
                older_than_hours,
            )

        return count

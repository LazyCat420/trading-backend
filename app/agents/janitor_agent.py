"""
LLM Janitor Agent
Periodically cleans up old database entries, moves stale data to archive,
prunes outdated analysis blobs, and auto-purges expired archive entries.

Runs via APScheduler at midnight daily.
"""

import logging
from datetime import datetime, timezone, timedelta

from app.db.connection import get_db

logger = logging.getLogger(__name__)


async def _archive_old_news(archive_days: int = 14, purge_days: int = 60):
    """Move news articles older than archive_days to the archive table."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=archive_days)
    purge_at = datetime.now(timezone.utc) + timedelta(days=purge_days)

    with get_db() as db:
        # Move old news to archive
        archived = 0
        while True:
            rows = db.execute(
                """
                SELECT id, 'news_articles', ticker, title, summary, published_at
                FROM news_articles
                WHERE published_at < %s
                LIMIT 500
                """,
                [cutoff],
            ).fetchall()
            
            if not rows:
                break

            for row in rows:
                try:
                    db.execute(
                        """
                        INSERT INTO data_archive
                            (source_table, source_id, ticker, title, content, original_date, purge_after)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source_table, source_id) DO NOTHING
                        """,
                        ["news_articles", row[0], row[2], row[3], row[4], row[5], purge_at],
                    )
                    db.execute("DELETE FROM news_articles WHERE id = %s", [row[0]])
                    archived += 1
                except Exception as e:
                    logger.warning("[JANITOR] Failed to archive news %s: %s", row[0], e)

        if archived:
            logger.info("[JANITOR] Archived %d old news articles", archived)

        # Move old reddit posts
        reddit_archived = 0
        while True:
            rows = db.execute(
                """
                SELECT id, 'reddit_posts', ticker, title, body, created_utc
                FROM reddit_posts
                WHERE created_utc < %s
                LIMIT 500
                """,
                [cutoff],
            ).fetchall()
            
            if not rows:
                break

            for row in rows:
                try:
                    db.execute(
                        """
                        INSERT INTO data_archive
                            (source_table, source_id, ticker, title, content, original_date, purge_after)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source_table, source_id) DO NOTHING
                        """,
                        ["reddit_posts", row[0], row[2], row[3], row[4], row[5], purge_at],
                    )
                    db.execute("DELETE FROM reddit_posts WHERE id = %s", [row[0]])
                    reddit_archived += 1
                except Exception as e:
                    logger.warning("[JANITOR] Failed to archive reddit %s: %s", row[0], e)

        if reddit_archived:
            logger.info("[JANITOR] Archived %d old reddit posts", reddit_archived)

        return archived + reddit_archived


async def _prune_old_analysis_blobs(days: int = 14):
    """NULL out massive JSON blobs from old analysis results to save space.

    Keeps: ticker, confidence, thesis_verdict, thesis_summary, cycle_id, created_at.
    Removes: result_json, rationale, agent_results (MBs of raw text).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with get_db() as db:
        total_pruned = 0
        while True:
            # We use a subquery to select a batch of ids to update
            res = db.execute(
                """
                UPDATE analysis_results
                SET result_json = NULL
                WHERE id IN (
                    SELECT id FROM analysis_results
                    WHERE created_at < %s
                      AND result_json IS NOT NULL
                    LIMIT 500
                )
                RETURNING id
                """,
                [cutoff],
            ).fetchall()
            
            pruned = len(res)
            total_pruned += pruned
            if pruned < 500:
                break
        
        if total_pruned > 0:
            logger.info("[JANITOR] Pruned old analysis blobs (cutoff: %s, total: %d)", cutoff.date(), total_pruned)


async def _purge_expired_archive():
    """Delete archive entries whose 60-day timer has expired."""
    with get_db() as db:
        db.execute(
            "DELETE FROM data_archive WHERE purge_after < CURRENT_TIMESTAMP"
        )
        logger.info("[JANITOR] Purged expired archive entries")


async def _manage_quarantine_expiration(expire_days: int = 7):
    """Remove tickers from quarantine after expiration period."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=expire_days)
    with get_db() as db:
        db.execute(
            "DELETE FROM ticker_quarantine WHERE quarantined_at < %s",
            [cutoff],
        )
        logger.info("[JANITOR] Expired old quarantine entries (cutoff: %s)", cutoff.date())


async def _prune_stale_debug_data(days: int = 30):
    """Remove old entries from rejected_symbols debug bucket."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with get_db() as db:
        db.execute(
            "DELETE FROM rejected_symbols WHERE created_at < %s",
            [cutoff],
        )
        logger.info("[JANITOR] Pruned stale rejected_symbols debug data (cutoff: %s)", cutoff.date())


async def _purge_agent_traces(days: int = 7):
    """JAN-03: Purge agent_traces older than X days in loop-until-empty batches."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    total_deleted = 0
    with get_db() as db:
        while True:
            # Batching to prevent transaction log bloat
            res = db.execute(
                """
                DELETE FROM agent_traces 
                WHERE id IN (
                    SELECT id FROM agent_traces 
                    WHERE created_at < %s 
                    LIMIT 1000
                )
                RETURNING id
                """,
                [cutoff]
            ).fetchall()
            deleted = len(res)
            total_deleted += deleted
            if deleted < 1000:
                break
    if total_deleted > 0:
        logger.info("[JANITOR] Purged %d old agent_traces (cutoff: %s)", total_deleted, cutoff.date())
    return total_deleted


async def _purge_agent_loop_stats(days: int = 7):
    """JAN-04: Purge agent_loop_stats older than X days in loop-until-empty batches."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    total_deleted = 0
    with get_db() as db:
        while True:
            res = db.execute(
                """
                DELETE FROM agent_loop_stats 
                WHERE id IN (
                    SELECT id FROM agent_loop_stats 
                    WHERE created_at < %s 
                    LIMIT 1000
                )
                RETURNING id
                """,
                [cutoff]
            ).fetchall()
            deleted = len(res)
            total_deleted += deleted
            if deleted < 1000:
                break
    if total_deleted > 0:
        logger.info("[JANITOR] Purged %d old agent_loop_stats (cutoff: %s)", total_deleted, cutoff.date())
    return total_deleted


async def _purge_pending_approvals(days: int = 3):
    """JAN-05: Purge pending_approvals older than X days in loop-until-empty batches."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    total_deleted = 0
    with get_db() as db:
        while True:
            res = db.execute(
                """
                DELETE FROM pending_approvals 
                WHERE id IN (
                    SELECT id FROM pending_approvals 
                    WHERE created_at < %s 
                    LIMIT 1000
                )
                RETURNING id
                """,
                [cutoff]
            ).fetchall()
            deleted = len(res)
            total_deleted += deleted
            if deleted < 1000:
                break
    if total_deleted > 0:
        logger.info("[JANITOR] Purged %d old pending_approvals (cutoff: %s)", total_deleted, cutoff.date())
    return total_deleted


async def _write_janitor_run_log(run_details: dict):
    """JAN-07: Write a summary of the janitor run to janitor_run_log table."""
    try:
        import json
        with get_db() as db:
            db.execute(
                """
                INSERT INTO janitor_run_log (run_time, details)
                VALUES (NOW(), %s)
                """,
                [json.dumps(run_details)]
            )
    except Exception as e:
        logger.warning("[JANITOR] Failed to write run log: %s", e)


async def run_janitor_cleanup():
    """Main execution function for the Janitor."""
    logger.info("[JANITOR] Starting database cleanup cycle...")
    try:
        archived = await _archive_old_news()
        await _prune_old_analysis_blobs()
        await _purge_expired_archive()
        await _manage_quarantine_expiration()
        await _prune_stale_debug_data()
        
        traces_deleted = await _purge_agent_traces()
        stats_deleted = await _purge_agent_loop_stats()
        approvals_deleted = await _purge_pending_approvals()

        run_details = {
            "archived_items": archived,
            "traces_deleted": traces_deleted,
            "stats_deleted": stats_deleted,
            "approvals_deleted": approvals_deleted,
            "status": "success"
        }
        await _write_janitor_run_log(run_details)

        logger.info(
            "[JANITOR] Cleanup complete. Archived %d items.", archived
        )
    except Exception as e:
        logger.error("[JANITOR] Cleanup failed: %s", e, exc_info=True)

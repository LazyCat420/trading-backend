"""
Data Lifecycle Manager — 3-stage data lifecycle enforcement.

Stages:
    Active (0-72h)    → Raw content preserved, available for multi-angle analysis
    Summarized (72h+) → Raw content replaced with LLM summary, saves storage
    Archived (30d+)   → Deleted entirely (unless linked to winning strategies)

Exception: Records linked to winning strategies (via strategy_candidates →
strategy_performance where win=True) are exempt from summarization and archival.

Usage:
    from app.pipeline.data.data_lifecycle import run_lifecycle_pass

    # Run as part of post-cycle maintenance:
    stats = await run_lifecycle_pass()
"""

import logging
from datetime import datetime, timezone, timedelta

from app.config import settings
from app.db.connection import get_db
from app.services.vllm_client import llm, Priority

logger = logging.getLogger(__name__)


def is_off_peak() -> bool:
    """Check if the current time is outside active US market hours.
    Market hours: 9:30 AM - 4:00 PM EST, Mon-Fri.
    Off-peak is anything else.
    """
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        # Fallback for Python < 3.9
        import pytz

        now = datetime.now(pytz.timezone("America/New_York"))

    # Weekend
    if now.weekday() >= 5:
        return True

    # Before 9:30 AM
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return True

    # After 4:00 PM
    if now.hour >= 16:
        return True

    return False


def _get_winning_tickers() -> set[str]:
    """Get tickers linked to winning strategy performances.

    These records are exempt from summarization and archival.
    """
    try:
        with get_db() as db:
            rows = db.execute(
                """
                SELECT DISTINCT sp.ticker
                FROM strategy_performance sp
                WHERE sp.win = TRUE
                """
            ).fetchall()
            return {row[0] for row in rows}
    except Exception:
        return set()


def _summarize_table(
    table: str,
    content_column: str,
    summary_column: str,
    ticker_column: str = "ticker",
    id_column: str = "id",
    timestamp_column: str = "published_at",
    exempt_tickers: set[str] | None = None,
) -> dict:
    """Find records older than RAW_DATA_TTL_HOURS and mark them for summarization.

    This function does NOT perform the LLM summarization itself — it identifies
    candidates and clears raw content for records that already have summaries.

    Returns stats about records processed.
    """
    with get_db() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=settings.RAW_DATA_TTL_HOURS
        )
        exempt = exempt_tickers or set()

        stats = {"candidates": 0, "cleared": 0, "exempt": 0}

        try:
            # Find records with raw content that are past TTL
            # and already have a summary (can clear raw content)
            rows = db.execute(
                f"""
                SELECT {id_column}, {ticker_column}
                FROM {table}
                WHERE {timestamp_column} < %s
                  AND {content_column} IS NOT NULL
                  AND LENGTH({content_column}) > 100
                  AND {summary_column} IS NOT NULL
                  AND LENGTH({summary_column}) > 10
                LIMIT 500
                """,
                [cutoff],
            ).fetchall()

            for row in rows:
                record_id, ticker = row[0], row[1]
                if ticker in exempt:
                    stats["exempt"] += 1
                    continue

                # Clear raw content — summary already exists
                db.execute(
                    f"""
                    UPDATE {table}
                    SET {content_column} = '[SUMMARIZED — raw content archived]'
                    WHERE {id_column} = %s
                    """,
                    [record_id],
                )
                stats["cleared"] += 1

            # Find records past TTL that still need summarization
            need_summary = db.execute(
                f"""
                SELECT COUNT(*)
                FROM {table}
                WHERE {timestamp_column} < %s
                  AND {content_column} IS NOT NULL
                  AND LENGTH({content_column}) > 100
                  AND ({summary_column} IS NULL OR LENGTH({summary_column}) < 10)
                """,
                [cutoff],
            ).fetchone()

            stats["candidates"] = need_summary[0] if need_summary else 0

        except Exception as e:
            logger.warning("[LIFECYCLE] Error processing %s: %s", table, e)

        return stats


def _archive_old_records(
    table: str,
    timestamp_column: str = "published_at",
    exempt_tickers: set[str] | None = None,
) -> int:
    """Delete records older than ARCHIVE_TTL_DAYS.

    Exempt tickers (linked to winning strategies) are preserved.
    Returns the number of records deleted.
    """
    with get_db() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.ARCHIVE_TTL_DAYS)
        exempt = exempt_tickers or set()

        try:
            # Count candidates for deletion
            if exempt:
                placeholders = ", ".join(["%s"] * len(exempt))
                count_row = db.execute(
                    f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE {timestamp_column} < %s
                      AND (ticker NOT IN ({placeholders}) OR ticker IS NULL)
                    """,
                    [cutoff, *exempt],
                ).fetchone()

                count = count_row[0] if count_row else 0
                if count > 0:
                    db.execute(
                        f"""
                        DELETE FROM {table}
                        WHERE {timestamp_column} < %s
                          AND (ticker NOT IN ({placeholders}) OR ticker IS NULL)
                        """,
                        [cutoff, *exempt],
                    )
            else:
                count_row = db.execute(
                    f"""
                    SELECT COUNT(*) FROM {table}
                    WHERE {timestamp_column} < %s
                    """,
                    [cutoff],
                ).fetchone()

                count = count_row[0] if count_row else 0
                if count > 0:
                    db.execute(
                        f"""
                        DELETE FROM {table}
                        WHERE {timestamp_column} < %s
                        """,
                        [cutoff],
                    )

            if count > 0:
                logger.info(
                    "[LIFECYCLE] Archived %d records from %s (>%dd old)",
                    count,
                    table,
                    settings.ARCHIVE_TTL_DAYS,
                )
            return count

        except Exception as e:
            logger.warning("[LIFECYCLE] Archive error for %s: %s", table, e)
            return 0


async def run_lifecycle_pass() -> dict:
    """Run a full lifecycle maintenance pass.

    Steps:
        1. Identify winning tickers (exempt from cleanup)
        2. Clear raw content from summarized records (>72h old)
        3. Delete archived records (>30d old)
        4. Clean resolved scraper queue entries

    Returns stats dict.
    """
    logger.info("[LIFECYCLE] Starting lifecycle maintenance pass")

    exempt_tickers = _get_winning_tickers()
    if exempt_tickers:
        logger.info(
            "[LIFECYCLE] %d tickers exempt (linked to winning strategies): %s",
            len(exempt_tickers),
            ", ".join(sorted(exempt_tickers)[:10]),
        )

    stats = {
        "summarization": {},
        "archival": {},
        "exempt_tickers": len(exempt_tickers),
    }

    # ── Stage 1: Summarization (clear raw content where summary exists) ──
    summarization_targets = [
        ("news_articles", "summary", "llm_summary", "published_at"),
        ("reddit_posts", "body", "summary", "created_utc"),
        ("youtube_transcripts", "raw_transcript", "summary", "published_at"),
    ]

    for table, content_col, summary_col, ts_col in summarization_targets:
        result = _summarize_table(
            table=table,
            content_column=content_col,
            summary_column=summary_col,
            timestamp_column=ts_col,
            exempt_tickers=exempt_tickers,
        )
        stats["summarization"][table] = result
        if result["cleared"] > 0:
            logger.info(
                "[LIFECYCLE] %s: cleared %d raw records, %d still need LLM summary, %d exempt",
                table,
                result["cleared"],
                result["candidates"],
                result["exempt"],
            )

    # ── Stage 2: Archival (delete old records) ──
    archive_targets = [
        ("news_articles", "published_at"),
        ("reddit_posts", "created_utc"),
        ("youtube_transcripts", "published_at"),
    ]

    total_archived = 0
    for table, ts_col in archive_targets:
        count = _archive_old_records(
            table=table,
            timestamp_column=ts_col,
            exempt_tickers=exempt_tickers,
        )
        stats["archival"][table] = count
        total_archived += count

    # ── Stage 3: Clean scraper queue ──
    try:
        from app.pipeline.data.scraper_queue import purge_resolved

        queue_purged = purge_resolved(older_than_hours=48)
        stats["queue_purged"] = queue_purged
    except Exception as e:
        logger.debug("[LIFECYCLE] Queue purge skipped: %s", e)
        stats["queue_purged"] = 0

    logger.info(
        "[LIFECYCLE] Pass complete: %d raw records cleared, %d archived, %d queue entries purged",
        sum(v.get("cleared", 0) for v in stats["summarization"].values()),
        total_archived,
        stats.get("queue_purged", 0),
    )

    return stats


async def summarize_stale_records(limit: int = 50) -> int:
    """Aggressively summarize raw data using the Jetson LLM.

    Runs during off-peak hours to free up VRAM during trading hours.
    Finds records that need summarization and uses the LLM to generate one.
    """
    if not is_off_peak():
        logger.debug("[LIFECYCLE] Skipping LLM summarization (active market hours)")
        return 0

    logger.info("[LIFECYCLE] Starting off-peak LLM summarization")
    with get_db() as db:
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=settings.RAW_DATA_TTL_HOURS
        )

        summarization_targets = [
            ("news_articles", "summary", "llm_summary", "id", "published_at"),
            ("reddit_posts", "body", "summary", "id", "created_utc"),
            (
                "youtube_transcripts",
                "raw_transcript",
                "summary",
                "video_id",
                "published_at",
            ),
        ]

        total_summarized = 0

        system_prompt = (
            "You are a financial data summarizer. Summarize the following raw text into a concise, "
            "dense 3-4 sentence paragraph. Focus on actionable insights, numbers, sentiment, and "
            "forward-looking statements. Do not add any conversational filler."
        )

        for table, content_col, summary_col, id_col, ts_col in summarization_targets:
            if total_summarized >= limit:
                break

            try:
                # Find records that need summarization
                rows = db.execute(
                    f"""
                    SELECT {id_col}, ticker, {content_col}
                    FROM {table}
                    WHERE {ts_col} < %s
                      AND {content_col} IS NOT NULL
                      AND LENGTH({content_col}) > 100
                      AND ({summary_col} IS NULL OR LENGTH({summary_col}) < 10)
                    LIMIT %s
                    """,
                    [cutoff, limit - total_summarized],
                ).fetchall()

                for row in rows:
                    record_id, ticker, raw_content = row[0], row[1], row[2]

                    try:
                        user_prompt = (
                            f"Summarize this data for {ticker}:\n\n{raw_content[:4000]}"
                        )

                        # LLM call via vllm_client using Jetson (collector model)
                        summary, _, _ = await llm.chat(
                            system=system_prompt,
                            user=user_prompt,
                            temperature=0.2,
                            max_tokens=256,
                            priority=Priority.LOW,
                            agent_name="lifecycle_summarizer",
                            ticker=ticker,
                        )

                        if summary and len(summary) > 20:
                            db.execute(
                                f"""
                                UPDATE {table}
                                SET {summary_col} = %s,
                                    {content_col} = '[SUMMARIZED — raw content archived]',
                                    summarized_at = %s
                                WHERE {id_col} = %s
                                """,
                                [summary, datetime.now(timezone.utc), record_id],
                            )
                            total_summarized += 1
                            logger.info(
                                f"[LIFECYCLE] Summarized {table} record {record_id[:8]} for {ticker}"
                            )
                    except Exception as ex:
                        logger.warning(
                            f"[LIFECYCLE] Failed to summarize {table} record {record_id}: {ex}"
                        )

            except Exception as e:
                logger.warning(
                    "[LIFECYCLE] Error querying %s for summarization: %s", table, e
                )

        logger.info(
            "[LIFECYCLE] Off-peak summarization complete: %d records summarized",
            total_summarized,
        )
        return total_summarized

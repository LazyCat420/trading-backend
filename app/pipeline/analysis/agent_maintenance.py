"""
Janitor Agent — Unified Data Cleanup & Self-Healing System.

Responsibilities:
  1. Scraper Repair  — fix broken YouTube handles/transcripts via tool-calling
  2. DB Pruning      — TTL-based cleanup of old audit logs, duplicate news, orphaned embeddings
  3. Semantic Dedup   — flag near-identical news articles via vector similarity
  4. Memory Consolidation — compress contradictory autoresearch lessons (Phase 3)

Called at the end of every pipeline cycle from pipeline_service.py.
"""

import asyncio
import logging
import re

from app.config import settings
from app.db.connection import get_db
from app.services.vllm_client import Priority
from app.tools.executor import run_tool_agent
from app.tools.registry import registry

logger = logging.getLogger(__name__)

MAINTENANCE_SYSTEM_PROMPT = """You are the Maintenance Agent for an autonomous trading bot.
Your job is to fix scraper failures that occur in the data ingestion pipeline.
You have access to tools like `youtube_search_handle`, `youtube_test_channel`, `update_youtube_channel_handle`, and `run_playwright_script`.

When a channel handle fails (e.g., 404 error):
1. Search for the new handle using `youtube_search_handle`.
2. Test it using `youtube_test_channel` to verify it has a videos tab.
3. Update the database using `update_youtube_channel_handle`.

When a transcript fails to extract:
1. FIRST: Check if the video actually has captions. Use `run_playwright_script` to navigate
   to the video page and check for `"hasCaption":true` in the page source.
2. If the video has NO captions (hasCaption is false or missing), respond with:
   "UNFIXABLE: Video has no captions available. Cannot extract transcript."
   Do NOT attempt further scripts. Move on immediately.
3. If the video DOES have captions but extraction failed (anti-bot measures), then write
   a Playwright script to click the transcript button and extract the text.
4. You have a maximum of 2 script attempts. If both fail, report the issue as unfixable.

IMPORTANT: Do NOT keep iterating with increasingly complex scripts. If your first 2 attempts
fail, the video likely has no extractable captions. Report it and move on.

Always document your findings and exactly what you fixed.
"""


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════


async def run_janitor_tasks():
    """Run all janitor tasks in sequence.

    1. Scraper repairs (existing logic — fix broken YouTube handles/transcripts)
    2. DB pruning (TTL-based cleanup)
    3. Semantic dedup (vector similarity flagging)
    4. Memory consolidation (compress autoresearch lessons)
    """
    logger.info("[JANITOR] ═══ Starting Janitor Cycle ═══")

    # 1. Scraper repairs
    await _run_scraper_repairs()

    # 2. DB pruning
    _run_db_pruning()

    # 3. Semantic dedup
    await _run_semantic_dedup()

    # 4. Memory consolidation
    await _run_memory_consolidation()

    logger.info("[JANITOR] ═══ Janitor Cycle Complete ═══")


# ═══════════════════════════════════════════════════════════════════
# TASK 1: SCRAPER REPAIRS (unchanged logic from previous version)
# ═══════════════════════════════════════════════════════════════════


async def _run_scraper_repairs():
    """Scan for scraper issues and assign them to the maintenance agent.

    Deduplicates by (target_name, motivation prefix) and tracks attempt_count
    to auto-close issues that have been tried multiple times without success.
    """
    with get_db() as db:
        # Find pending scraper issues (deduplicated: pick one per unique target+motivation)
        try:
            issues = db.execute(
                "SELECT DISTINCT ON (target_name, LEFT(motivation, 80)) "
                "id, target_name, motivation "
                "FROM pending_evolution_fixes "
                "WHERE target_type = 'scraper_issue' AND status = 'pending' "
                "ORDER BY target_name, LEFT(motivation, 80), created_at ASC "
                "LIMIT 5"
            ).fetchall()
        except Exception as e:
            logger.error(f"[JANITOR] Failed to read pending scraper issues: {e}")
            return

        if not issues:
            logger.info("[JANITOR] No pending scraper issues found.")
            return

        logger.info(
            f"[JANITOR] Found {len(issues)} pending scraper issues. Starting agent..."
        )

        for issue_id, target_name, motivation in issues:
            logger.info(f"[JANITOR] Processing issue {issue_id}: {target_name}")

            # Increment attempt_count
            try:
                db.execute(
                    "UPDATE pending_evolution_fixes SET attempt_count = COALESCE(attempt_count, 0) + 1 WHERE id = %s",
                    (issue_id,),
                )
            except Exception:
                pass

            # Check if we've already tried too many times
            try:
                row = db.execute(
                    "SELECT COALESCE(attempt_count, 0) FROM pending_evolution_fixes WHERE id = %s",
                    (issue_id,),
                ).fetchone()
                attempt_count = row[0] if row else 0
                if attempt_count > 2:
                    logger.warning(
                        f"[JANITOR] Issue {issue_id} exceeded max attempts ({attempt_count}). Flagging for human intervention."
                    )
                    db.execute(
                        "UPDATE pending_evolution_fixes SET status = 'FAILED_REQUIRES_HUMAN', "
                        "proposed_fix = 'Requires manual intervention: exceeded max retry attempts', "
                        "resolved_at = CURRENT_TIMESTAMP WHERE id = %s",
                        (issue_id,),
                    )
                    _close_duplicate_issues(db, target_name, motivation, issue_id)
                    continue
            except Exception as e:
                logger.warning(f"[JANITOR] Failed to check attempt count: {e}")

            try:
                # Run via Prism agent harness first if enabled, falling back to local
                from app.tools.prism_agent_harness import run_prism_agent
                result = await run_prism_agent(
                    system_prompt=MAINTENANCE_SYSTEM_PROMPT,
                    user_prompt=f"Please fix the following scraper issue:\nTarget: {target_name}\nError Context: {motivation}\n\nExecute the necessary tool calls to repair the pipeline.",
                    ticker="SYSTEM",
                    agent_name="maintenance_agent",
                    priority=Priority.NORMAL,
                    tools_override=registry.schemas,
                    timeout_seconds=90,
                )

                final_text = result.get("final_text", "")

                # Check if the agent reported the issue as unfixable
                is_unfixable = any(
                    kw in final_text.upper()
                    for kw in [
                        "UNFIXABLE",
                        "NO CAPTIONS",
                        "CANNOT EXTRACT",
                        "NOT AVAILABLE",
                        "NO TRANSCRIPT",
                        "HAS NO CAPTIONS",
                    ]
                )

                if is_unfixable:
                    db.execute(
                        "UPDATE pending_evolution_fixes SET status = 'error', "
                        "proposed_fix = %s, resolved_at = CURRENT_TIMESTAMP WHERE id = %s",
                        (f"Unfixable: {final_text[:500]}", issue_id),
                    )
                    logger.info(
                        f"[JANITOR] Issue {issue_id} marked unfixable: {final_text[:100]}"
                    )
                else:
                    db.execute(
                        "UPDATE pending_evolution_fixes SET status = 'deployed', "
                        "proposed_fix = %s, resolved_at = CURRENT_TIMESTAMP WHERE id = %s",
                        (final_text, issue_id),
                    )
                    logger.info(f"[JANITOR] Successfully resolved issue {issue_id}")

                # Close any duplicates regardless of outcome
                _close_duplicate_issues(db, target_name, motivation, issue_id)

            except Exception as e:
                logger.error(f"[JANITOR] Agent failed to resolve issue {issue_id}: {e}")
                db.execute(
                    "UPDATE pending_evolution_fixes SET status = 'error', proposed_fix = %s WHERE id = %s",
                    (f"Agent failed with error: {str(e)}", issue_id),
                )


def _close_duplicate_issues(db, target_name: str, motivation: str, keep_id: str):
    """Close duplicate pending issues that match the same target and motivation prefix."""
    try:
        video_match = re.search(r"Video ID ([\w-]+)", motivation)
        channel_match = re.search(r"Channel (\w+)", motivation)

        if video_match:
            pattern = f"%{video_match.group(1)}%"
        elif channel_match:
            pattern = f"%{channel_match.group(1)}%"
        else:
            return

        db.execute(
            "UPDATE pending_evolution_fixes SET status = 'error', "
            "proposed_fix = 'Closed as duplicate', resolved_at = CURRENT_TIMESTAMP "
            "WHERE target_name = %s AND motivation LIKE %s AND status = 'pending' AND id != %s",
            (target_name, pattern, keep_id),
        )
        logger.info(
            f"[JANITOR] Closed duplicate issues for {target_name} matching '{pattern}'"
        )
    except Exception as e:
        logger.warning(f"[JANITOR] Failed to close duplicates: {e}")


# ═══════════════════════════════════════════════════════════════════
# TASK 2: DB PRUNING (TTL-based cleanup)
# ═══════════════════════════════════════════════════════════════════


def _run_db_pruning():
    """Delete old audit logs, duplicate news, and orphaned embeddings."""
    logger.info("[JANITOR] Running DB pruning...")
    with get_db() as db:
        total_deleted = 0

        # 1. Prune old llm_audit_logs
        ttl_days = settings.AUDIT_LOG_TTL_DAYS
        try:
            result = db.execute(
                "DELETE FROM llm_audit_logs WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * %s",
                (ttl_days,),
            )
            count = result.rowcount if hasattr(result, "rowcount") else 0
            if count > 0:
                logger.info(
                    f"[JANITOR] Pruned {count} audit log rows older than {ttl_days} days"
                )
                total_deleted += count
        except Exception as e:
            logger.warning(f"[JANITOR] Audit log pruning failed: {e}")

        # 2. Prune duplicate news articles older than threshold
        dup_ttl = settings.NEWS_DUPLICATE_TTL_DAYS
        try:
            result = db.execute(
                "DELETE FROM news_articles "
                "WHERE quality_status = 'duplicate' "
                "AND collected_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * %s",
                (dup_ttl,),
            )
            count = result.rowcount if hasattr(result, "rowcount") else 0
            if count > 0:
                logger.info(
                    f"[JANITOR] Pruned {count} duplicate news articles older than {dup_ttl} days"
                )
                total_deleted += count
        except Exception as e:
            logger.warning(f"[JANITOR] Duplicate news pruning failed: {e}")

        # 3. Prune orphaned embeddings (source_id no longer exists in parent table)
        try:
            result = db.execute(
                "DELETE FROM embeddings WHERE source_table = 'news_articles' "
                "AND source_id NOT IN (SELECT id FROM news_articles)"
            )
            count = result.rowcount if hasattr(result, "rowcount") else 0
            if count > 0:
                logger.info(f"[JANITOR] Pruned {count} orphaned news embeddings")
                total_deleted += count
        except Exception as e:
            logger.warning(f"[JANITOR] Orphaned embedding pruning failed: {e}")

        if total_deleted > 0:
            logger.info(
                f"[JANITOR] DB pruning complete: {total_deleted} total rows removed"
            )
        else:
            logger.info("[JANITOR] DB pruning: nothing to prune")


# ═══════════════════════════════════════════════════════════════════
# TASK 3: SEMANTIC DEDUP (vector similarity flagging)
# ═══════════════════════════════════════════════════════════════════


async def _run_semantic_dedup():
    """Flag near-identical news articles via vector similarity."""
    try:
        from app.pipeline.data.utility_worker import task_deduplicate_news

        def _noop(*_args, **_kwargs):
            pass

        await task_deduplicate_news(_noop)
        logger.info("[JANITOR] Semantic dedup pass complete")
    except Exception as e:
        logger.warning(f"[JANITOR] Semantic dedup failed (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════════════
# TASK 4: MEMORY CONSOLIDATION (compress autoresearch lessons)
# ═══════════════════════════════════════════════════════════════════


async def _run_memory_consolidation():
    """Consolidate autoresearch lessons if they exceed the threshold."""
    threshold = settings.LESSON_CONSOLIDATION_THRESHOLD
    try:
        with get_db() as db:
            row = db.execute("SELECT COUNT(*) FROM evolution_lessons").fetchone()
            count = row[0] if row else 0

            if count < threshold:
                logger.info(
                    f"[JANITOR] Memory consolidation skipped ({count}/{threshold} lessons)"
                )
                return

            logger.info(
                f"[JANITOR] {count} lessons exceed threshold ({threshold}). Consolidating..."
            )
            from app.cognition.memory_consolidation import consolidate_lessons

            result = await consolidate_lessons(max_lessons=count)
            logger.info(f"[JANITOR] Memory consolidation result: {result}")
    except Exception as e:
        logger.warning(f"[JANITOR] Memory consolidation failed (non-fatal): {e}")


if __name__ == "__main__":
    asyncio.run(run_janitor_tasks())

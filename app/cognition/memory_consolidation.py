"""
Episodic Memory Consolidation — Compresses autoresearch lessons.

Prevents contradictory lessons from accumulating over time by:
  1. Fetching all evolution_lessons from the DB
  2. Grouping by semantic similarity (simple text clustering)
  3. Sending each cluster to the LLM to produce ONE unified rule
  4. Archiving raw lessons and storing consolidated versions
"""

import logging
from datetime import datetime, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)


async def consolidate_lessons(max_lessons: int = 100) -> dict:
    """Consolidate evolution lessons by resolving contradictions.

    Returns:
        Dict with stats: { total_before, clusters, consolidated, archived }
    """
    with get_db() as db:
        # 1. Fetch all lessons ordered by timestamp (newest first for tiebreaking)
        rows = db.execute(
            "SELECT id, session_id, round, score, status, lesson_text, timestamp "
            "FROM evolution_lessons ORDER BY timestamp DESC"
        ).fetchall()

        if not rows:
            return {"total_before": 0, "clusters": 0, "consolidated": 0, "archived": 0}

        total_before = len(rows)
        logger.info("[CONSOLIDATION] Starting with %d lessons", total_before)

        # 2. Group lessons into clusters by simple keyword overlap
        # (We avoid embedding calls here to keep it lightweight)
        clusters = _cluster_lessons(rows)
        logger.info(
            "[CONSOLIDATION] Formed %d clusters from %d lessons",
            len(clusters),
            total_before,
        )

        # 3. For each cluster with 2+ lessons, ask LLM to consolidate
        consolidated_count = 0
        archived_count = 0

        for cluster in clusters:
            if len(cluster) < 2:
                # Single lessons don't need consolidation
                continue

            cluster_texts = [r[5] for r in cluster]  # lesson_text field

            # Consolidate via LLM
            unified_text = await _llm_consolidate(cluster_texts)
            if not unified_text:
                continue

            # Archive old lessons
            for row in cluster:
                try:
                    db.execute(
                        "INSERT INTO evolution_lessons_archive "
                        "(id, session_id, round, score, status, lesson_text, timestamp, archived_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)",
                        [row[0], row[1], row[2], row[3], row[4], row[5], row[6]],
                    )
                    db.execute("DELETE FROM evolution_lessons WHERE id = %s", [row[0]])
                    archived_count += 1
                except Exception as e:
                    logger.debug("[CONSOLIDATION] Archive failed for %s: %s", row[0], e)

            # Also clean up the archived lesson's embedding
            for row in cluster:
                try:
                    db.execute(
                        "DELETE FROM embeddings WHERE source_table = 'evolution_lessons' AND source_id = %s",
                        [row[0]],
                    )
                except Exception:
                    pass

            # Store the consolidated lesson
            try:
                from app.cognition.lesson_store import add_lesson

                add_lesson(
                    text=unified_text,
                    metadata={
                        "session_id": f"consolidated_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                        "round": 0,
                        "score": 0,
                        "status": "consolidated",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                consolidated_count += 1
            except Exception as e:
                logger.warning(
                    "[CONSOLIDATION] Failed to store consolidated lesson: %s", e
                )

        result = {
            "total_before": total_before,
            "clusters": len(clusters),
            "consolidated": consolidated_count,
            "archived": archived_count,
        }
        logger.info("[CONSOLIDATION] Complete: %s", result)
        return result


def _cluster_lessons(rows: list, similarity_threshold: float = 0.85) -> list[list]:
    """Group lessons into clusters using pgvector cosine similarity.

    Uses 1 - (e1.embedding <=> e2.embedding) to measure semantic overlap.
    """
    with get_db() as db:
        assigned = [False] * len(rows)
        clusters = []

        for i in range(len(rows)):
            if assigned[i]:
                continue

            row_i = rows[i]
            cluster = [row_i]
            assigned[i] = True

            row_id = row_i[0]

            # Find similar unassigned lessons using pgvector
            try:
                similar = db.execute(
                    "SELECT e2.source_id "
                    "FROM embeddings e1 "
                    "JOIN embeddings e2 ON e2.source_table = 'evolution_lessons' "
                    "WHERE e1.source_table = 'evolution_lessons' "
                    "AND e1.source_id = %s "
                    "AND e2.source_id != %s "
                    "AND 1 - (e1.embedding <=> e2.embedding) >= %s",
                    (row_id, row_id, similarity_threshold),
                ).fetchall()

                similar_ids = {s[0] for s in similar}

                for j in range(i + 1, len(rows)):
                    if assigned[j]:
                        continue
                    other_id = rows[j][0]
                    if other_id in similar_ids:
                        cluster.append(rows[j])
                        assigned[j] = True
            except Exception as e:
                logger.warning(
                    "[CONSOLIDATION] Vector similarity check failed for %s: %s",
                    row_id,
                    e,
                )
                # Fallback to no grouping for this item if query fails

            clusters.append(cluster)

        return clusters


async def _llm_consolidate(lesson_texts: list[str]) -> str | None:
    """Use LLM to consolidate multiple lessons into one unified rule."""
    try:
        from app.services.vllm_client import Priority
        from app.services.prism_agent_caller import call_prism_agent

        lessons_block = "\n".join(f"- {t}" for t in lesson_texts[:10])

        system_prompt = (
            "You are a trading strategy auditor. Below are multiple lessons learned "
            "from past trading cycles. Some may contradict each other.\n"
            "Your job is to:\n"
            "1. Identify contradictions\n"
            "2. Resolve them using the MOST RECENT lesson as the tiebreaker "
            "(lessons are listed newest-first)\n"
            "3. Output a single, clear, actionable rule\n\n"
            "Output ONE consolidated rule (max 120 characters). No explanation needed."
        )

        response_text, tokens, ms = await call_prism_agent(
            agent_id="CUSTOM_MEMORY_CONSOLIDATION_AGENT",
            user_message=f"Lessons to consolidate:\n{lessons_block}",
            fallback_system_prompt=system_prompt,
            fallback_agent_name="memory_consolidation",
            temperature=0.1,
            max_tokens=100,
            priority=Priority.LOW,
            ticker="_system",
        )

        unified = response_text.strip() if response_text else ""
        if len(unified) > 5:
            logger.info(
                "[CONSOLIDATION] Unified %d lessons → '%s'",
                len(lesson_texts),
                unified[:80],
            )
            return unified[:120]
        return None

    except Exception as e:
        logger.warning("[CONSOLIDATION] LLM consolidation failed: %s", e)
        return None

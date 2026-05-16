"""
Meta-Agent Runner — pipeline wrapper for prompt generation.

Gathers context from strategy_performance and debate results,
feeds it to the meta_agent, and persists new lenses to the
generated_agent_prompts table.

Architecture:
    - This file handles ALL data fetching (Rule 7: agents don't fetch)
    - Passes data to meta_agent.generate_prompt() as parameters
    - Handles dedup, cap enforcement, and persistence

Usage:
    from app.pipeline.analysis.meta_agent_runner import run_meta_agent

    # Run periodically (every META_AGENT_INTERVAL_HOURS):
    new_prompt = await run_meta_agent(cycle_id="cycle-abc")
"""

import logging
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def _build_winning_patterns_context() -> str:
    """Query strategy_performance for top winning patterns."""
    with get_db() as db:
        try:
            rows = db.execute(
                """
                SELECT
                    agent_prompt_hash,
                    COUNT(*) as trades,
                    AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(return_pct) as avg_return,
                    STRING_AGG(ticker, ', ') as tickers
                FROM strategy_performance
                WHERE resolved_at IS NOT NULL AND win = TRUE
                GROUP BY agent_prompt_hash
                ORDER BY win_rate DESC, avg_return DESC
                LIMIT 5
                """
            ).fetchall()

            if not rows:
                return "No winning strategy data available yet."

            lines = []
            for row in rows:
                # Try to get the prompt name
                name_row = db.execute(
                    "SELECT name, lens_type FROM generated_agent_prompts WHERE prompt_hash = %s",
                    [row[0]],
                ).fetchone()

                name = name_row[0] if name_row else f"prompt_{row[0][:8]}"
                lens_type = name_row[1] if name_row else "unknown"

                lines.append(
                    f"- **{name}** ({lens_type}): {row[1]} trades, "
                    f"{row[2] * 100:.0f}% win rate, avg return {row[3]:.1f}%, "
                    f"tickers: {row[4][:100]}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.debug("[META_RUNNER] Winning patterns query failed: %s", e)
            return "No winning strategy data available yet."


def _build_losing_patterns_context() -> str:
    """Query strategy_performance for underperforming patterns."""
    with get_db() as db:
        try:
            rows = db.execute(
                """
                SELECT
                    agent_prompt_hash,
                    COUNT(*) as trades,
                    AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(return_pct) as avg_return
                FROM strategy_performance
                WHERE resolved_at IS NOT NULL
                GROUP BY agent_prompt_hash
                HAVING COUNT(*) >= 3 AND AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) < 0.4
                ORDER BY win_rate ASC
                LIMIT 5
                """
            ).fetchall()

            if not rows:
                return "No underperforming patterns identified yet."

            lines = []
            for row in rows:
                name_row = db.execute(
                    "SELECT name FROM generated_agent_prompts WHERE prompt_hash = %s",
                    [row[0]],
                ).fetchone()

                name = name_row[0] if name_row else f"prompt_{row[0][:8]}"
                lines.append(
                    f"- **{name}**: {row[1]} trades, {row[2] * 100:.0f}% win rate, "
                    f"avg return {row[3]:.1f}%"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.debug("[META_RUNNER] Losing patterns query failed: %s", e)
            return "No underperforming patterns identified yet."


def _build_debate_insights_context() -> str:
    """Get recent debate outcomes for meta-agent context."""
    with get_db() as db:
        try:
            rows = db.execute(
                """
                SELECT ticker, result_json
                FROM analysis_results
                WHERE agent_name LIKE 'hybrid_%%'
                  AND created_at > NOW() - INTERVAL '7 days'S
                ORDER BY created_at DESC
                LIMIT 10
                """
            ).fetchall()

            if not rows:
                return "No recent debate data available."

            lines = []
            for row in rows:
                ticker = row[0]
                try:
                    import json

                    result = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                    action = result.get("action", "%s")
                    confidence = result.get("confidence", 0)
                    config = result.get("config_used", "%s")
                    lines.append(
                        f"- {ticker}: {action} @ {confidence}% (config: {config})"
                    )
                except Exception:
                    lines.append(f"- {ticker}: [parse error]")

            return "\n".join(lines[:10])

        except Exception as e:
            logger.debug("[META_RUNNER] Debate insights query failed: %s", e)
            return "No recent debate data available."


def _get_existing_lens_names() -> str:
    """Get comma-separated list of existing active lens names."""
    from app.config.config_lenses import get_active_lenses

    lenses = get_active_lenses()
    return ", ".join(lens["name"] for lens in lenses)


async def run_meta_agent(
    cycle_id: str = "",
    bot_id: str = "",
    emit=None,
) -> dict | None:
    """Run the meta-agent to generate a new analytical lens.

    Steps:
        1. Check if meta-agent is enabled and under prompt cap
        2. Build context from strategy_performance and debate results
        3. Call meta_agent.generate_prompt()
        4. Check for duplicate prompts (by hash)
        5. Persist to generated_agent_prompts table

    Returns:
        Dict with the new prompt details, or None if skipped/failed.
    """
    if not settings.META_AGENT_ENABLED:
        logger.debug("[META_RUNNER] Disabled by config (META_AGENT_ENABLED=False)")
        return None

    with get_db() as db:
        # Check active prompt cap
        try:
            count_row = db.execute(
                "SELECT COUNT(*) FROM generated_agent_prompts WHERE active = TRUE"
            ).fetchone()
            active_count = count_row[0] if count_row else 0

            if active_count >= settings.MAX_ACTIVE_GENERATED_PROMPTS:
                logger.info(
                    "[META_RUNNER] At prompt cap (%d/%d), skipping generation",
                    active_count,
                    settings.MAX_ACTIVE_GENERATED_PROMPTS,
                )
                return None
        except Exception:
            pass  # Table might not exist yet

        # Check if enough time has passed since last generation
        try:
            last_row = db.execute(
                """
                SELECT MAX(created_at) FROM generated_agent_prompts
                """
            ).fetchone()

            if last_row and last_row[0]:
                from datetime import timedelta

                last_created = last_row[0]
                if isinstance(last_created, str):
                    last_created = datetime.fromisoformat(last_created)
                if last_created.tzinfo is None:
                    last_created = last_created.replace(tzinfo=timezone.utc)

                interval = timedelta(hours=settings.META_AGENT_INTERVAL_HOURS)
                if datetime.now(timezone.utc) - last_created < interval:
                    logger.debug(
                        "[META_RUNNER] Too soon since last generation, skipping"
                    )
                    return None
        except Exception:
            pass

        if emit:
            emit(
                "analyzing",
                "meta_agent_start",
                "Meta-Agent: generating new analytical lens",
                status="running",
            )

        # Build context for the meta-agent (this is where data fetching happens)
        winning_context = _build_winning_patterns_context()
        losing_context = _build_losing_patterns_context()
        debate_context = _build_debate_insights_context()
        existing_lenses = _get_existing_lens_names()

        # Call the meta-agent (pure LLM, no data fetching)
        from app.agents.meta_agent import generate_prompt

        result = await generate_prompt(
            winning_patterns=winning_context,
            losing_patterns=losing_context,
            debate_insights=debate_context,
            existing_lenses=existing_lenses,
            cycle_id=cycle_id,
            bot_id=bot_id,
        )

        if not result or not result.get("system_prompt"):
            logger.warning("[META_RUNNER] Meta-agent returned empty prompt")
            if emit:
                emit(
                    "analyzing",
                    "meta_agent_fail",
                    "Meta-Agent: failed to generate lens",
                    status="error",
                )
            return None

        # Dedup check by prompt hash
        from app.utils.text_utils import hash_prompt

        prompt_hash = hash_prompt(result["system_prompt"])

        try:
            existing = db.execute(
                "SELECT id FROM generated_agent_prompts WHERE prompt_hash = %s",
                [prompt_hash],
            ).fetchone()

            if existing:
                logger.info(
                    "[META_RUNNER] Duplicate prompt detected (hash=%s), skipping",
                    prompt_hash[:8],
                )
                return None
        except Exception:
            pass

        # Persist the new prompt
        prompt_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        try:
            db.execute(
                """
                INSERT INTO generated_agent_prompts
                (id, name, lens_type, system_prompt, prompt_hash,
                 created_by, created_at)
                VALUES (%s, %s, %s, %s, %s, 'meta_agent', %s)
                """,
                [
                    prompt_id,
                    result["name"],
                    result.get("lens_type", "custom"),
                    result["system_prompt"],
                    prompt_hash,
                    now,
                ],
            )

            logger.info(
                "[META_RUNNER] ✅ New lens created: '%s' (%s) — hash=%s, tokens=%d",
                result["name"],
                result.get("lens_type", "custom"),
                prompt_hash[:8],
                result.get("tokens_used", 0),
            )

            if emit:
                emit(
                    "analyzing",
                    "meta_agent_done",
                    f"Meta-Agent: created lens '{result['name']}' ({result.get('lens_type', 'custom')})",
                    status="ok",
                    data={
                        "name": result["name"],
                        "lens_type": result.get("lens_type", "custom"),
                        "prompt_hash": prompt_hash,
                    },
                )

            return {
                "id": prompt_id,
                "name": result["name"],
                "lens_type": result.get("lens_type", "custom"),
                "prompt_hash": prompt_hash,
                "rationale": result.get("rationale", ""),
                "tokens_used": result.get("tokens_used", 0),
            }

        except Exception as e:
            logger.error("[META_RUNNER] Failed to persist prompt: %s", e)
            return None

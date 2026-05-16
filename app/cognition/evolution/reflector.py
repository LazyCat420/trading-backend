"""
Reflector Engine — Autoresearch feedback loop.
Evaluates agent trajectories to extract behavioral lessons.
"""

import logging
import uuid
from typing import Any

from app.services.vllm_client import llm, Priority
from app.db.connection import get_db

logger = logging.getLogger(__name__)


async def reflect_on_trajectory(
    agent_name: str,
    ticker: str,
    cycle_id: str,
    loops_used: int,
    yielded: bool,
    chat_history: list[dict[str, Any]],
    success: bool = False,
    spotlight_tools: list[str] | None = None,
):
    """
    Analyze an agent trajectory to extract a behavioral lesson.
    If success=True, extracts a positive heuristic (what worked well).
    If success=False, extracts a corrective rule (what to avoid).
    Stores the lesson in agent_experiences for future context injection.
    """
    try:
        if not chat_history:
            return

        # Prepare trajectory string
        trajectory = []
        for msg in chat_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if tool_calls:
                tc_names = [tc.get("function", {}).get("name") for tc in tool_calls]
                trajectory.append(f"[{role}] Requested tools: {tc_names}")
            elif content:
                # Truncate content to avoid huge prompts
                trunc_content = content[:500] + "..." if len(content) > 500 else content
                trajectory.append(f"[{role}] {trunc_content}")

        trajectory_text = "\n".join(trajectory)

        # Build prompt depending on success/failure
        if success:
            spotlight_text = f"\nThe agent was nudged to try these tools: {spotlight_tools}\n" if spotlight_tools else ""
            system_prompt = (
                "You are an AI optimization expert for a multi-agent trading system.\n"
                f"The agent '{agent_name}' just successfully finished a task on ticker '{ticker}'.\n"
                f"Loops used: {loops_used}.{spotlight_text}\n\n"
                "This agent performed well. Review its trajectory below. "
                "Identify the most effective tool or decision it made. "
                "Then, extract a single, concise positive behavioral rule (less than 20 words) that the agent should "
                "continue following in the future to replicate this success. Respond ONLY with the rule text."
            )
        else:
            system_prompt = (
                "You are an AI optimization expert for a multi-agent trading system.\n"
                f"The agent '{agent_name}' just finished a task on ticker '{ticker}'.\n"
                f"Loops used: {loops_used} (Limit: 5). Yielded (Failed): {yielded}\n\n"
                "This agent was inefficient or failed. Review its trajectory below. "
                "Identify the primary mistake (e.g. repetitive tool calls, ignoring context, hallucination). "
                "Then, extract a single, concise behavioral rule (less than 20 words) that the agent must follow "
                "in the future to avoid this mistake. Respond ONLY with the rule text."
            )

        user_prompt = f"### TRAJECTORY\n{trajectory_text}\n\n### EXTRACTED LESSON:"

        result = await llm.chat(
            system=system_prompt,
            user=user_prompt,
            priority=Priority.LOW,  # Run in background without blocking core trading
            agent_name="reflector",
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id="system",
        )

        lesson = result.get("text", "").strip()
        if not lesson:
            logger.warning("[Reflector] LLM returned empty lesson.")
            return

        logger.info(f"[Reflector] Extracted lesson for {agent_name}: {lesson}")

        # Save to database
        with get_db() as db:
            db.execute(
                """
                INSERT INTO agent_experiences (id, agent_name, task_context, lesson_learned)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    str(uuid.uuid4()),
                    agent_name,
                    f"Ticker: {ticker}, Cycle: {cycle_id}, Loops: {loops_used}",
                    lesson,
                ],
            )

    except Exception as e:
        logger.error(f"[Reflector] Failed to reflect on trajectory: {e}", exc_info=True)


def get_agent_lessons(agent_name: str, limit: int = 3) -> list[str]:
    """Retrieve the top active lessons for an agent."""
    try:
        with get_db() as db:
            # We fetch lessons with success_score > 0.5, sorted by score and recency
            cur = db.execute(
                """
                SELECT lesson_learned FROM agent_experiences 
                WHERE agent_name = %s AND success_score >= 0.5
                ORDER BY success_score DESC, created_at DESC
                LIMIT %s
                """,
                [agent_name, limit],
            )
            rows = cur.fetchall()

            if rows:
                # Update last_applied
                db.execute(
                    "UPDATE agent_experiences SET last_applied = CURRENT_TIMESTAMP WHERE agent_name = %s AND success_score >= 0.5",
                    [agent_name],
                )

            return [row[0] for row in rows]
    except Exception as e:
        logger.error(f"[Reflector] Failed to fetch lessons: {e}")
        return []


def adjust_lesson_score(agent_name: str, success: bool):
    """
    Adjust the score of recent lessons applied by this agent based on success/failure.
    """
    try:
        with get_db() as db:
            # Only affect lessons applied in the last 1 hour
            if success:
                db.execute(
                    "UPDATE agent_experiences SET success_score = success_score + 0.1 "
                    "WHERE agent_name = %s AND last_applied > CURRENT_TIMESTAMP - INTERVAL '1 hour' AND success_score < 5.0",
                    [agent_name],
                )
            else:
                db.execute(
                    "UPDATE agent_experiences SET success_score = success_score - 0.2 "
                    "WHERE agent_name = %s AND last_applied > CURRENT_TIMESTAMP - INTERVAL '1 hour'",
                    [agent_name],
                )
    except Exception as e:
        logger.error(f"[Reflector] Failed to adjust lesson score: {e}")


def get_spotlight_tools(limit: int = 5) -> list[str]:
    """
    Fetch the least used tools from the database over the last 7 days to encourage exploration.
    We exclude tools that have never been used if they aren't in the DB yet, but we can query 
    the tool_usage_stats to find the lowest non-zero counts.
    """
    try:
        from app.tools.registry import registry
        
        with get_db() as db:
            cur = db.execute(
                """
                SELECT tool_name, COUNT(*) as usage_count 
                FROM tool_usage_stats 
                WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
                GROUP BY tool_name
                ORDER BY usage_count ASC
                LIMIT %s
                """,
                [limit * 2],  # Fetch more to filter out destructive or internal tools
            )
            rows = cur.fetchall()
            
            # Find registered tools with low usage that are safe to spotlight
            spotlight = []
            for row in rows:
                tool_name = row[0]
                meta = registry.get_tool_meta(tool_name)
                # Only spotlight safe tools (read-only or write), avoid destructive
                if meta and meta.permission.value != "destructive":
                    spotlight.append(tool_name)
                    if len(spotlight) >= limit:
                        break
                        
            # If we didn't find enough, maybe the DB is empty or has too few tools.
            # Let's add some from the registry that have no usage recorded at all.
            if len(spotlight) < limit:
                all_tools = [s["function"]["name"] for s in registry.get_primary_schemas()]
                db_tools = {row[0] for row in rows}
                for t in all_tools:
                    if t not in db_tools and t not in spotlight:
                        meta = registry.get_tool_meta(t)
                        if meta and meta.permission.value != "destructive":
                            spotlight.append(t)
                            if len(spotlight) >= limit:
                                break
                                
            return spotlight
    except Exception as e:
        logger.warning(f"[Reflector] Failed to get spotlight tools: {e}")
        return []

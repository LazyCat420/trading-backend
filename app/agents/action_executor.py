"""
Action Executor — Runs an isolated, fresh-context agent loop for tool execution.

This module implements the "Action Agent" half of the Brain-Action split.
It spins up a short-lived, disposable agent session with ONLY the tools
selected by the Tool Selector, executes them, and returns a dense summary
back to the caller (the "Brain Agent").

Key design properties:
  - Fresh context: no prior history pollution
  - Minimal tool schemas: only the subset selected by the Tool Selector
  - Disposable: session is discarded after execution
  - Self-healing: retries within its own isolated thread
"""

import logging
from typing import Any

from app.services.vllm_client import llm, Priority
from app.agents.agent_loop import run_agent_loop
from app.agents.agent_budget import AgentBudget

logger = logging.getLogger(__name__)

# System prompt for the Action Executor — kept minimal to preserve context space
ACTION_EXECUTOR_SYSTEM = (
    "You are a data retrieval and execution agent for a financial trading system. "
    "Execute the requested tools to gather the data described in the task. "
    "After gathering data, provide a dense, structured summary of your findings. "
    "Do NOT provide trading advice — only report the raw data and facts retrieved."
)


async def run_isolated_action_agent(
    task_instruction: str,
    tool_schemas: list[dict],
    ticker: str,
    parent_agent_name: str = "unknown",
    cycle_id: str = "",
    bot_id: str = "",
    priority: Priority = Priority.NORMAL,
    max_turns: int = 5,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Run a fresh, isolated agent loop with a specific tool subset.

    This is the core "Action Agent" function. It creates a brand-new
    conversation session with zero prior history, executes the requested
    tools, and returns the results.

    Args:
        task_instruction: What the agent should do (from the Brain/Selector).
        tool_schemas: The pre-selected subset of tool schemas.
        ticker: Current ticker context.
        parent_agent_name: The Brain Agent that spawned this executor (for tracing).
        cycle_id: Cycle ID for tracing and DB linkage.
        bot_id: Bot ID for tracing.
        priority: LLM queue priority.
        max_turns: Maximum tool-calling turns before forced termination.
        model_override: Optional model override for the action agent.

    Returns:
        Dict with keys: final_text, token_usage, execution_ms, loops_used, yielded, stop_reason
    """
    action_agent_name = f"{parent_agent_name}_action"

    logger.info(
        "[ActionExecutor] Spawning isolated action agent '%s' with %d tools for %s",
        action_agent_name,
        len(tool_schemas),
        ticker,
    )

    budget = AgentBudget(max_turns=max_turns)

    try:
        result = await run_agent_loop(
            system_prompt=ACTION_EXECUTOR_SYSTEM,
            user_prompt=task_instruction,
            ticker=ticker,
            agent_name=action_agent_name,
            cycle_id=cycle_id,
            bot_id=bot_id,
            budget=budget,
            priority=priority,
            tools_override=tool_schemas,
            model_override=model_override,
            require_json_schema=False,  # Action agents return free-text summaries
        )

        logger.info(
            "[ActionExecutor] Agent '%s' completed: %d turns, %d tokens, stop=%s",
            action_agent_name,
            result.get("loops_used", 0),
            result.get("token_usage", 0),
            result.get("stop_reason", "unknown"),
        )

        return result

    except Exception as e:
        logger.error(
            "[ActionExecutor] Isolated agent '%s' failed: %s",
            action_agent_name,
            e,
        )
        return {
            "final_text": f"Action executor failed: {str(e)}",
            "token_usage": 0,
            "execution_ms": 0,
            "loops_used": 0,
            "yielded": False,
            "stop_reason": "error",
        }

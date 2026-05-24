import logging
from typing import Any
from app.services.vllm_client import llm, Priority
from app.tools.registry import registry
from app.config import settings

logger = logging.getLogger(__name__)


class AgentYielded(Exception):
    """Raised when an agent hits max_loops while still wanting to call tools.

    Carries the partial state so the caller can decide how to handle it:
      - Summarize and return partial results (subagent pattern)
      - Resume later with the saved chat_history (continuation pattern)
    """

    def __init__(self, partial_result: dict):
        self.partial_result = partial_result
        super().__init__(
            f"Agent yielded after {partial_result.get('loops_used', '?')} loops"
        )


async def run_tool_agent(
    system_prompt: str,
    user_prompt: str,
    ticker: str,
    max_loops: int = 5,
    agent_name: str = "tool_analyst",
    cycle_id: str = "",
    bot_id: str = "",
    priority: Priority = Priority.NORMAL,
    previous_messages: list = None,
    model_override: str | None = None,
    tools_override: list[dict] | None = None,
    yield_on_limit: bool = False,
    bypass_prism: bool = False,
) -> dict[str, Any]:
    """
    Run an LLM agent with tools. Automatically loops back to the LLM
    if it requests tool executions, up to max_loops.

    Args:
        tools_override: If provided, only these tool schemas are sent to the LLM.
                        Use registry.get_schemas_by_names() to build a whitelist.
        yield_on_limit: If True, raise AgentYielded instead of silently returning
                        when max_loops is exhausted while the LLM still wants tools.
                        The caller can then decide to summarize, retry, or resume.
    """
    if not bypass_prism and settings.PRISM_ENABLED and settings.PRISM_AGENT_ROUTING:
        try:
            prism_healthy = await llm.prism_client.check_health()
            if prism_healthy:
                from app.tools.prism_agent_harness import run_prism_agent
                logger.info("[Executor] Routing %s agentic loop to Prism /agent", agent_name)
                return await run_prism_agent(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    ticker=ticker,
                    agent_name=agent_name,
                    cycle_id=cycle_id,
                    bot_id=bot_id,
                    priority=priority,
                    tools_override=tools_override,
                    temperature=0.3,
                )
        except Exception as pe:
            logger.error("[Executor] Prism routing failed for %s, falling back to local: %s", agent_name, pe)

    if previous_messages:
        messages = previous_messages.copy()
        # Ensure we don't duplicate user prompts if they're just appending
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # Use restricted tool set if provided, otherwise full registry
    active_tools = tools_override if tools_override is not None else registry.schemas

    total_tokens_used = 0
    total_time_ms = 0
    final_content = ""
    hit_limit_with_pending_tools = False

    for i in range(max_loops):
        try:
            result = await llm.chat_with_tools(
                messages=messages,
                tools=active_tools,
                agent_name=agent_name,
                ticker=ticker,
                cycle_id=cycle_id,
                bot_id=bot_id,
                priority=priority,
                max_tokens=2048,
                model_override=model_override,
            )
        except Exception as e:
            logger.error(f"[ToolExecutor] chat_with_tools failed: {e}")
            final_content = f"Error during agent execution: {str(e)}"
            break

        content = result.get("text", "")
        tool_calls = result.get("tool_calls")
        total_tokens_used += result.get("total_tokens", 0)
        total_time_ms += result.get("elapsed_ms", 0)
        final_content = content

        # Append assistant message
        assistant_msg = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            logger.info(
                f"[ToolExecutor] Agent '{agent_name}' finished successfully after {i + 1} turns."
            )
            break

        # Log tool calls
        for tc in tool_calls:
            logger.info(
                f"[ToolExecutor] Turn {i + 1}: Agent requested tool -> {tc.get('function', {}).get('name')}"
            )

        # Execute tool calls
        for tc in tool_calls:
            tool_res = await registry.execute_tool_call(
                tc, agent_name=agent_name, ticker=ticker, cycle_id=cycle_id
            )
            messages.append(tool_res)
    else:
        # for/else: the loop completed without break → max_loops exhausted
        # Check if the last turn had tool_calls (meaning the agent wanted to keep going)
        if tool_calls:
            hit_limit_with_pending_tools = True
            logger.warning(
                f"[ToolExecutor] Agent '{agent_name}' hit max_loops={max_loops} "
                f"with pending tool calls — {'yielding' if yield_on_limit else 'returning partial'}."
            )

    base_result = {
        "final_text": final_content,
        "token_usage": total_tokens_used,
        "execution_ms": total_time_ms,
        "chat_history": messages,
        "loops_used": min(i + 1, max_loops) if "i" in dir() else 0,
        "yielded": hit_limit_with_pending_tools,
    }

    if hit_limit_with_pending_tools and yield_on_limit:
        raise AgentYielded(base_result)

    return base_result

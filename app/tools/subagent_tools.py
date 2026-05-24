import logging
import json
from app.tools.registry import registry
from app.tools.executor import run_tool_agent, AgentYielded
from app.services.vllm_client import llm, Priority
from app.services.prism_agent_caller import call_prism_agent

logger = logging.getLogger(__name__)

SUBAGENT_SYSTEM_PROMPT = """You are an expert Research Subagent. 
Your goal is to gather specific information requested by the primary Analyst Agent.
You have access to web search, url scraping, and specialized models (Hermes).
CRITICAL: To prevent context overflow when analyzing huge datasets or scraped pages, you MUST prioritize using the `grep_search_text` and `paginated_read` tools rather than returning or reading entire raw documents.
1. Break down the task.
2. Use tools to gather data (use grep/pagination for large texts).
3. Formulate a concise, highly factual summary.
Output your final answer as JSON in the following format:
{
  "status": "success|failed",
  "summary": "Your detailed findings here. Include numbers, facts, and sources.",
  "confidence": 0-100
}
"""

YIELD_SUMMARY_PROMPT = """You were a research subagent that ran out of execution steps before finishing.
Below is your conversation so far. Summarize ALL information you have gathered into a final JSON response.
Even partial data is valuable — report what you found.

Output your answer as JSON:
{
  "status": "partial",
  "summary": "Everything you discovered so far, with numbers and sources.",
  "confidence": 0-100,
  "note": "What was left unfinished"
}
"""


@registry.register(
    name="spawn_research_subagent",
    description=(
        "Spawn a subagent to perform complex research tasks involving multiple tool calls "
        "(e.g., search web and read articles). Returns a detailed summary. "
        "You can optionally specify which tools the subagent should have access to "
        "via the 'enabled_tools' parameter. If omitted, the subagent gets all available tools."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "A detailed explanation of what the subagent needs to research or find out.",
            },
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol relevant to the task.",
            },
            "enabled_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of specific tool names the subagent should have access to. "
                    "Example: ['search_web', 'scrape_url', 'get_market_data']. "
                    "If omitted, the subagent receives all available tools (minus spawn_research_subagent)."
                ),
            },
        },
        "required": ["task_description", "ticker"],
    },
)
async def spawn_research_subagent(
    task_description: str, ticker: str, enabled_tools: list[str] | None = None
) -> str:
    """
    Spawns a subagent via run_tool_agent with yield-on-limit enabled.
    If the subagent hits max_loops, it gracefully summarizes partial work
    instead of returning incomplete/empty results.

    Args:
        task_description: What the subagent should research.
        ticker: Stock ticker context.
        enabled_tools: Optional whitelist of tool names. If provided, the
                       subagent ONLY gets these tools (intersected with the
                       registry). If omitted, gets all tools minus itself.
    """
    logger.info(
        "[SubagentHarness] Spawning research subagent for %s. Task: %s... | tools: %s",
        ticker,
        task_description[:50],
        enabled_tools or "ALL",
    )

    try:
        # ── Phase 4: Dynamic Tool Provisioning ──
        # If the parent agent specified a whitelist, use it.
        # Otherwise, give all tools minus the spawn tool itself.
        if enabled_tools:
            # Intersect requested names with actual registered tools
            active_schemas = registry.get_schemas_by_names(enabled_tools)
            # Safety: remove spawn tool even if explicitly requested
            active_schemas = [
                s for s in active_schemas
                if s["function"]["name"] != "spawn_research_subagent"
            ]
            if not active_schemas:
                logger.warning(
                    "[SubagentHarness] No valid tools matched enabled_tools=%s. "
                    "Falling back to all tools.",
                    enabled_tools,
                )
                active_schemas = [
                    s for s in registry.schemas
                    if s["function"]["name"] != "spawn_research_subagent"
                ]
        else:
            # Prevent infinite recursion by not passing spawn tool to the subagent
            active_schemas = [
                s for s in registry.schemas
                if s["function"]["name"] != "spawn_research_subagent"
            ]

        logger.info(
            "[SubagentHarness] Subagent for %s provisioned with %d tools: %s",
            ticker,
            len(active_schemas),
            [s["function"]["name"] for s in active_schemas[:10]],
        )

        # ── Phase 6: Onion Layer Routing ──
        # Try Prism-delegated harness first for full observability.
        # Falls back to local executor if Prism is unhealthy.
        try:
            from app.tools.prism_agent_harness import run_prism_agent

            result = await run_prism_agent(
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                user_prompt=f"Research Task: {task_description}\nTicker: {ticker}\nGather the data and provide the final JSON summary.",
                ticker=ticker,
                agent_name="research_subagent",
                priority=Priority.NORMAL,
                tools_override=active_schemas,
                max_tokens=2048,
                temperature=0.3,
            )

            routed_via = result.get("routed_via", "unknown")
            logger.info(
                "[SubagentHarness] Subagent for %s completed via %s",
                ticker,
                routed_via,
            )
        except ImportError:
            # Prism harness not available — use local executor directly
            result = await run_tool_agent(
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                user_prompt=f"Research Task: {task_description}\nTicker: {ticker}\nGather the data and provide the final JSON summary.",
                ticker=ticker,
                max_loops=8,
                agent_name="research_subagent",
                priority=Priority.NORMAL,
                tools_override=active_schemas,
                yield_on_limit=True,
            )

        final_text = result.get("final_text", "")

        return json.dumps(
            {
                "subagent_task": task_description,
                "subagent_result": final_text,
                "tokens_used": result.get("token_usage", 0),
                "execution_ms": result.get("execution_ms", 0),
                "routed_via": result.get("routed_via", "local"),
                "status": "complete",
            }
        )

    except AgentYielded as yielded:
        # The subagent ran out of loops but was still working.
        # Ask the LLM for a final summary of what it gathered so far.
        logger.warning(
            f"[SubagentHarness] Subagent yielded after {yielded.partial_result.get('loops_used')} loops. "
            f"Requesting summary of partial work..."
        )

        try:
            # Re-use the existing conversation history and ask for a summary
            partial_history = yielded.partial_result.get("chat_history", [])

            # Append a new user message asking for the summary
            partial_history.append({"role": "user", "content": YIELD_SUMMARY_PROMPT})

            # One final LLM call — NO tools, just summarize
            summary_response, summary_tokens, summary_ms = await call_prism_agent(
                agent_id="CUSTOM_RESEARCH_SUBAGENT_YIELD_AGENT",
                user_message=YIELD_SUMMARY_PROMPT,
                fallback_system_prompt="You are summarizing your own partial research. Be factual and concise.",
                fallback_agent_name="research_subagent_yield",
                temperature=0.2,
                max_tokens=512,
                priority=Priority.NORMAL,
                ticker=ticker,
            )

            partial_tokens = yielded.partial_result.get("token_usage", 0)
            partial_ms = yielded.partial_result.get("execution_ms", 0)

            return json.dumps(
                {
                    "subagent_task": task_description,
                    "subagent_result": summary_response,
                    "tokens_used": partial_tokens + summary_tokens,
                    "execution_ms": partial_ms + summary_ms,
                    "status": "yielded",
                    "loops_used": yielded.partial_result.get("loops_used", 0),
                    "note": "Subagent hit loop limit; partial results summarized.",
                }
            )
        except Exception as summary_err:
            logger.error(
                f"[SubagentHarness] Failed to summarize yielded work: {summary_err}"
            )
            # Fall back to whatever text we had
            return json.dumps(
                {
                    "subagent_task": task_description,
                    "subagent_result": yielded.partial_result.get("final_text", ""),
                    "tokens_used": yielded.partial_result.get("token_usage", 0),
                    "execution_ms": yielded.partial_result.get("execution_ms", 0),
                    "status": "yielded_raw",
                    "note": f"Summary generation failed: {summary_err}",
                }
            )

    except Exception as e:
        logger.error(f"[SubagentHarness] Subagent failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})

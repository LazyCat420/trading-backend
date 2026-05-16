import logging
import json
from app.tools.registry import registry
from app.tools.executor import run_tool_agent, AgentYielded
from app.services.vllm_client import llm, Priority

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
    description="Spawn a subagent to perform complex research tasks involving multiple tool calls (e.g., search web and read articles). Returns a detailed summary.",
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
        },
        "required": ["task_description", "ticker"],
    },
)
async def spawn_research_subagent(task_description: str, ticker: str) -> str:
    """
    Spawns a subagent via run_tool_agent with yield-on-limit enabled.
    If the subagent hits max_loops, it gracefully summarizes partial work
    instead of returning incomplete/empty results.
    """
    logger.info(
        f"[SubagentHarness] Spawning research subagent for {ticker}. Task: {task_description[:50]}..."
    )

    try:
        # Prevent infinite recursion by not passing `spawn_research_subagent` to the subagent itself
        active_schemas = [
            s
            for s in registry.schemas
            if s["function"]["name"] != "spawn_research_subagent"
        ]

        # With yielding enabled, we can safely allow more loops (8 instead of 3)
        # because if the agent is still working at loop 8, it yields gracefully
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
            summary_response, summary_tokens, summary_ms = await llm.chat(
                system="You are summarizing your own partial research. Be factual and concise.",
                user=YIELD_SUMMARY_PROMPT,
                temperature=0.2,
                max_tokens=512,
                priority=Priority.NORMAL,
                agent_name="research_subagent_yield",
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

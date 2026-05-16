import time
import logging
import json
import re
from app.tools.executor import run_tool_agent
from app.services.vllm_client import Priority

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior hedge fund analyst with access to real-time market data tools.
You are evaluating a ticker. You MUST call tools to gather all necessary context before making your decision.

1. Fetch pricing and fundamentals
2. Retrieve the latest news if relevant
3. Gather technical indicators

Once you have enough context, output your final decision in JSON matching the exact format:
{
  "action": "BUY|SELL|HOLD",
  "confidence": 0-100,
  "rationale": "2-3 sentences explaining your decision"
}
Do NOT output anything other than JSON in your final response.
"""


async def analyze_with_tools(ticker: str, cycle_id: str = "", bot_id: str = "") -> dict:
    logger.info(f"[ToolAnalyst] Starting tool-driven analysis for {ticker}")
    start = time.monotonic()

    result = await run_tool_agent(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Please analyze {ticker} and provide your final BUY/SELL/HOLD decision.",
        ticker=ticker,
        max_loops=5,
        agent_name="tool_analyst",
        cycle_id=cycle_id,
        bot_id=bot_id,
        priority=Priority.NORMAL,
    )

    final_text = result.get("final_text", "")
    action = "HOLD"
    confidence = 0
    rationale = "Failed to parse final decision."

    try:
        match = re.search(r"\{.*\}", final_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            action = parsed.get("action", "HOLD").upper()
            confidence = int(parsed.get("confidence", 0))
            rationale = parsed.get("rationale", final_text[:200])
    except Exception as e:
        logger.warning(
            f"[ToolAnalyst] Failed to parse JSON from {ticker} analysis: {e}"
        )
        rationale = final_text[:200]

    execution_time = time.monotonic() - start

    return {
        "ticker": ticker,
        "action": action if action in ["BUY", "SELL", "HOLD"] else "HOLD",
        "confidence": confidence,
        "rationale": rationale,
        "config_used": "Tool_Analyst",
        "tokens_used": result.get("token_usage", 0),
        "total_time_s": round(execution_time, 2),
        "method": "tool_agent",
    }

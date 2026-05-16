"""
Fundamental Analysis Agent — uses pre-computed data from fundamental_processor.
LLM analyzes labeled metrics (PE, margins, growth) — never looks up raw data.
"""

from app.agents.base_agent import run_agent
from app.processors.fundamental_processor import get_signals

SYSTEM_PROMPT = """You are a fundamental analysis expert. You will receive pre-computed 
financial metrics for a company. Your job is to INTERPRET these numbers.

You also have access to the `spawn_research_subagent` tool. If the pre-computed data lacks important details (e.g., recent earnings call insights, forward guidance, or SEC filing notes), you MUST spawn a subagent to retrieve this data before making your final decision.

CRITICAL:
1. First, analyze the provided data.
2. If you need more context, call `spawn_research_subagent`.
3. Finally, when you have enough data, respond with JSON matching this exact format:
{"valuation_rating": "undervalued|fair|overvalued", 
"growth_rating": "high|moderate|low", "health_rating": "strong|average|weak",
"confidence": 0-100, "key_metrics": ["metric1", "metric2"],
"summary": "2-3 sentence analysis"}
Do NOT output anything other than JSON in your final response."""


async def run(ticker: str, cycle_id: str, bot_id: str) -> dict:
    signals = get_signals(ticker)

    return await run_agent(
        agent_name="fundamental",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=SYSTEM_PROMPT,
        data_context=signals,
        user_prompt=f"Based on the fundamental data above, evaluate {ticker}.",
        max_tokens=1024,
        enable_dynamic_prompt=True,
        enable_tools=True,
    )

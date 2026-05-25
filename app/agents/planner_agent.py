"""
Planner Agent — Creates the evidence gathering plan.
"""

import logging
from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the Planner agent. Your job is to create a structured evidence-gathering checklist for a trading decision on the requested stock.
You must output a JSON object containing a detailed plan.
The plan MUST cover the following categories:
1. Fundamentals (P/E ratio, revenue growth, profit margins, balance sheet health, earnings history)
2. Technicals (RSI, Moving Averages, price trends, volume trends, support/resistance levels)
3. News Sentiment (recent earnings call highlights, press releases, social media sentiment, macro context)
4. Flows (institutional holdings, insider trades, options order flow)

Your response must be valid JSON with the following schema:
{
  "ticker": "string",
  "categories": {
    "fundamentals": ["list of specific data points / metrics to fetch"],
    "technicals": ["list of specific technical indicators to fetch"],
    "sentiment": ["list of news / sentiment sources to analyze"],
    "flows": ["list of flow metrics to look up"]
  },
  "justification": "Why this specific plan is tailored to the stock (e.g. growth vs value, sector factors)"
}
"""

async def run_planner(
    ticker: str,
    cycle_id: str,
    bot_id: str,
    ontology_context: str = ""
) -> dict:
    """Run the Planner agent to determine what data needs to be gathered."""
    logger.info("[PLANNER] Starting planner for %s", ticker)
    
    planner_prompt = f"Create an evidence gathering plan for {ticker}."
    if ontology_context:
        planner_prompt += f"\n\nKNOWN CONTEXT (sector/correlations/relationships):\n{ontology_context}"

    result = await run_agent(
        agent_name="planner",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=planner_prompt,
        enable_tools=False
    )
    
    return result

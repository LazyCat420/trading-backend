"""
Retriever Agent — Calls data tools to fetch evidence based on a plan.
"""

import logging
from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)

RETRIEVER_SYSTEM_PROMPT = """You are the Retriever agent. Your goal is to systematically execute the evidence gathering plan by calling the appropriate data tools.
You MUST gather data for ALL of the categories listed in the plan (Fundamentals, Technicals, News Sentiment, and Flows).
Use the available tools (e.g. read_file, get_cycle_context, or whatever tools are whitelisted for your role) to retrieve the information.
Do not make up, estimate, or assume any metrics. The data retrieved must be precise and match the tool outputs exactly.

Respond with a JSON object summarizing your findings:
{
  "ticker": "string",
  "gathered_data": {
    "fundamentals": "summarized fundamentals evidence with exact metrics (e.g. P/E, PEG, debt)",
    "technicals": "summarized technical indicators and levels",
    "sentiment": "summarized news and sentiment trends",
    "flows": "summarized flows (insider/options/institutional)"
  },
  "missing_data": ["any requested data points you were unable to fetch"],
  "data_freshness": "how recent the data is based on the fetched files"
}
"""

async def run_retriever(
    ticker: str,
    cycle_id: str,
    bot_id: str,
    capsule_context: str
) -> dict:
    """Run the Retriever agent to fetch data."""
    logger.info("[RETRIEVER] Starting retriever for %s", ticker)

    result = await run_agent(
        agent_name="retriever",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=RETRIEVER_SYSTEM_PROMPT,
        user_prompt=f"Execute this plan for {ticker} by calling tools:\n{capsule_context}",
        enable_tools=True
    )
    
    return result

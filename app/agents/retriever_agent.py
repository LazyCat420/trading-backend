"""
Retriever Agent — Calls data tools to fetch evidence based on a plan.
"""

import logging
from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)

RETRIEVER_SYSTEM_PROMPT = "You are the Retriever agent. You must call data tools to fetch evidence. Respond in JSON."

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

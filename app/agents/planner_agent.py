"""
Planner Agent — Creates the evidence gathering plan.
"""

import logging
from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = "You are the Planner agent. Determine what data is needed to make a trading decision. Respond in JSON."

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

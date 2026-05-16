"""
Verifier Agent — Checks for contradictions in retrieved data.
"""

import logging
from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = "You are the Verifier agent. Check for contradictions in retrieved data. Use get_cycle_context tool to expand capsule references if you need full details. Respond in JSON."

async def run_verifier(
    ticker: str,
    cycle_id: str,
    bot_id: str,
    capsule_context: str
) -> dict:
    """Run the Verifier agent to check data."""
    logger.info("[VERIFIER] Starting verifier for %s", ticker)

    result = await run_agent(
        agent_name="verifier",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_prompt=f"Verify this evidence for {ticker}:\n{capsule_context}",
        enable_tools=True
    )
    
    return result

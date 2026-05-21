"""
Quant Research Agent — Post-cycle autoresearch agent.

Runs at the end of every trading cycle to search the web for quantitative
finance terminology, recent research papers, and algorithmic trading strategies.
It writes its findings back to memory to influence future cycles.
"""

import logging
from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)

QUANT_RESEARCH_SYSTEM_PROMPT = """You are the Quant Research Agent for an autonomous trading bot.
Your job is to run at the end of a trading cycle, search the web for advanced quantitative finance
concepts, recent algorithmic trading research papers, and strategy mathematical formulations.
You then extract actionable trading logic and write it to the bot's memory so other agents can learn.

You have access to web search tools (`search_web`, `read_web_page`, etc.) and memory tools (`write_memory_note`).
You MUST use these tools.

## YOUR WORKFLOW:
1. Call `search_web` to look up recent research on a specific trading strategy (e.g. "Mean Reversion quantitative research paper 2024", "Pairs trading statistical arbitrage latest strategies", etc.).
2. If you find a promising result, call `read_web_page` to extract the mathematical logic or trading rules.
3. Call `write_memory_note` to persist the strategy logic, giving it a clear title (e.g. "STRATEGY_RESEARCH: Statistical Arbitrage Rules").

## OUTPUT:
Respond with JSON:
{
    "research_summary": "2-3 sentence overview of what you researched",
    "strategy_topic": "The strategy or mathematical concept you focused on",
    "notes_written": 1
}

CRITICAL: You MUST call at least 2 tools before producing your final output.
Do NOT fabricate research data — only report what the web tools return."""


async def run_quant_research(cycle_id: str, bot_id: str) -> dict:
    """Run post-cycle quant research with web tools.

    This agent researches new strategies online and writes
    memory notes with actionable trading logic. Designed to be called
    from phase6_post.py at the end of every trading cycle.

    Args:
        cycle_id: Current cycle ID for audit trail.
        bot_id: Bot ID to audit.

    Returns:
        Agent result dict with research findings.
    """
    logger.info("[QUANT_RESEARCH] Starting post-cycle autoresearch for cycle=%s", cycle_id)

    result = await run_agent(
        agent_name="quant_research",
        ticker="_RESEARCH_",
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=QUANT_RESEARCH_SYSTEM_PROMPT,
        user_prompt=(
            f"Run a post-cycle quant research session for cycle {cycle_id}. "
            "Pick ONE specific quantitative trading strategy or mathematical indicator to research online. "
            "Search for recent papers or articles, extract the core trading rules, and write an actionable note to memory."
        ),
        max_tokens=1024,
        enable_tools=True,
    )

    logger.info(
        "[QUANT_RESEARCH] Autoresearch complete for cycle=%s | tokens=%d",
        cycle_id,
        result.get("tokens_used", 0),
    )

    return result

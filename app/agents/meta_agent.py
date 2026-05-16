"""
Meta-Agent — LLM-powered prompt generator for novel analytical lenses.

Reviews winning strategies and debate patterns to generate new system
prompts (analytical lenses). Generated prompts are tracked for P&L
and benched if they underperform.

Architecture:
    - Pure LLM call (Rule 7 compliant — no data fetching)
    - Data comes IN via parameters from meta_agent_runner.py
    - Uses base_agent pattern for LLM interaction
    - Outputs a new system prompt as JSON

Usage:
    from app.agents.meta_agent import generate_prompt

    result = await generate_prompt(
        winning_patterns="Top 3 strategies: ...",
        debate_insights="Recent debates showed: ...",
        cycle_id="cycle-abc",
    )
"""

import logging

from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)


META_SYSTEM_PROMPT = """You are an expert at designing analytical frameworks for stock market trading.

Your job: Given data about which trading strategies have been WINNING and which LOSING, 
invent a NEW analytical lens — a system prompt for a specialist trading agent.

Requirements for the generated system prompt:
1. It must focus on a SPECIFIC analytical angle not already covered by existing lenses
2. It must instruct the agent to analyze pre-computed market data (never fetch its own data)
3. It must produce JSON output: {"signal": "BUY|SELL|HOLD", "confidence": 0-100, "rationale": "..."}
4. It must be concise (under 200 words) — shorter prompts produce better LLM output
5. It should incorporate lessons from winning and losing patterns

Respond with ONLY JSON:
{
    "name": "short_descriptive_name (e.g., 'earnings_momentum_tracker')",
    "lens_type": "category (e.g., 'momentum', 'value', 'macro', 'sentiment', 'hybrid')",
    "system_prompt": "the full system prompt for the new agent (include JSON output schema)",
    "rationale": "1-2 sentences explaining what gap this lens fills and why it should work"
}"""


META_USER_TEMPLATE = """{focus_section}### Current Strategy Performance

### Top Winning Patterns
{winning_patterns}

### Underperforming Patterns
{losing_patterns}

### Recent Debate Insights
{debate_insights}

### Existing Active Lenses
{existing_lenses}

---

Based on the winning and losing patterns above, design a NEW analytical lens that:
1. Exploits patterns seen in winning strategies
2. Avoids the mistakes of losing strategies  
3. Covers a gap not addressed by existing lenses
4. Is grounded in real trading principles (not generic advice)"""


async def generate_prompt(
    winning_patterns: str,
    losing_patterns: str = "No data yet",
    debate_insights: str = "No data yet",
    existing_lenses: str = "",
    cycle_id: str = "",
    bot_id: str = "",
) -> dict:
    """Generate a new analytical lens via LLM.

    Args:
        winning_patterns: Text summary of top-performing strategies
        losing_patterns: Text summary of underperforming strategies
        debate_insights: Text summary of recent debate outcomes
        existing_lenses: Comma-separated list of existing lens names
        cycle_id: For audit trail
        bot_id: For audit trail

    Returns:
        Dict with name, lens_type, system_prompt, rationale.
        Returns empty dict on failure.
    """
    import os

    cycle_research_focus = os.environ.get("CYCLE_RESEARCH_FOCUS", "")
    focus_section = (
        f"## Current Cycle Research Focus\nThe trader has explicitly requested the following focus for this cycle:\n{cycle_research_focus}\n\n"
        if cycle_research_focus
        else ""
    )

    user_prompt = META_USER_TEMPLATE.format(
        focus_section=focus_section,
        winning_patterns=winning_patterns or "No winning strategy data available yet",
        losing_patterns=losing_patterns or "No losing strategy data available yet",
        debate_insights=debate_insights or "No debate data available yet",
        existing_lenses=existing_lenses or "None active",
    )

    result = await run_agent(
        agent_name="meta_agent",
        ticker="_META_",
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=META_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.7,  # Higher temp for creative prompt generation
        max_tokens=768,
    )

    from app.utils.text_utils import parse_json_response

    parsed = parse_json_response(result.get("response", ""))

    if not parsed.get("system_prompt"):
        logger.warning("[META_AGENT] Generated prompt was empty or unparseable")
        return {}

    logger.info(
        "[META_AGENT] Generated lens: '%s' (%s) — %s",
        parsed.get("name", "unknown"),
        parsed.get("lens_type", "unknown"),
        parsed.get("rationale", "")[:100],
    )

    return {
        "name": parsed.get("name", "unnamed_lens"),
        "lens_type": parsed.get("lens_type", "custom"),
        "system_prompt": parsed["system_prompt"],
        "rationale": parsed.get("rationale", ""),
        "tokens_used": result.get("tokens_used", 0),
    }

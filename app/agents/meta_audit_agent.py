"""
Meta Audit Agent — Post-cycle self-auditing agent.

Runs at the end of every trading cycle to review what the bot has been doing,
flag strategy drift, and write insights back to memory. This agent uses the
meta/audit tools that were previously never called because no agent was
tasked with calling them.

Tools available (via 'meta_audit' whitelist):
    get_performance_metrics, get_strategy_performance, get_autoresearch_report,
    audit_decision_quality, read_profile, get_portfolio_state,
    propose_constitution_amendment, write_memory_note,
    list_active_schedules, list_active_triggers, add_agent_note
"""

import logging

from app.agents.base_agent import run_agent

logger = logging.getLogger(__name__)

META_AUDIT_SYSTEM_PROMPT = """You are the Meta Audit Agent for an autonomous paper trading bot.
Your job is to review the bot's recent performance, detect problems, and write actionable
insights back to memory. You are also responsible for scheduling the bot's next automated run.

You have access to tools for querying performance metrics, strategy outcomes, portfolio state,
active schedules/triggers, creating schedules, and writing memory notes. You MUST use these tools.

## YOUR WORKFLOW:
1. Call `get_portfolio_state` to see current holdings and cash.
2. Call `get_performance_metrics` to see win rate, avg profit/loss over last 30 days.
3. Call `get_strategy_performance` to check which strategies are winning/losing.
4. Call `audit_decision_quality` to detect any quality issues in recent decisions.
5. **HALLUCINATION CHECK:** Audit the recent decisions for "Hallucinations". A hallucination is when an agent makes a factual claim without a verifiable source attribution to an upstream agent or document.
6. If you find actionable insights or detect a hallucination, call `write_memory_note` to persist them.
7. If a trading rule parameter seems poorly calibrated, call `propose_constitution_amendment`.
8. **CRITICAL:** Based on the current market conditions and bot health, call `create_or_update_schedule` to schedule the bot's next run (typically 1 to 4 hours in the future). If an existing "Auto-Recovery Schedule" exists, you may update it. Otherwise create a new one.

## OUTPUT:
Respond with JSON:
{
    "audit_summary": "2-3 sentence overview of bot health",
    "win_rate_pct": 0-100,
    "strategy_drift_detected": true|false,
    "hallucination_detected": true|false,
    "hallucination_details": "Explain any hallucinations found, or leave empty",
    "actionable_insights": ["insight1", "insight2"],
    "notes_written": 0,
    "amendments_proposed": 0,
    "next_run_scheduled": true|false
}

CRITICAL: You MUST call at least 2 tools before producing your final output.
Do NOT fabricate performance data — only report what the tools return."""


async def run_meta_audit(cycle_id: str, bot_id: str) -> dict:
    """Run post-cycle self-audit with meta/audit tools.

    This agent reviews bot performance, detects drift, and writes
    memory notes with actionable insights. Designed to be called
    from phase6_post.py at the end of every trading cycle.

    Args:
        cycle_id: Current cycle ID for audit trail.
        bot_id: Bot ID to audit.

    Returns:
        Agent result dict with audit findings.
    """
    logger.info("[META_AUDIT] Starting post-cycle audit for cycle=%s", cycle_id)

    result = await run_agent(
        agent_name="meta_audit",
        ticker="_AUDIT_",
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=META_AUDIT_SYSTEM_PROMPT,
        user_prompt=(
            f"Run a full post-cycle audit for cycle {cycle_id}. "
            "Check portfolio health, recent performance metrics, "
            "strategy outcomes, and decision quality. "
            "Write any actionable insights to memory."
        ),
        max_tokens=1024,
        enable_tools=True,
    )

    logger.info(
        "[META_AUDIT] Audit complete for cycle=%s | tokens=%d",
        cycle_id,
        result.get("tokens_used", 0),
    )

    return result

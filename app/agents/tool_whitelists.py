"""
Tool Whitelists — Per-agent tool filtering.

Each specialist agent should only see the tools relevant to its role.
This prevents the LLM from being overwhelmed by 66+ tool schemas and
dramatically increases the probability of calling the right tools.

Usage:
    from app.agents.tool_whitelists import get_agent_tools
    schemas = get_agent_tools("risk")  # Returns filtered list of tool schemas
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Agent → Tool Mappings ───────────────────────────────────────────────
# Each key is an agent_name, each value is the list of tool names that
# agent should have access to. Tools not in the whitelist are invisible
# to that agent during its run_agent_loop() execution.
#
# If an agent_name is NOT in this dict, it gets ALL tools (legacy behavior).

AGENT_TOOL_WHITELISTS: dict[str, list[str]] = {
    # ── V1 Specialist Agents ──
    "sentiment": [
        "get_market_data",
        "get_finnhub_news",
        "search_web",
        "query_hermes",
        "scrape_url",
        "search_internal_database",
        "search_trading_skills",
        "post_finding",
        "read_team_findings",
        "write_memory_note",
        "read_memory_note",
        "request_data_collection",
    ],
    "technical": [
        "get_market_data",
        "get_technical_indicators",
        "get_polygon_price_history",
        "get_options_flow",
        "search_trading_skills",
        "post_finding",
        "read_team_findings",
    ],
    "fundamental": [
        "get_market_data",
        "get_finviz_fundamentals",
        "get_sec_filings",
        "get_earnings_data",
        "search_trading_skills",
        "post_finding",
        "read_team_findings",
        "request_data_collection",
    ],
    "risk": [
        "get_market_data",
        "get_technical_indicators",
        "calculate_stop_loss",
        "calculate_position_size",
        "calculate_risk_reward",
        "calculate_portfolio_allocation",
        "get_portfolio_state",
        "get_position_pnl",
        "get_options_flow",
        "audit_data_quality",
    ],
    "fund_flow": [
        "get_sec_filings",
        "get_congress_trades",
        "get_insider_trades",
        "search_internal_database",
        "post_finding",
        "read_team_findings",
    ],
    "comparative": [
        "get_market_data",
        "get_technical_indicators",
        "get_polygon_price_history",
    ],
    # ── V2 Role-Based Agents ──
    "retriever": [
        "get_market_data",
        "get_finnhub_news",
        "get_technical_indicators",
        "get_finviz_fundamentals",
        "get_polygon_price_history",
        "search_web",
        "query_hermes",
        "scrape_url",
        "get_sec_filings",
        "get_options_flow",
        "get_insider_trades",
        "get_earnings_data",
        "get_macro_data",
        "get_congress_trades",
        "search_internal_database",
        "search_trading_skills",
        "request_data_collection",
    ],
    "verifier": [
        "audit_data_quality",
        "check_hallucination",
        "search_web",
        "get_market_data",
        "get_cycle_context",
    ],
    "synthesizer": [
        "get_cycle_context",
        "get_cycle_context_all",
        "execute_momentum_strategy",
        "execute_value_strategy",
        "run_quant_equation",
    ],
    # ── Pre-Trade Execution Agent ──
    "pre_trade": [
        "calculate_portfolio_allocation",
        "calculate_stop_loss",
        "calculate_position_size",
        "calculate_risk_reward",
        "set_price_trigger",
        "get_portfolio_state",
        "get_market_data",
        "get_technical_indicators",
        "buy_stock",
        "sell_stock",
        "get_cycle_context",
        "get_cycle_context_all",
    ],
    # ── Meta Audit Agent ──
    "meta_audit": [
        "get_performance_metrics",
        "get_strategy_performance",
        "get_autoresearch_report",
        "audit_decision_quality",
        "read_profile",
        "get_portfolio_state",
        "propose_constitution_amendment",
        "write_memory_note",
        "list_active_schedules",
        "list_active_triggers",
        "add_agent_note",
        "create_or_update_schedule",
    ],
    # ── Quant Research Agent ──
    "quant_research": [
        "search_web",
        "scrape_url",
        "search_wiki",
        "write_memory_note",
        "read_memory_note",
    ],
}


def get_agent_tools(agent_name: str) -> Optional[list[dict]]:
    """Resolve tool schemas for a given agent from the whitelist.

    Returns:
        A filtered list of tool schemas if the agent has a whitelist,
        or None if the agent should receive all tools (legacy behavior).
    """
    if agent_name not in AGENT_TOOL_WHITELISTS:
        return None

    from app.tools.registry import registry

    tool_names = AGENT_TOOL_WHITELISTS[agent_name]
    schemas = registry.get_schemas_by_names(tool_names)

    # Warn if any whitelisted tools don't exist in the registry
    found_names = {s["function"]["name"] for s in schemas}
    missing = set(tool_names) - found_names
    if missing:
        logger.warning(
            "[ToolWhitelist] Agent '%s' references %d unregistered tools: %s",
            agent_name,
            len(missing),
            sorted(missing),
        )

    logger.debug(
        "[ToolWhitelist] Agent '%s' → %d/%d tools resolved",
        agent_name,
        len(schemas),
        len(tool_names),
    )
    return schemas if schemas else None
"""
Deterministic budget overrides per agent role.

Data collector agents stay at 3 turns (they just fetch).
Risk/validation agents get 5 turns (need to call calculators AFTER getting data).
Audit agents get 10 turns (need to review multiple performance dimensions).
"""

AGENT_BUDGET_OVERRIDES: dict[str, int] = {
    "risk": 5,
    "verifier": 5,
    "retriever": 5,
    "pre_trade": 6,
    "meta_audit": 10,
}


def get_agent_budget_turns(agent_name: str, enable_tools: bool) -> int:
    """Return the max_turns budget for a given agent.

    Args:
        agent_name: The name of the agent.
        enable_tools: Whether tools are enabled for this agent.

    Returns:
        Number of max turns for the agent's budget.
    """
    if not enable_tools:
        return 2  # Non-tool agents only need 1 turn for answer + 1 for potential REPAIR
    return AGENT_BUDGET_OVERRIDES.get(agent_name, 3)

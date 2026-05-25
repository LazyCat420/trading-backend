"""
Test: Tool Whitelists — Verify per-agent tool filtering.

Validates that:
1. Every tool name in every whitelist actually exists in the registry
2. No whitelist exceeds 20 tools (sanity cap)
3. Critical agent→tool mappings are present
4. get_agent_tools() returns filtered schemas
"""

import pytest


def test_all_whitelisted_tools_exist_in_registry():
    """Every tool name in AGENT_TOOL_WHITELISTS should exist in registry.tools.

    Note: In test environments without full dependencies (psycopg, aiohttp, etc.),
    the registry may only be partially populated. This test warns on missing tools
    but only fails if the registry has >30 tools registered (meaning deps are available).
    """
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS
    from app.tools.registry import registry

    registered_names = set(registry.tools.keys())

    # If registry is under-populated due to missing deps, skip gracefully
    if len(registered_names) < 30:
        pytest.skip(
            f"Registry only has {len(registered_names)} tools "
            f"(expected 60+). Missing runtime deps."
        )

    missing = {
        agent: [t for t in tool_list if t not in registered_names]
        for agent, tool_list in AGENT_TOOL_WHITELISTS.items()
        if any(t not in registered_names for t in tool_list)
    }

    assert not missing, f"Unregistered tools in whitelists: {missing}"


def test_no_whitelist_exceeds_cap():
    """No agent whitelist should have more than 20 tools."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    MAX_CAP = 20
    for agent, tool_list in AGENT_TOOL_WHITELISTS.items():
        assert len(tool_list) <= MAX_CAP, (
            f"Agent '{agent}' has {len(tool_list)} tools (cap={MAX_CAP})"
        )


def test_risk_agent_has_calculator_tools():
    """Risk agent MUST have access to all 4 calculator tools."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    risk_tools = set(AGENT_TOOL_WHITELISTS.get("risk", []))
    required = {
        "calculate_stop_loss",
        "calculate_position_size",
        "calculate_risk_reward",
        "calculate_portfolio_allocation",
    }
    missing = required - risk_tools
    assert not missing, f"Risk agent missing calculator tools: {missing}"


def test_pre_trade_agent_has_buy_and_calculators():
    """Pre-trade agent MUST have buy_stock and all calculator tools."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    pt_tools = set(AGENT_TOOL_WHITELISTS.get("pre_trade", []))
    required = {
        "buy_stock",
        "calculate_stop_loss",
        "calculate_position_size",
        "calculate_risk_reward",
        "calculate_portfolio_allocation",
    }
    missing = required - pt_tools
    assert not missing, f"Pre-trade agent missing required tools: {missing}"


def test_meta_audit_has_performance_tools():
    """Meta audit agent MUST have the performance/audit tools."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    meta_tools = set(AGENT_TOOL_WHITELISTS.get("meta_audit", []))
    required = {
        "get_performance_metrics",
        "audit_decision_quality",
        "write_memory_note",
    }
    missing = required - meta_tools
    assert not missing, f"Meta audit agent missing required tools: {missing}"


def test_sentiment_agent_does_not_have_calculator_tools():
    """Sentiment agent should NOT have calculator tools (irrelevant)."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    sentiment_tools = set(AGENT_TOOL_WHITELISTS.get("sentiment", []))
    calc_tools = {
        "calculate_stop_loss",
        "calculate_position_size",
        "calculate_risk_reward",
        "calculate_portfolio_allocation",
    }
    overlap = sentiment_tools & calc_tools
    assert not overlap, f"Sentiment agent should not have calculator tools: {overlap}"


def test_get_agent_tools_returns_filtered_schemas():
    """get_agent_tools() should return only the whitelisted schemas."""
    from app.agents.tool_whitelists import get_agent_tools, AGENT_TOOL_WHITELISTS

    schemas = get_agent_tools("technical")
    assert schemas is not None, "Technical agent should have a whitelist"

    schema_names = {s["function"]["name"] for s in schemas}
    expected = set(AGENT_TOOL_WHITELISTS["technical"])

    # Only registered tools should appear (some may be missing if unregistered)
    assert schema_names.issubset(expected), (
        f"Unexpected tools in technical schemas: {schema_names - expected}"
    )


def test_get_agent_tools_returns_none_for_unknown():
    """get_agent_tools() should return None for agents without a whitelist."""
    from app.agents.tool_whitelists import get_agent_tools

    result = get_agent_tools("nonexistent_agent_xyz")
    assert result is None, "Unknown agents should get None (= all tools)"


def test_no_duplicate_tools_in_whitelists():
    """No agent should have duplicate tool names in its whitelist."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    for agent, tool_list in AGENT_TOOL_WHITELISTS.items():
        dupes = [t for t in tool_list if tool_list.count(t) > 1]
        assert not dupes, f"Agent '{agent}' has duplicate tools: {set(dupes)}"

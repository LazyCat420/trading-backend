"""
Test: Tool Registry Cleanup — Verify removed tools are deregistered,
ghost tool references in swarm_consensus are fixed, and new whitelist
entries are correct.

Covers:
  1. Removed tools no longer in registry (grep_search_text, paginated_read,
     spawn_research_subagent, amend_constitution, browser_click, browser_type,
     browser_screenshot, browser_evaluate, browser_close)
  2. Swarm UNIVERSAL_TOOLS references only registered tools
  3. search_trading_skills is now in sentiment, technical, fundamental, retriever whitelists
  4. Tool count sanity check (should be < 70 after cleanup)
  5. SQL INTERVAL parameterization regression locks
"""

import pytest


# ── 1. Removed tools are gone from registry ──────────────────────────

REMOVED_TOOLS = [
    "grep_search_text",
    "paginated_read",
    "spawn_research_subagent",
    "amend_constitution",
    "browser_click",
    "browser_type",
    "browser_screenshot",
    "browser_evaluate",
    "browser_close",
]


@pytest.mark.parametrize("tool_name", REMOVED_TOOLS)
def test_removed_tool_not_in_registry(tool_name):
    """Verify each removed tool is no longer registered."""
    from app.tools.registry import registry

    assert tool_name not in registry.tools, (
        f"Tool '{tool_name}' should have been removed from the registry"
    )


def test_removed_tools_not_in_schemas():
    """Verify removed tools don't appear in the schema list."""
    from app.tools.registry import registry

    schema_names = {s["function"]["name"] for s in registry.schemas}
    overlap = schema_names & set(REMOVED_TOOLS)
    assert not overlap, f"Removed tools still in schemas: {overlap}"


# ── 2. Swarm UNIVERSAL_TOOLS only references registered tools ────────


def test_swarm_universal_tools_all_registered():
    """Every tool in swarm_consensus.UNIVERSAL_TOOLS must exist in the registry."""
    from app.pipeline.analysis.swarm_consensus import UNIVERSAL_TOOLS
    from app.tools.registry import registry

    registered_names = set(registry.tools.keys())

    # Skip if registry is under-populated (missing deps)
    if len(registered_names) < 30:
        pytest.skip(f"Registry only has {len(registered_names)} tools")

    missing = [t for t in UNIVERSAL_TOOLS if t not in registered_names]
    assert not missing, (
        f"Swarm UNIVERSAL_TOOLS references unregistered tools: {missing}"
    )


def test_swarm_does_not_reference_ghost_tools():
    """Regression: get_options_data and get_sector_competitors should NOT be in UNIVERSAL_TOOLS."""
    from app.pipeline.analysis.swarm_consensus import UNIVERSAL_TOOLS

    ghost_tools = {"get_options_data", "get_sector_competitors"}
    found = ghost_tools & set(UNIVERSAL_TOOLS)
    assert not found, f"Ghost tools still in UNIVERSAL_TOOLS: {found}"


def test_swarm_has_correct_options_tool():
    """Swarm should reference get_options_flow (not get_options_data)."""
    from app.pipeline.analysis.swarm_consensus import UNIVERSAL_TOOLS

    assert "get_options_flow" in UNIVERSAL_TOOLS, (
        "UNIVERSAL_TOOLS should have get_options_flow after fixing ghost reference"
    )


# ── 3. search_trading_skills whitelist coverage ──────────────────────


@pytest.mark.parametrize("agent_name", [
    "sentiment",
    "technical",
    "fundamental",
    "retriever",
])
def test_search_trading_skills_in_whitelist(agent_name):
    """search_trading_skills should now be in these agent whitelists."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    tools = AGENT_TOOL_WHITELISTS.get(agent_name, [])
    assert "search_trading_skills" in tools, (
        f"Agent '{agent_name}' should have search_trading_skills in whitelist"
    )


def test_search_trading_skills_not_in_risk_agent():
    """Risk agent doesn't need trading skills — only calculators."""
    from app.agents.tool_whitelists import AGENT_TOOL_WHITELISTS

    risk_tools = AGENT_TOOL_WHITELISTS.get("risk", [])
    assert "search_trading_skills" not in risk_tools


# ── 4. Tool count sanity ─────────────────────────────────────────────


def test_tool_count_after_cleanup():
    """After removing 9 tools, total should be < 65."""
    from app.tools.registry import registry

    count = len(registry.tools)
    assert count < 65, (
        f"Registry has {count} tools — expected < 65 after cleanup"
    )
    # Should still have the core tools
    assert count > 50, (
        f"Registry has {count} tools — suspiciously low, may have import issues"
    )


# ── 5. Kept tools still work ─────────────────────────────────────────


KEPT_TOOLS = [
    "browser_navigate",
    "run_playwright_script",
    "search_web",
    "scrape_url",
    "get_market_data",
    "search_trading_skills",
    "run_quant_equation",
]


@pytest.mark.parametrize("tool_name", KEPT_TOOLS)
def test_kept_tool_still_registered(tool_name):
    """Verify important kept tools are still registered."""
    from app.tools.registry import registry

    assert tool_name in registry.tools, (
        f"Tool '{tool_name}' should still be in the registry"
    )


# ── 6. SQL INTERVAL Parameterization Regression ──────────────────────


def test_tools_router_uses_correct_interval_pattern():
    """Regression: tools.py should use INTERVAL '1 hour' * %s, not INTERVAL '%s hours'."""
    import os
    path = "app/routers/tools.py"
    if not os.path.exists(path):
        # Try sibling directory for trading-client
        path = "../trading-client/app/routers/tools.py"
        if not os.path.exists(path):
            pytest.skip("tools.py not found in trading-service or trading-client")

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    # The broken pattern should NOT be present
    assert "INTERVAL '%s hours'" not in source, (
        "tools.py still contains the broken INTERVAL '%s hours' pattern"
    )
    assert "INTERVAL '%s days'" not in source, (
        "tools.py still contains the broken INTERVAL '%s days' pattern"
    )

    # The correct pattern SHOULD be present
    assert "INTERVAL '1 hour' * %s" in source, (
        "tools.py should use INTERVAL '1 hour' * %s for correct parameterization"
    )


def test_maintenance_agent_uses_correct_interval_pattern():
    """Regression: agent_maintenance.py should use INTERVAL '1 day' * %s."""
    with open("app/pipeline/analysis/agent_maintenance.py", "r", encoding="utf-8") as f:
        source = f.read()

    # The broken pattern should NOT be present
    assert "INTERVAL '%s days'" not in source, (
        "agent_maintenance.py still contains the broken INTERVAL '%s days' pattern"
    )

    # The correct pattern SHOULD be present
    assert "INTERVAL '1 day' * %s" in source, (
        "agent_maintenance.py should use INTERVAL '1 day' * %s"
    )

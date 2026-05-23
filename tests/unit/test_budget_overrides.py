"""
Test: Budget Overrides — Verify role-differentiated turn budgets.

Validates that:
1. Risk agent gets max_turns=5
2. Meta audit agent gets max_turns=10
3. Default agents still get max_turns=3
4. Non-tool agents get max_turns=2
"""

import pytest


@pytest.mark.parametrize("agent_name, enable_tools, expected_turns", [
    ("risk", True, 5),          # Needs room for calculator tools
    ("verifier", True, 5),      # Extra turns for verification
    ("retriever", True, 5),     # Extra turns for data gathering
    ("pre_trade", True, 12),    # Full calculator chain + buy
    ("meta_audit", True, 10),   # Deep audit cycle
    ("sentiment", True, 3),     # Not overridden, default
    ("unknown_agent_xyz", True, 3), # Unknown agent gets default
    ("risk", False, 2),         # Non-tool agents get 2
    ("meta_audit", False, 2),   # Non-tool agents get 2
])
def test_agent_budget_turns(agent_name, enable_tools, expected_turns):
    """Verify that agent role and tool availability determine their turn budget."""
    from app.agents.tool_whitelists import get_agent_budget_turns

    turns = get_agent_budget_turns(agent_name, enable_tools=enable_tools)
    assert turns == expected_turns, (
        f"Agent '{agent_name}' (tools={enable_tools}) should get {expected_turns} turns, got {turns}"
    )

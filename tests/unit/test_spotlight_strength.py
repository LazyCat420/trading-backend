"""
Test: Spotlight Strength — Verify the enhanced spotlight mechanism.

Validates that:
1. Spotlight returns 5 tools by default
2. The system prompt uses "REQUIRED TOOL CHECK" language
"""

import pytest
from unittest.mock import patch, MagicMock


def test_spotlight_default_limit_is_5():
    """get_spotlight_tools should have a default limit of 5."""
    import inspect
    from app.cognition.evolution.reflector import get_spotlight_tools

    sig = inspect.signature(get_spotlight_tools)
    limit_param = sig.parameters.get("limit")
    assert limit_param is not None, "get_spotlight_tools should have a 'limit' parameter"
    assert limit_param.default == 5, (
        f"Default limit should be 5, got {limit_param.default}"
    )


def test_agent_loop_calls_spotlight_with_limit_5():
    """agent_loop.py should call get_spotlight_tools(limit=5)."""
    import ast
    import app.agents.agent_loop as agent_loop

    with open(agent_loop.__file__, "r", encoding="utf-8") as f:
        source = f.read()

    # Parse the source to find the get_spotlight_tools call
    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "get_spotlight_tools":
                for kw in node.keywords:
                    if kw.arg == "limit":
                        if isinstance(kw.value, ast.Constant) and kw.value.value == 5:
                            found = True
    assert found, "agent_loop.py should call get_spotlight_tools(limit=5)"


def test_enhanced_prompt_uses_required_tool_check():
    """The spotlight prompt should use 'REQUIRED TOOL CHECK' language."""
    import app.agents.agent_loop as agent_loop

    with open(agent_loop.__file__, "r", encoding="utf-8") as f:
        source = f.read()

    assert "REQUIRED TOOL CHECK" in source, (
        "agent_loop.py should contain 'REQUIRED TOOL CHECK' in the spotlight prompt"
    )
    assert "consider testing" not in source, (
        "agent_loop.py should NOT contain the old weak 'consider testing' language"
    )

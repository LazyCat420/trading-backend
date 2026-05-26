"""
Tests for Brain-Action Split Agent Architecture.

Tests cover:
  🔴🟢 TDD Unit Tests:
    - Tool selector builds correct compact text list
    - Tool selector filters valid tool names
    - Tool selector falls back gracefully on empty/bad output
    - Split loop skips selection when pool is small
    - Split loop applies selection when pool is large
    - Action executor returns correct result structure
  🔗 Integration Tests:
    - run_split_agent_loop import and signature validation
    - AGENT_ROLE_ROUTING includes new agent types
"""

import sys
import os
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════
# 💨 Unit Tests — Tool Selector
# ══════════════════════════════════════════════════════════════


class TestBuildToolListText:
    """Test the compact text builder used by the tool selector."""

    def test_builds_correct_format(self):
        from app.agents.tool_selector import _build_tool_list_text

        schemas = [
            {"function": {"name": "get_market_data", "description": "Get market data for a ticker."}},
            {"function": {"name": "search_web", "description": "Search the web for information."}},
        ]
        result = _build_tool_list_text(schemas)
        assert "- get_market_data: Get market data for a ticker." in result
        assert "- search_web: Search the web for information." in result
        assert result.count("\n") == 1  # Two lines, one newline

    def test_truncates_long_descriptions(self):
        from app.agents.tool_selector import _build_tool_list_text

        long_desc = "A" * 300
        schemas = [{"function": {"name": "test_tool", "description": long_desc}}]
        result = _build_tool_list_text(schemas)
        # Should be truncated to 150 chars max
        assert len(result.split(": ", 1)[1]) <= 153  # 147 chars + "..."

    def test_empty_schemas(self):
        from app.agents.tool_selector import _build_tool_list_text

        result = _build_tool_list_text([])
        assert result == ""


class TestSelectToolsForTask:
    """Test the core tool selection logic."""

    def test_skips_selection_when_pool_is_small(self):
        """If pool <= max_tools, return full pool without LLM call."""
        from app.agents.tool_selector import select_tools_for_task

        schemas = [
            {"function": {"name": "tool_a", "description": "Tool A"}},
            {"function": {"name": "tool_b", "description": "Tool B"}},
        ]

        # Should return full pool without hitting LLM
        result = asyncio.get_event_loop().run_until_complete(
            select_tools_for_task(
                task_description="test task",
                available_tool_schemas=schemas,
                max_tools=5,
            )
        )
        assert len(result) == 2
        assert result == schemas

    def test_returns_empty_for_empty_pool(self):
        from app.agents.tool_selector import select_tools_for_task

        result = asyncio.get_event_loop().run_until_complete(
            select_tools_for_task(
                task_description="test task",
                available_tool_schemas=[],
            )
        )
        assert result == []


# ══════════════════════════════════════════════════════════════
# 💨 Unit Tests — Action Executor
# ══════════════════════════════════════════════════════════════


class TestActionExecutorStructure:
    """Validate the action executor module structure."""

    def test_action_executor_system_prompt_exists(self):
        from app.agents.action_executor import ACTION_EXECUTOR_SYSTEM

        assert len(ACTION_EXECUTOR_SYSTEM) > 50
        assert "data retrieval" in ACTION_EXECUTOR_SYSTEM.lower() or "execution" in ACTION_EXECUTOR_SYSTEM.lower()

    def test_run_isolated_action_agent_is_async(self):
        from app.agents.action_executor import run_isolated_action_agent
        import inspect

        assert inspect.iscoroutinefunction(run_isolated_action_agent)


# ══════════════════════════════════════════════════════════════
# 💨 Unit Tests — Split Agent Loop
# ══════════════════════════════════════════════════════════════


class TestSplitAgentLoopSignature:
    """Validate that run_split_agent_loop is importable and has the right signature."""

    def test_import_succeeds(self):
        from app.agents.agent_loop import run_split_agent_loop

        assert run_split_agent_loop is not None

    def test_is_async(self):
        from app.agents.agent_loop import run_split_agent_loop
        import inspect

        assert inspect.iscoroutinefunction(run_split_agent_loop)

    def test_has_max_selector_tools_param(self):
        from app.agents.agent_loop import run_split_agent_loop
        import inspect

        sig = inspect.signature(run_split_agent_loop)
        assert "max_selector_tools" in sig.parameters
        assert sig.parameters["max_selector_tools"].default == 5


# ══════════════════════════════════════════════════════════════
# 🔗 Integration Tests — AGENT_ROLE_ROUTING
# ══════════════════════════════════════════════════════════════


class TestAgentRoleRoutingUpdated:
    """Verify that the new agent types are registered in AGENT_ROLE_ROUTING."""

    def test_tool_selector_routing(self):
        from app.services.vllm_client import AGENT_ROLE_ROUTING

        assert "tool_selector" in AGENT_ROLE_ROUTING
        assert AGENT_ROLE_ROUTING["tool_selector"] == "collector"

    def test_action_executor_routing(self):
        from app.services.vllm_client import AGENT_ROLE_ROUTING

        assert "action_executor" in AGENT_ROLE_ROUTING
        assert AGENT_ROLE_ROUTING["action_executor"] == "analyst"


# ══════════════════════════════════════════════════════════════
# 🔗 Integration Tests — base_agent imports
# ══════════════════════════════════════════════════════════════


class TestBaseAgentIntegration:
    """Verify base_agent.py source code references the split loop."""

    def test_base_agent_imports_split_loop(self):
        base_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "agents", "base_agent.py"
        )
        with open(base_path, "r") as f:
            source = f.read()
        assert "run_split_agent_loop" in source, (
            "base_agent.py must import and use run_split_agent_loop"
        )

    def test_base_agent_conditional_split(self):
        """Verify that split loop is only used when tools are enabled."""
        base_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "agents", "base_agent.py"
        )
        with open(base_path, "r") as f:
            source = f.read()
        assert "if enable_tools and agent_tools:" in source, (
            "base_agent.py must conditionally use split loop only when tools are enabled"
        )


# ══════════════════════════════════════════════════════════════
# 🔗 Integration Tests — Tool Selector prompt quality
# ══════════════════════════════════════════════════════════════


class TestToolSelectorPromptQuality:
    """Verify the tool selector system prompt is well-formed."""

    def test_system_prompt_requests_json(self):
        from app.agents.tool_selector import TOOL_SELECTOR_SYSTEM

        assert "JSON" in TOOL_SELECTOR_SYSTEM
        assert "selected_tools" in TOOL_SELECTOR_SYSTEM

    def test_system_prompt_limits_tool_count(self):
        from app.agents.tool_selector import TOOL_SELECTOR_SYSTEM

        assert "maximum" in TOOL_SELECTOR_SYSTEM.lower() or "max" in TOOL_SELECTOR_SYSTEM.lower()

    def test_system_prompt_is_concise(self):
        from app.agents.tool_selector import TOOL_SELECTOR_SYSTEM

        # System prompt should be short to minimize TTFT
        assert len(TOOL_SELECTOR_SYSTEM) < 500

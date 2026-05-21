"""
Test: Ticker-Lock Enforcement in Tool Registry.

Regression test for the always-on ticker-lock guardrail that prevents
cross-contamination in debates (e.g., GEV debate calling NVDA tools).

The guardrail fires whenever a `ticker` context is passed AND the tool
call's `ticker` argument doesn't match — regardless of enforce_ticker.
"""

import json
import pytest


@pytest.fixture
def clean_registry():
    """Create a fresh ToolRegistry with test tools registered."""
    from app.tools.registry import ToolRegistry
    reg = ToolRegistry()

    # Register a mock tool that takes a ticker parameter
    async def mock_get_data(ticker: str = ""):
        return json.dumps({"ticker": ticker, "price": 100.0})

    reg.register(
        mock_get_data,
        name="test_get_data",
        description="Test tool with ticker param",
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    )

    # Register a tool without ticker parameter
    async def mock_portfolio():
        return json.dumps({"positions": []})

    reg.register(
        mock_portfolio,
        name="test_portfolio",
        description="Test tool without ticker param",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    return reg


class TestTickerLock:

    @pytest.mark.asyncio
    async def test_mismatched_ticker_blocked(self, clean_registry):
        """Tool calls with wrong ticker are blocked when context ticker is set."""
        tool_call = {
            "id": "tc_001",
            "function": {
                "name": "test_get_data",
                "arguments": json.dumps({"ticker": "NVDA"}),
            },
        }

        result = await clean_registry.execute_tool_call(
            tool_call,
            agent_name="bull_agent",
            ticker="GEV",
            cycle_id="test-cycle",
        )

        assert result["role"] == "tool"
        content = json.loads(result["content"])
        assert "error" in content
        assert "Unauthorized ticker access" in content["error"]
        assert "GEV" in content["error"]
        assert "NVDA" in content["error"]

    @pytest.mark.asyncio
    async def test_matching_ticker_passes(self, clean_registry):
        """Tool calls with correct ticker should pass through and execute."""
        tool_call = {
            "id": "tc_002",
            "function": {
                "name": "test_get_data",
                "arguments": json.dumps({"ticker": "GEV"}),
            },
        }

        result = await clean_registry.execute_tool_call(
            tool_call,
            agent_name="bull_agent",
            ticker="GEV",
            cycle_id="test-cycle",
        )

        content = json.loads(result["content"])
        # Should execute successfully — ticker matches
        assert "Unauthorized ticker access" not in content.get("error", "")
        assert content.get("ticker") == "GEV"

    @pytest.mark.asyncio
    async def test_no_ticker_arg_passes(self, clean_registry):
        """Tools without a ticker parameter should not be blocked."""
        tool_call = {
            "id": "tc_003",
            "function": {
                "name": "test_portfolio",
                "arguments": json.dumps({}),
            },
        }

        result = await clean_registry.execute_tool_call(
            tool_call,
            agent_name="bull_agent",
            ticker="GEV",
            cycle_id="test-cycle",
        )

        content = json.loads(result["content"])
        assert "Unauthorized ticker access" not in content.get("error", "")
        assert "positions" in content

    @pytest.mark.asyncio
    async def test_no_context_ticker_allows_any(self, clean_registry):
        """Without a context ticker, any ticker argument is allowed."""
        tool_call = {
            "id": "tc_004",
            "function": {
                "name": "test_get_data",
                "arguments": json.dumps({"ticker": "NVDA"}),
            },
        }

        result = await clean_registry.execute_tool_call(
            tool_call,
            agent_name="bull_agent",
            ticker="",  # No context ticker
            cycle_id="test-cycle",
        )

        content = json.loads(result["content"])
        assert "Unauthorized ticker access" not in content.get("error", "")
        assert content.get("ticker") == "NVDA"

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, clean_registry):
        """Ticker matching should be case-insensitive."""
        tool_call = {
            "id": "tc_005",
            "function": {
                "name": "test_get_data",
                "arguments": json.dumps({"ticker": "gev"}),
            },
        }

        result = await clean_registry.execute_tool_call(
            tool_call,
            agent_name="bull_agent",
            ticker="GEV",
            cycle_id="test-cycle",
        )

        content = json.loads(result["content"])
        # Lowercase "gev" should match "GEV" — no block
        assert "Unauthorized ticker access" not in content.get("error", "")

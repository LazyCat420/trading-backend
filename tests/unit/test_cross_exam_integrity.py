"""
Test: Cross-Exam Audit Integrity.

Validates that the ToolRegistry properly enforces ticker matching
to prevent cross-contamination during debate cycles.
"""

import json
import pytest
from unittest.mock import patch
from app.tools.registry import ToolRegistry


@pytest.fixture
def clean_registry():
    test_reg = ToolRegistry()

    @test_reg.register(name="get_market_data")
    async def get_market_data(ticker: str):
        return f"market data for {ticker}"

    @test_reg.register(name="check_hallucination")
    async def check_hallucination(ticker: str):
        return f"hallucination check for {ticker}"

    return test_reg


@pytest.mark.asyncio
async def test_tool_calls_match_ticker(clean_registry):
    """Ensure tools block execution when requested ticker does not match assigned context ticker."""
    tc_mismatch = {
        "id": "call_123",
        "function": {
            "name": "get_market_data",
            "arguments": '{"ticker": "NVDA"}'
        }
    }

    # Patch _log_usage to avoid DB calls
    with patch.object(clean_registry, "_log_usage"):
        # Context is GEV, tool requested NVDA
        res = await clean_registry.execute_tool_call(tc_mismatch, ticker="GEV", enforce_ticker=True)

    content = json.loads(res["content"])
    assert "error" in content
    # Should flag either TICKER MISMATCH or Unauthorized ticker access
    assert "TICKER MISMATCH" in content["error"] or "Unauthorized ticker access" in content["error"]


@pytest.mark.asyncio
async def test_no_cross_contamination(clean_registry):
    """Ensure tools allow execution when requested ticker matches assigned context ticker."""
    tc_match = {
        "id": "call_124",
        "function": {
            "name": "get_market_data",
            "arguments": '{"ticker": "GEV"}'
        }
    }

    with patch.object(clean_registry, "_log_usage"):
        # Context is GEV, tool requested GEV
        res = await clean_registry.execute_tool_call(tc_match, ticker="GEV", enforce_ticker=True)

    assert res["content"] == "market data for GEV"


@pytest.mark.asyncio
async def test_no_total_unverified(clean_registry):
    """Ensure that tools lacking ticker argument are not blocked arbitrarily."""
    @clean_registry.register(name="get_general_news")
    async def get_general_news():
        return "general news"

    tc_no_ticker = {
        "id": "call_125",
        "function": {
            "name": "get_general_news",
            "arguments": '{}'
        }
    }

    with patch.object(clean_registry, "_log_usage"):
        res = await clean_registry.execute_tool_call(tc_no_ticker, ticker="GEV", enforce_ticker=True)

    assert res["content"] == "general news"

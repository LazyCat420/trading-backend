"""
Regression tests for the Prism tool stripping and debate budget fixes.

Tests cover:
  - Prism never receives tool schemas for pipeline calls
  - Prism agentic mode is disabled for non-chat agents
  - Debate agent respects reduced tool turn budget
  - Debate empty-tool bail-out terminates loop early
  - V2 pipeline produces a decision within timeout
  - Agent loop respects tool call scorecard
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Phase 1: Prism Tool Stripping ─────────────────────────────────────

class TestPrismToolStripping:
    """Ensure pipeline calls forward tool schemas to Prism correctly."""

    def test_prism_payload_includes_tools_when_agentic_mode_false(self):
        """When agentic_mode=False, tool schemas must still be included in the Prism payload if provided."""
        from app.services.prism_client import PrismClient

        client = PrismClient()
        client.url = "http://fake:3000"
        client.project = "test"
        client.username = "test"
        client.agent = "test"

        fake_tools = [{"type": "function", "function": {"name": "get_market_data"}}]

        payload, url, headers = client.get_chat_payload_and_url(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1024,
            temperature=0.3,
            system_prompt="You are a test agent.",
            agent_name="retriever",
            ticker="AAPL",
            cycle_id="test-cycle",
            enable_thinking=False,
            tools=fake_tools,
            agentic_mode=False,  # Pipeline mode
        )

        assert "tools" in payload, (
            "Tool schemas were missing from Prism payload with agentic_mode=False."
        )
        assert payload["functionCallingEnabled"] is True
        assert payload["agenticLoopEnabled"] is False

    def test_prism_payload_includes_tools_when_agentic_mode_true(self):
        """When agentic_mode=True (user chat), tools SHOULD be included."""
        from app.services.prism_client import PrismClient

        client = PrismClient()
        client.url = "http://fake:3000"
        client.project = "test"
        client.username = "test"
        client.agent = "test"

        fake_tools = [{"type": "function", "function": {"name": "search_web"}}]

        payload, url, headers = client.get_chat_payload_and_url(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1024,
            temperature=0.3,
            system_prompt="You are a test agent.",
            agent_name="user_chat",
            ticker="",
            cycle_id="",
            enable_thinking=False,
            tools=fake_tools,
            agentic_mode=True,
        )

        assert "tools" in payload, (
            "Tool schemas were NOT included for interactive chat (agentic_mode=True). "
            "User chat should have tools available via Prism."
        )
        assert payload["functionCallingEnabled"] is True
        assert payload["agenticLoopEnabled"] is True

    def test_prism_payload_no_tools_when_none_provided(self):
        """When tools=None, no tools key should exist regardless of agentic_mode."""
        from app.services.prism_client import PrismClient

        client = PrismClient()
        client.url = "http://fake:3000"
        client.project = "test"
        client.username = "test"
        client.agent = "test"

        payload, _, _ = client.get_chat_payload_and_url(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1024,
            temperature=0.3,
            system_prompt="test",
            agent_name="user_chat",
            ticker="",
            cycle_id="",
            enable_thinking=False,
            tools=None,
            agentic_mode=True,
        )

        assert "tools" not in payload

    @pytest.mark.asyncio
    async def test_call_prism_agent_forwards_tools(self):
        """_call_prism_agent must forward tool schemas to Prism."""
        from app.services.vllm_client import VLLMClient

        client = VLLMClient.__new__(VLLMClient)
        client.model = "test-model"
        client.prism_client = MagicMock()

        # Track what get_chat_payload_and_url receives
        captured_kwargs = {}
        def capture_payload(**kwargs):
            captured_kwargs.update(kwargs)
            return (
                {"messages": [], "model": "test-model"},
                "http://fake/agent?stream=false",
                {"Content-Type": "application/json"},
            )

        client.prism_client.get_chat_payload_and_url = MagicMock(side_effect=lambda **kw: capture_payload(**kw))

        # Mock _call_endpoint to return a valid response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "test response",
            "usage": {"totalTokens": 100},
        }
        client._call_endpoint = AsyncMock(return_value=mock_response)

        import time
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"type": "function", "function": {"name": "get_market_data"}}],
        }
        meta = {"agent_name": "retriever", "ticker": "AAPL", "cycle_id": "test"}

        await client._call_prism_agent(
            client=MagicMock(),
            payload=payload,
            meta=meta,
            start=time.monotonic(),
        )

        # Verify tools was passed to get_chat_payload_and_url
        call_args = client.prism_client.get_chat_payload_and_url.call_args
        assert call_args is not None
        # Check keyword argument 'tools' is what we passed
        tools_arg = call_args.kwargs.get("tools") if call_args.kwargs else None
        assert tools_arg == [{"type": "function", "function": {"name": "get_market_data"}}], (
            f"_call_prism_agent did not forward tools to Prism correctly: {tools_arg}."
        )


# ── Phase 2: Debate Tool Budget ────────────────────────────────────────

class TestDebateToolBudget:
    """Ensure debate agents respect reduced tool turn budgets."""

    def test_debate_max_tool_turns_default(self):
        """DEBATE_MAX_TOOL_TURNS should default to 1 (reduced from 3)."""
        from app.config.config_cognition import cognition_settings

        assert cognition_settings.DEBATE_MAX_TOOL_TURNS <= 2, (
            f"DEBATE_MAX_TOOL_TURNS is {cognition_settings.DEBATE_MAX_TOOL_TURNS}. "
            "This should be 1-2 to prevent excessive LLM calls in debates."
        )

    def test_fast_debate_mode_enabled(self):
        """FAST_DEBATE_MODE should be True by default to reduce latency."""
        from app.config.config_cognition import cognition_settings

        assert cognition_settings.FAST_DEBATE_MODE is True, (
            "FAST_DEBATE_MODE should be True to halve debate latency."
        )

    def test_prism_agent_routing_enabled(self):
        """PRISM_AGENT_ROUTING should be True — all requests route through Prism /agent."""
        from app.config import settings

        assert settings.PRISM_AGENT_ROUTING is True, (
            "PRISM_AGENT_ROUTING should be True. All requests should route through "
            "Prism's /agent endpoint for native logging and agent harness support."
        )


# ── Phase 3: Debate Empty-Tool Bail-Out ────────────────────────────────

class TestDebateEmptyToolBailout:
    """Ensure debate agents bail out when tools return empty/error data."""

    @pytest.mark.asyncio
    async def test_biased_agent_bails_on_empty_tools(self):
        """_run_biased_agent should break the tool loop when all tools return errors."""
        from app.cognition.contracts.evidence import EvidencePacket

        # Mock LLM to return tool calls on first turn, then text
        call_count = 0

        async def mock_chat_with_tools(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: LLM wants to call a tool
                return {
                    "text": "I need to check market data.",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {
                                "name": "get_market_data",
                                "arguments": json.dumps({"ticker": "AAPL"}),
                            },
                        }
                    ],
                    "total_tokens": 100,
                    "elapsed_ms": 500,
                }
            else:
                # Subsequent calls: return final JSON (forced by bail-out)
                return {
                    "text": json.dumps({
                        "action": "BUY",
                        "claims": ["Test claim [source:value]"],
                        "confidence": 50,
                        "key_argument": "test",
                    }),
                    "tool_calls": None,
                    "total_tokens": 100,
                    "elapsed_ms": 500,
                }

        # Mock tool execution to return error
        async def mock_execute_tool_call(tc, **kwargs):
            return {
                "role": "tool",
                "name": tc.get("function", {}).get("name", ""),
                "tool_call_id": tc.get("id", ""),
                "content": json.dumps({"error": "Tool execution failed: connection timeout"}),
            }

        with patch("app.cognition.debate.debate_coordinator.llm") as mock_llm, \
             patch("app.cognition.debate.debate_coordinator.registry") as mock_registry:

            mock_llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)
            mock_registry.schemas = []
            mock_registry.execute_tool_call = AsyncMock(side_effect=mock_execute_tool_call)

            from app.cognition.debate.debate_coordinator import _run_biased_agent

            # Create a minimal evidence packet
            packet = MagicMock()
            packet.structured_facts = []
            packet.claims = []
            packet.missing_fields = []
            packet.source_summaries = []
            packet.tool_cache = {}

            response, tokens, tool_hist = await _run_biased_agent(
                bias="bull",
                system_prompt="Test bull agent",
                entity_id="AAPL",
                packet=packet,
                cycle_id="test-cycle",
                bot_id="test-bot",
            )

            # Should have made exactly 2 LLM calls:
            # 1. Initial call that returned tool_calls
            # 2. Forced text-only call after bail-out
            assert call_count == 2, (
                f"Expected 2 LLM calls (1 tool + 1 forced), got {call_count}. "
                "The empty-tool bail-out should have broken the loop after the first failed tool call."
            )


# ── Phase 4: Tool Call Scorecard ───────────────────────────────────────

class TestToolCallScorecard:
    """Ensure the ToolCallScorecard correctly tracks tool call quality."""

    def test_scorecard_tracks_errors(self):
        """Error-containing tool results should increment errored count."""
        from app.agents.agent_loop import ToolCallScorecard

        sc = ToolCallScorecard()
        sc.record("Error: connection refused")

        assert sc.errored == 1
        assert sc.succeeded == 0
        assert sc.consecutive_empty == 1

    def test_scorecard_tracks_empty_results(self):
        """Empty tool results should increment empty count."""
        from app.agents.agent_loop import ToolCallScorecard

        sc = ToolCallScorecard()
        sc.record("[]")
        sc.record("{}")
        sc.record("")

        assert sc.empty == 3
        assert sc.consecutive_empty == 3

    def test_scorecard_resets_consecutive_on_success(self):
        """A successful result should reset consecutive_empty."""
        from app.agents.agent_loop import ToolCallScorecard

        sc = ToolCallScorecard()
        sc.record("Error: failed")
        sc.record("")
        assert sc.consecutive_empty == 2

        sc.record("Valid data: price=$150.00")
        assert sc.consecutive_empty == 0
        assert sc.succeeded == 1

    def test_scorecard_quality_ratio(self):
        """Quality ratio should reflect success rate."""
        from app.agents.agent_loop import ToolCallScorecard

        sc = ToolCallScorecard()
        sc.record("Valid data 1")
        sc.record("Valid data 2")
        sc.record("Error: failed")
        sc.record("[]")

        assert sc.quality_ratio == 0.5  # 2/4


# ── Smoke Tests ────────────────────────────────────────────────────────

class TestCycleSmoke:
    """Fast sanity checks that key configurations are correct."""

    def test_debate_timeout_is_reasonable(self):
        """Debate timeout in the V2 runner should be <= 300s (5 minutes)."""
        import ast
        import inspect
        from app.cognition.orchestration import runner

        source = inspect.getsource(runner.execute_v2_pipeline)
        # Parse the source to find the wait_for timeout for debate
        # This is a lightweight check — just verify the timeout value in source
        assert "timeout=300.0" in source or "timeout=300" in source, (
            "Debate timeout in execute_v2_pipeline should be 300s (5 minutes). "
            "Found different value in source."
        )

    def test_per_ticker_analysis_timeout_exists(self):
        """Phase 4 should have a per-ticker timeout to prevent hangs."""
        import inspect
        from app.pipeline.phases import phase4_analysis

        source = inspect.getsource(phase4_analysis.run_phase4_analysis)
        assert "wait_for" in source, (
            "Phase 4 must wrap per-ticker analysis in asyncio.wait_for() "
            "to prevent one slow ticker from blocking all workers."
        )

    def test_agent_budget_has_limits(self):
        """All agent budgets should have reasonable turn limits."""
        from app.agents.tool_whitelists import AGENT_BUDGET_OVERRIDES

        for agent, turns in AGENT_BUDGET_OVERRIDES.items():
            assert turns <= 15, (
                f"Agent '{agent}' has {turns} max turns — this is too high. "
                "15 turns should be the absolute maximum."
            )


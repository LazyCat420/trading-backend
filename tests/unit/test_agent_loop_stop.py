"""
Unit tests for agent_loop stop awareness.

Validates that:
  - Agent loop aborts immediately when cycle_control.is_stopped is True
  - Agent loop raises CancelledError (not a soft return) on stop
  - Agent loop respects stop even mid-turn (before calling LLM)
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.pipeline.orchestration.cycle_control import CycleControl
from app.agents.agent_budget import AgentBudget


@pytest.fixture
def stopped_cycle_control():
    """CycleControl that's already stopped."""
    cc = CycleControl()
    cc.stop()
    return cc


@pytest.fixture
def active_cycle_control():
    """CycleControl that's active (not stopped)."""
    return CycleControl()


def _mock_db_ctx():
    """Return a context manager mock for get_db() that handles execute().fetchone()."""
    db_conn = MagicMock()
    db_conn.execute.return_value.fetchone.return_value = None
    
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=db_conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ── 4.1: Agent loop aborts when stopped ──────────────────────────────────

@pytest.mark.asyncio
async def test_agent_loop_aborts_when_stopped(stopped_cycle_control):
    """If cycle_control.is_stopped during a turn, the loop exits immediately."""
    from app.agents.agent_loop import run_agent_loop

    with patch(
        "app.pipeline.orchestration.cycle_control.cycle_control", stopped_cycle_control
    ):
        with patch("app.db.connection.get_db", return_value=_mock_db_ctx()):
            with pytest.raises(asyncio.CancelledError, match="Pipeline stopped"):
                await run_agent_loop(
                    system_prompt="test system",
                    user_prompt="test user",
                    ticker="AAPL",
                    agent_name="test_agent",
                    cycle_id="test-cycle",
                    budget=AgentBudget(max_turns=10),
                )


# ── 4.2: Agent loop raises CancelledError, not soft return ───────────────

@pytest.mark.asyncio
async def test_agent_loop_raises_cancelled_on_stop(stopped_cycle_control):
    """Stop mid-loop must raise CancelledError, not return a partial result."""
    from app.agents.agent_loop import run_agent_loop

    with patch(
        "app.pipeline.orchestration.cycle_control.cycle_control", stopped_cycle_control
    ):
        with patch("app.db.connection.get_db", return_value=_mock_db_ctx()):
            raised = False
            try:
                await run_agent_loop(
                    system_prompt="test",
                    user_prompt="test",
                    ticker="AAPL",
                    agent_name="test_agent",
                    budget=AgentBudget(max_turns=5),
                )
            except asyncio.CancelledError:
                raised = True

            assert raised, "Agent loop should have raised CancelledError, not returned"


# ── 4.3: Agent loop runs normally when not stopped ───────────────────────

@pytest.mark.asyncio
async def test_agent_loop_runs_when_active(active_cycle_control):
    """Agent loop should complete normally when pipeline is not stopped."""
    from app.agents.agent_loop import run_agent_loop

    mock_llm_response = {
        "text": "Final analysis complete.",
        "total_tokens": 100,
        "elapsed_ms": 50,
        "tool_calls": None,
    }

    mock_llm = MagicMock()
    mock_llm.chat_with_tools = AsyncMock(return_value=mock_llm_response)

    with patch(
        "app.pipeline.orchestration.cycle_control.cycle_control", active_cycle_control
    ):
        with patch("app.agents.agent_loop.llm", mock_llm):
            with patch("app.db.connection.get_db", return_value=_mock_db_ctx()):
                with patch("app.agents.context_compressor.compress_history", new_callable=AsyncMock, side_effect=lambda m: m):
                    with patch("app.cognition.evolution.reflector.reflect_on_trajectory", new_callable=AsyncMock):
                        with patch("app.cognition.evolution.reflector.adjust_lesson_score"):
                            result = await run_agent_loop(
                                system_prompt="test",
                                user_prompt="test",
                                ticker="AAPL",
                                agent_name="test_agent",
                                budget=AgentBudget(max_turns=3),
                            )

                            assert result["final_text"] == "Final analysis complete."
                            assert result["stop_reason"] == "success"


# ── 4.4: Agent loop stops between turns, not mid-LLM-call ───────────────

@pytest.mark.asyncio
async def test_agent_loop_checks_stop_between_turns():
    """Agent loop should check stop at the START of each turn, not during LLM call."""
    cc = CycleControl()
    call_count = 0

    async def mock_chat_with_tools(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Stop the pipeline after the first LLM call completes
            cc.stop()
            return {
                "text": "Need more data.",
                "total_tokens": 50,
                "elapsed_ms": 25,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ],
            }
        return {
            "text": "Done.",
            "total_tokens": 50,
            "elapsed_ms": 25,
            "tool_calls": None,
        }

    from app.agents.agent_loop import run_agent_loop

    mock_llm = MagicMock()
    mock_llm.chat_with_tools = AsyncMock(side_effect=mock_chat_with_tools)

    mock_registry = MagicMock()
    mock_registry.schemas = []
    mock_registry.execute_tool_call = AsyncMock(
        return_value={
            "role": "tool",
            "name": "test_tool",
            "tool_call_id": "call_1",
            "content": '{"result": "data"}',
        }
    )

    with patch("app.pipeline.orchestration.cycle_control.cycle_control", cc):
        with patch("app.agents.agent_loop.llm", mock_llm):
            with patch("app.agents.agent_loop.registry", mock_registry):
                with patch("app.db.connection.get_db", return_value=_mock_db_ctx()):
                    with patch("app.agents.context_compressor.compress_history", new_callable=AsyncMock, side_effect=lambda m: m):
                        with patch("app.agents.context_compressor.summarize_tool_result", side_effect=lambda c, **kw: c):
                            with patch("app.cognition.evolution.reflector.reflect_on_trajectory", new_callable=AsyncMock):
                                with patch("app.cognition.evolution.reflector.adjust_lesson_score"):
                                    with pytest.raises(asyncio.CancelledError):
                                        await run_agent_loop(
                                            system_prompt="test",
                                            user_prompt="test",
                                            ticker="AAPL",
                                            agent_name="test_agent",
                                            budget=AgentBudget(max_turns=5),
                                        )

    # Only 1 LLM call should have been made
    assert call_count == 1, f"Expected 1 LLM call, got {call_count}"

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.pipeline.analysis.agent_execution import _get_resp, _try_recover_agent


def test_get_resp():
    assert _get_resp({"response": "hello"}) == "hello"
    assert _get_resp({"other": "data", "response": "world"}) == "world"
    assert _get_resp({"no_response_key": "here"}) == ""
    assert _get_resp("just a string") == "just a string"
    assert _get_resp(123) == "123"

    # Nested dictionary in response (LLM returned JSON object instead of string)
    nested_resp = _get_resp({"response": {"nested": "value"}})
    assert "nested" in nested_resp
    assert "value" in nested_resp
    assert isinstance(nested_resp, str)


@pytest.mark.asyncio
async def test_try_recover_agent_success():
    """Test successful fallback agent routing via registry."""
    with patch("app.pipeline.analysis.agent_execution.run_agent", new_callable=AsyncMock) as mock_run_agent:
        with patch("app.recovery.registry.agent_registry.find_fallback") as mock_find_fallback:
            with patch("app.recovery.registry.agent_registry.mark_degraded") as mock_mark_degraded:
                mock_find_fallback.return_value = "analyst"
                mock_run_agent.return_value = {"response": "recovered data", "tokens_used": 10}

                result = await _try_recover_agent("planner", Exception("crash"), "AAPL", "cycle-1", "bot-1")

                mock_mark_degraded.assert_called_once_with("planner_agent")
                mock_run_agent.assert_called_once()
                # Check args for run_agent
                call_args = mock_run_agent.call_args[1]
                assert call_args["agent_name"] == "analyst_fallback_for_planner"

                assert result["response"] == "recovered data"
                assert result["fallback_for"] == "planner"
                assert result["fallback_agent"] == "analyst"


@pytest.mark.asyncio
async def test_try_recover_agent_failure():
    """Test fallback agent failure defaults to generic failure dict."""
    with patch("app.recovery.registry.agent_registry.find_fallback") as mock_find_fallback:
        mock_find_fallback.return_value = None

        result = await _try_recover_agent("planner", Exception("crash"), "AAPL", "cycle-1", "bot-1")

        assert result["agent"] == "planner"
        assert result["ticker"] == "AAPL"
        assert result["response"] == "Agent failed: crash"
        assert result["tokens_used"] == 0




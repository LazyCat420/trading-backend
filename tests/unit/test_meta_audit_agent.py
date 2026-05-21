import pytest
import json
from unittest.mock import patch, MagicMock

from app.agents.meta_audit_agent import run_meta_audit

@pytest.fixture
def mock_agent_loop_meta():
    with patch("app.agents.meta_audit_agent.run_agent") as mock_agent:
        # Mock successful JSON response
        mock_agent.return_value = {
            "audit_summary": "The bot is performing well.",
            "win_rate_pct": 65.5,
            "strategy_drift_detected": False,
            "hallucination_detected": False,
            "hallucination_details": "",
            "actionable_insights": ["Focus on tech stocks"],
            "notes_written": 1,
            "amendments_proposed": 0,
            "next_run_scheduled": True,
            "tokens_used": 150
        }
        yield mock_agent

@pytest.mark.asyncio
async def test_meta_audit_agent_success(mock_agent_loop_meta):
    """
    Test that the meta audit agent correctly invokes run_agent and
    returns the parsed schema.
    """
    result = await run_meta_audit(cycle_id="test_cycle", bot_id="test_bot")
    
    # Verify the loop was called
    mock_agent_loop_meta.assert_called_once()
    
    # Verify the response
    assert result["audit_summary"] == "The bot is performing well."
    assert result["win_rate_pct"] == 65.5
    assert result["strategy_drift_detected"] is False
    assert len(result["actionable_insights"]) == 1
    assert result["notes_written"] == 1
    assert result["next_run_scheduled"] is True

@pytest.mark.asyncio
async def test_meta_audit_agent_prompt_injection(mock_agent_loop_meta):
    """
    Test that the meta audit agent injects the right prompts.
    """
    await run_meta_audit(cycle_id="cycle_123", bot_id="bot_456")
    
    # Inspect what was passed to run_agent_loop
    call_kwargs = mock_agent_loop_meta.call_args.kwargs
    system_prompt = call_kwargs.get("system_prompt", "")
    user_prompt = call_kwargs.get("user_prompt", "")
    
    assert "You are the Meta Audit Agent" in system_prompt
    assert "win_rate_pct" in system_prompt
    assert "actionable_insights" in system_prompt
    assert "Run a full post-cycle audit for cycle cycle_123" in user_prompt

@pytest.mark.asyncio
async def test_audit_flags_hallucination(mock_agent_loop_meta):
    """
    Assert that the system prompt explicitly requires checking for hallucinations.
    """
    await run_meta_audit(cycle_id="cycle_123", bot_id="bot_456")
    
    call_kwargs = mock_agent_loop_meta.call_args.kwargs
    system_prompt = call_kwargs.get("system_prompt", "")
    
    assert "HALLUCINATION CHECK" in system_prompt, "Meta auditor prompt is missing hallucination check rule"
    assert "hallucination_detected" in system_prompt, "Meta auditor schema is missing hallucination_detected flag"

@pytest.mark.asyncio
async def test_audit_enforces_source_attribution(mock_agent_loop_meta):
    """
    Assert that the system prompt defines a hallucination as a factual claim without a verifiable source attribution.
    """
    await run_meta_audit(cycle_id="cycle_123", bot_id="bot_456")
    
    call_kwargs = mock_agent_loop_meta.call_args.kwargs
    system_prompt = call_kwargs.get("system_prompt", "")
    
    assert "verifiable source attribution" in system_prompt, "Meta auditor is not enforcing source attribution tracking"

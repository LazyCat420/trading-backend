import pytest
from unittest.mock import AsyncMock, patch
import json

from app.agents.task_board import task_board
from app.tools.coordination_tools import request_investigation

@pytest.fixture(autouse=True)
def setup_teardown():
    """Ensure the task board is clean before and after tests."""
    task_board.clear_board("TEST_ROUTING")
    yield
    task_board.clear_board("TEST_ROUTING")

@pytest.mark.asyncio
async def test_routing_fundamental_target():
    """
    Simulate a scenario where the sentiment agent needs EPS data.
    Assert that it routes specifically to the fundamental_agent and not * or sentiment_agent.
    """
    # Simulate the LLM agent explicitly requesting 'fundamental_agent'
    resp_req = await request_investigation(
        question="What is the latest EPS and revenue growth?",
        ticker="TEST_ROUTING",
        target_agent="fundamental_agent",
        _agent_name="sentiment_agent",
        _cycle_id="cycle-1"
    )
    req_data = json.loads(resp_req)
    
    assert req_data["status"] == "requested"
    assert req_data["target"] == "fundamental_agent", "Agent failed to route to the specialized fundamental agent."
    
    # Verify that only the fundamental agent sees this request
    open_invs = await task_board.get_open_investigations(
        ticker="TEST_ROUTING", cycle_id="cycle-1", for_agent="fundamental_agent"
    )
    assert len(open_invs) == 1
    assert open_invs[0]["target_agent"] == "fundamental_agent"

@pytest.mark.asyncio
async def test_routing_avoids_loops():
    """
    Assert that the request_investigation tool rejects self-targeting 
    (an agent investigating itself) to prevent infinite dead-ends.
    """
    # Note: If the tool currently allows self-targeting, this test will fail, 
    # acting as a Red/Green TDD test to force us to fix the coordination_tools.py logic.
    resp_req = await request_investigation(
        question="Self-verification check",
        ticker="TEST_ROUTING",
        target_agent="sentiment_agent",
        _agent_name="sentiment_agent",
        _cycle_id="cycle-1"
    )
    req_data = json.loads(resp_req)
    
    # The system should either reject this or rewrite it. Let's enforce rejection for strictness.
    # If this fails, we need to implement the fix in coordination_tools.py!
    assert req_data.get("status") == "error", "System allowed an agent to route a request to itself, causing a potential infinite loop!"
    assert "cannot request investigation from yourself" in req_data.get("message", "").lower()

@pytest.mark.asyncio
async def test_routing_invalid_agent_fallback():
    """
    Assert that if an agent tries to send a report to a non-existent persona,
    the pipeline intercepts it and defaults to a valid fallback rather than crashing.
    """
    # Assuming valid agents are strict. If an LLM hallucinates an agent name:
    resp_req = await request_investigation(
        question="Check the CEO's astrology sign",
        ticker="TEST_ROUTING",
        target_agent="astrology_agent",
        _agent_name="sentiment_agent",
        _cycle_id="cycle-1"
    )
    req_data = json.loads(resp_req)
    
    # We expect the system to either reject the invalid agent or fallback to '*'
    assert req_data.get("status") == "error" or req_data.get("target") == "*", "System failed to handle a hallucinated agent routing target."

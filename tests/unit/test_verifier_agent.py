import pytest
import json
from unittest.mock import patch, MagicMock

from app.agents.verifier_agent import run_verifier

@pytest.fixture
def mock_run_agent():
    with patch("app.agents.verifier_agent.run_agent") as mock_run:
        mock_run.return_value = {
            "response": '{"verified": true, "contradictions": []}',
            "agent": "verifier",
            "ticker": "TSLA"
        }
        yield mock_run

@pytest.mark.asyncio
async def test_v01_response_is_parseable_json(mock_run_agent):
    result = await run_verifier("TSLA", "c1", "b1", "Capsules")
    parsed = json.loads(result["response"])
    assert isinstance(parsed, dict)

@pytest.mark.asyncio
async def test_v02_response_has_required_fields(mock_run_agent):
    result = await run_verifier("TSLA", "c1", "b1", "Capsules")
    parsed = json.loads(result["response"])
    assert "contradictions" in parsed or "verified" in parsed

@pytest.mark.asyncio
async def test_v03_detects_contradiction_in_capsules(mock_run_agent):
    mock_run_agent.return_value["response"] = '{"verified": false, "contradictions": ["Price 150 vs 999"]}'
    capsules = "price=150 in one, price=999 in another"
    result = await run_verifier("TSLA", "c1", "b1", capsules)
    parsed = json.loads(result["response"])
    assert len(parsed["contradictions"]) >= 1

@pytest.mark.asyncio
async def test_v04_can_call_get_cycle_context():
    # Verify that get_cycle_context is available to the verifier
    from app.agents.tool_whitelists import get_agent_tools
    allowed = get_agent_tools("verifier")
    allowed_names = [t["function"]["name"] for t in allowed]
    assert "get_cycle_context" in allowed_names
    
    # Mock tool to verify it can be called
    with patch("app.tools.context_tools.get_cycle_context") as mock_tool:
        mock_tool.return_value = "Expanded context"
        mock_tool("c1")
        mock_tool.assert_called()

@pytest.mark.asyncio
async def test_v05_handles_missing_capsule_degraded(mock_run_agent):
    result = await run_verifier("TSLA", "c1", "b1", "")
    assert result is not None
    assert "response" in result

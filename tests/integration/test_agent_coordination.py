import pytest
import json
from app.agents.task_board import task_board
from app.tools.coordination_tools import (
    post_finding,
    read_team_findings,
    request_investigation,
    check_open_investigations,
)


@pytest.fixture(autouse=True)
def setup_teardown_taskboard():
    """Clear the task board before and after each test for a clean state."""
    # We use a test ticker 'TEST'
    task_board.clear_board("TEST")
    task_board.clear_board("TEST", cycle_id="cycle-1")
    yield
    task_board.clear_board("TEST")
    task_board.clear_board("TEST", cycle_id="cycle-1")


@pytest.mark.asyncio
async def test_post_and_read_findings():
    """Verify that an agent can post a finding and another agent can read it, but the posting agent cannot read its own finding."""

    # Agent 1 posts a finding
    resp_post = await post_finding(
        content="RSI is 28.5, deeply oversold.",
        ticker="TEST",
        category="fact",
        confidence=90,
        _agent_name="agent_1",
        _cycle_id="cycle-1",
    )
    post_data = json.loads(resp_post)
    assert post_data["status"] == "posted"
    assert "finding_id" in post_data

    # Agent 2 reads findings
    resp_read_2 = await read_team_findings(
        ticker="TEST", category=None, _agent_name="agent_2", _cycle_id="cycle-1"
    )
    read_data_2 = json.loads(resp_read_2)
    assert read_data_2["status"] == "success"
    assert read_data_2["findings_count"] == 1
    assert read_data_2["findings"][0]["source_agent"] == "agent_1"
    assert "RSI is 28.5" in read_data_2["findings"][0]["content"]

    # Agent 1 tries to read its own findings (should be excluded)
    resp_read_1 = await read_team_findings(
        ticker="TEST", category=None, _agent_name="agent_1", _cycle_id="cycle-1"
    )
    read_data_1 = json.loads(resp_read_1)
    assert read_data_1["status"] == "success"
    # Should be empty because exclude_agent=agent_1
    assert len(read_data_1["findings"]) == 0
    assert "No findings from other agents yet." in read_data_1["message"]


@pytest.mark.asyncio
async def test_request_and_check_investigation():
    """Verify that an agent can request an open investigation and any other agent can see it."""

    # Agent 1 requests an investigation from ANY agent
    resp_req = await request_investigation(
        question="What is the PE ratio?",
        ticker="TEST",
        target_agent="*",
        _agent_name="agent_1",
        _cycle_id="cycle-1",
    )
    req_data = json.loads(resp_req)
    assert req_data["status"] == "requested"
    assert "investigation_id" in req_data
    assert req_data["target"] == "*"

    # Agent 2 checks for open investigations
    resp_check = await check_open_investigations(
        ticker="TEST", _agent_name="agent_2", _cycle_id="cycle-1"
    )
    check_data = json.loads(resp_check)
    assert check_data["status"] == "success"
    assert check_data["investigation_count"] == 1
    assert check_data["open_investigations"][0]["requester"] == "agent_1"
    assert check_data["open_investigations"][0]["question"] == "What is the PE ratio?"


@pytest.mark.asyncio
async def test_targeted_investigation():
    """Verify that an investigation targeted at a specific agent is only visible to that agent."""

    # fundamental_agent requests an investigation specifically from technical_agent
    await request_investigation(
        question="Check the latest 10-K filing",
        ticker="TEST",
        target_agent="technical_agent",
        _agent_name="fundamental_agent",
        _cycle_id="cycle-1",
    )

    # sentiment_agent checks for open investigations (should not see it)
    resp_check_2 = await check_open_investigations(
        ticker="TEST", _agent_name="sentiment_agent", _cycle_id="cycle-1"
    )
    check_data_2 = json.loads(resp_check_2)
    assert check_data_2["status"] == "success"
    assert len(check_data_2["open_investigations"]) == 0

    # technical_agent checks for open investigations (should see it)
    resp_check_3 = await check_open_investigations(
        ticker="TEST", _agent_name="technical_agent", _cycle_id="cycle-1"
    )
    check_data_3 = json.loads(resp_check_3)
    assert check_data_3["status"] == "success"
    assert check_data_3["investigation_count"] == 1
    assert check_data_3["open_investigations"][0]["target_agent"] == "technical_agent"

"""
Test: Debate Tool Turn Exhaustion Monitor.

Validates that debate agents properly restrict tool turns and force JSON 
claims upon exhaustion to prevent infinite loops.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.cognition.debate.debate_coordinator import _run_biased_agent, build_system_prompt
from app.cognition.contracts.evidence import EvidencePacket

@pytest.fixture
def mock_evidence_packet():
    packet = MagicMock(spec=EvidencePacket)
    packet.structured_facts = []
    packet.missing_fields = []
    packet.claims = []
    packet.source_summaries = []
    packet.tool_cache = {}
    return packet


@pytest.mark.asyncio
@patch("app.cognition.debate.debate_coordinator.llm.chat_with_tools")
async def test_agent_forces_json_on_exhaustion(mock_chat, mock_evidence_packet, monkeypatch):
    """Ensure the agent forces JSON output when tool turns are exhausted without claims."""
    
    monkeypatch.setattr("app.cognition.debate.debate_coordinator.cognition_settings.DEBATE_MAX_TOOL_TURNS", 3)

    # Mock chat to return tool calls for the first 3 turns, but NO JSON claims.
    # The 4th call should be the "force JSON" prompt with tools=None.
    mock_chat.side_effect = [
        {"text": "Thinking...", "tool_calls": [{"id": "1", "function": {"name": "get_market_data", "arguments": "{}"}}], "total_tokens": 10},
        {"text": "Still thinking...", "tool_calls": [{"id": "2", "function": {"name": "get_market_data", "arguments": "{}"}}], "total_tokens": 10},
        {"text": "One more...", "tool_calls": [{"id": "3", "function": {"name": "get_market_data", "arguments": "{}"}}], "total_tokens": 10},
        {"text": '{"action": "BUY", "claims": ["claim 1 [test:1]"], "confidence": 90, "key_argument": "test"}', "total_tokens": 50}
    ]
    
    # Patch tool_selector to prevent it from consuming mock chat responses,
    # and registry to prevent actual tool execution
    _fake_tool_schema = [{"type": "function", "function": {"name": "get_market_data", "parameters": {}}}]
    with patch("app.cognition.debate.debate_coordinator.registry.execute_tool_call") as mock_exec, \
         patch("app.agents.tool_selector.select_tools_for_task", return_value=_fake_tool_schema) as mock_selector:
        mock_exec.return_value = {"role": "tool", "content": "mock data"}
        
        system_prompt = build_system_prompt("bull", "Focus purely on fundamental data.")
        
        final_response, total_tokens, history = await _run_biased_agent(
            bias="bull",
            system_prompt=system_prompt,
            entity_id="GEV",
            packet=mock_evidence_packet,
            cycle_id="test",
            bot_id="bot1"
        )
        
    assert "claim 1 [test:1]" in final_response
    assert len(history) == 3  # 3 tool calls executed
    assert mock_chat.call_count == 4
    
    # Verify the 4th call enforced tools=None to force text/JSON output
    kwargs_4th_call = mock_chat.call_args_list[3].kwargs
    assert kwargs_4th_call.get("tools") is None


@pytest.mark.asyncio
@patch("app.cognition.debate.debate_coordinator.llm.chat_with_tools")
async def test_agent_stops_early_with_json(mock_chat, mock_evidence_packet, monkeypatch):
    """Ensure the agent stops and returns early if it outputs valid JSON without using all turns."""
    
    monkeypatch.setattr("app.cognition.debate.debate_coordinator.cognition_settings.DEBATE_MAX_TOOL_TURNS", 3)

    # Turn 1: uses tool
    # Turn 2: outputs JSON and stops
    mock_chat.side_effect = [
        {"text": "Thinking...", "tool_calls": [{"id": "1", "function": {"name": "get_market_data", "arguments": "{}"}}], "total_tokens": 10},
        {"text": '{"action": "SELL", "claims": ["claim 1 [test:1]"], "confidence": 90, "key_argument": "test"}', "total_tokens": 50}
    ]
    
    _fake_tool_schema = [{"type": "function", "function": {"name": "get_market_data", "parameters": {}}}]
    with patch("app.cognition.debate.debate_coordinator.registry.execute_tool_call") as mock_exec, \
         patch("app.agents.tool_selector.select_tools_for_task", return_value=_fake_tool_schema) as mock_selector:
        mock_exec.return_value = {"role": "tool", "content": "mock data"}
        
        system_prompt = build_system_prompt("bear", "Focus purely on fundamental data.")
        
        final_response, total_tokens, history = await _run_biased_agent(
            bias="bear",
            system_prompt=system_prompt,
            entity_id="SE",
            packet=mock_evidence_packet,
            cycle_id="test",
            bot_id="bot1"
        )
        
    assert "claim 1 [test:1]" in final_response
    assert len(history) == 1  # Only 1 tool call executed
    assert mock_chat.call_count == 2


import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from app.cognition.debate.debate_coordinator import _run_biased_agent

from app.cognition.contracts.evidence import EvidencePacket

from app.cognition.contracts.retrieval import StructuredFact

@pytest.mark.asyncio
async def test_realistic_evidence_payload_exhaustion():
    """Verify that the debate coordinator bails out and forces JSON after DEBATE_MAX_TOOL_TURNS."""
    
    # 1. Realistic Evidence Packet Mock
    evidence_packet = EvidencePacket(
        entity_id="AAPL",
        structured_facts=[
            StructuredFact(fact_type="price", key="current_price", value=150.0, verification_status="verified", confidence=1.0, sources=[], timestamp="2023-01-01T00:00:00Z"),
            StructuredFact(fact_type="metric", key="revenue_growth", value=0.05, verification_status="verified", confidence=0.9, sources=[], timestamp="2023-01-01T00:00:00Z")
        ],
        claims=[],
        source_summaries=[],
        contradictions=[],
        missing_fields=[],
        tool_cache={}
    )
    
    # 2. Mock LLM Client
    # We want the LLM to stubbornly output text in the first turn (a tool call or plain text without JSON claims).
    # Then in the forced second turn, we want it to output the valid JSON.
    
    call_count = 0
    total_tokens_accumulated = 0
    
    async def mock_chat_with_tools(messages, tools=None, **kwargs):
        nonlocal call_count, total_tokens_accumulated
        call_count += 1
        
        # Turn 1: Normal debate turn (with tools)
        if call_count == 1:
            # LLM ignores JSON and just calls a tool or outputs unstructured text
            response = {
                "text": "I need to check more data first.",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "get_insider_trades", "arguments": '{"ticker":"AAPL"}'}}
                ],
                "total_tokens": 150
            }
            total_tokens_accumulated += 150
            return response
            
        # Turn 2: The tool returned data, but DEBATE_MAX_TOOL_TURNS is 1.
        # Wait, if DEBATE_MAX_TOOL_TURNS is 1, the loop breaks AFTER the first tool call is processed,
        # and then the forced JSON fallback triggers!
        
        if call_count == 2:
            # This should be the forced JSON prompt turn
            assert tools is None, "Tools should be stripped in the forced JSON fallback!"
            last_msg = messages[-1]["content"]
            assert "You MUST now output your final verdict as JSON" in last_msg
            
            response = {
                "text": json.dumps({
                    "action": "BUY",
                    "claims": ["Strong revenue growth [source:financials]"],
                    "confidence": 85,
                    "key_argument": "Solid fundamentals outweigh supply constraints."
                }),
                "tool_calls": [],
                "total_tokens": 200
            }
            total_tokens_accumulated += 200
            return response
            
        raise ValueError("Too many calls to LLM")

    # We also need to mock the tool registry so the tool call succeeds
    async def mock_execute_tool(tc, *args, **kwargs):
        return {"role": "tool", "tool_call_id": tc["id"], "content": "No insider trades found."}

    # 3. Patching and Execution
    with patch("app.cognition.debate.debate_coordinator.llm") as mock_llm:
        mock_llm.chat_with_tools.side_effect = mock_chat_with_tools
        
        with patch("app.cognition.debate.debate_coordinator.cognition_settings") as mock_cog:
            mock_cog.DEBATE_MAX_TOOL_TURNS = 1
            
            with patch("app.cognition.debate.debate_coordinator.registry.execute_tool_call", side_effect=mock_execute_tool):
                
                final_response, tokens, history = await _run_biased_agent(
                    bias="bull",
                    system_prompt="You are a bull",
                    entity_id="AAPL",
                    packet=evidence_packet,
                    cycle_id="test_cycle",
                    bot_id="test_bot"
                )
                
                # 4. Assertions
                assert call_count == 2, "Should exactly be 1 tool turn + 1 forced JSON fallback"
                assert tokens == 350, f"Expected 350 tokens, got {tokens}"
                assert "BUY" in final_response
                assert len(history) == 1, "Should have exactly 1 tool execution history entry"
                assert "get_insider_trades" in history[0]

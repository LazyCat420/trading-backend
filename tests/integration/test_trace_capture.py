import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.agent_loop import run_agent_loop

@pytest.mark.asyncio
async def test_trace_capture_insertion(patch_llm, patch_get_db):
    """
    Test that the agent_loop extracts rationale and correctly 
    writes to the agent_traces table.
    """
    
    # Configure the mock LLM to return a tool call with a rationale
    mock_result = {
        "text": "I need to look at the financial data to make a decision.",
        "tool_calls": [
            {
                "function": {
                    "name": "fetch_financials",
                    "arguments": json.dumps({"ticker": "AAPL"})
                }
            }
        ],
        "total_tokens": 150
    }
    
    # Second turn it finishes
    mock_result_2 = {
        "text": "The financials look great. BUY.",
        "tool_calls": None,
        "total_tokens": 50
    }
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.chat_with_tools = AsyncMock(side_effect=[mock_result, mock_result_2])
    
    # Mock registry so it doesn't try to call real tools
    with patch("app.agents.agent_loop.llm", mock_llm_instance), \
         patch("app.agents.agent_loop.registry.execute_tool_call", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"role": "tool", "name": "fetch_financials", "content": "{\"revenue\": 1000}"}
        
        
        # Mock all the background task and memory functions to avoid hanging
        with patch("app.cognition.evolution.reflector.reflect_on_trajectory", new_callable=AsyncMock), \
             patch("app.cognition.evolution.reflector.get_agent_lessons", return_value=[]), \
             patch("app.cognition.evolution.reflector.get_spotlight_tools", return_value=[]), \
             patch("app.agents.context_compressor.compress_history", new_callable=AsyncMock) as mock_compress:
            
            async def mock_compress_fn(msgs, **kwargs): return msgs
            mock_compress.side_effect = mock_compress_fn
            
            result = await run_agent_loop(
                system_prompt="You are a trading agent.",
                user_prompt="Analyze AAPL.",
                ticker="AAPL",
                agent_name="tester",
                cycle_id="cycle_123"
            )
        
    assert result["stop_reason"] == "success"
    
    # Check that we executed an INSERT into agent_traces
    db_calls = patch_get_db.execute.call_args_list
    trace_inserts = [
        call for call in db_calls if "INSERT INTO agent_traces" in call[0][0]
    ]
    
    assert len(trace_inserts) == 1, "Expected exactly 1 agent trace insert for 1 tool call"
    
    # The second argument to execute is the list of parameters
    params = trace_inserts[0][0][1]
    
    # Verify the rationale (why_tool_was_called) is extracted correctly
    # (id, run_id, agent_name, task_type, goal, planned_next_action, 
    #  tool_name, tool_args, tool_result_summary, why_tool_was_called, 
    #  tokens_before, tokens_after, latency_ms, did_tool_change_decision, 
    #  loop_step, stop_reason)
    
    # run_id
    assert params[1] == "cycle_123"
    # agent_name
    assert params[2] == "tester"
    # tool_name
    assert params[6] == "fetch_financials"
    # why_tool_was_called
    assert params[9] == "I need to look at the financial data to make a decision."


import pytest
from unittest.mock import MagicMock, patch
from app.services.tool_optimizer import optimize_agent_tools, record_tool_optimization_usage, record_run_usage_from_db

@pytest.mark.asyncio
async def test_optimize_agent_tools_filtering():
    mock_db = MagicMock()
    # Mock returning tool1 as active, tool2 as highlighted, tool3 as pruned
    mock_db.fetchall.return_value = [
        ("tool1", "active", 0),
        ("tool2", "highlighted", 2),
        ("tool3", "pruned", 4)
    ]
    
    initial_tools = [
        {"function": {"name": "tool1"}},
        {"function": {"name": "tool2"}},
        {"function": {"name": "tool3"}}
    ]
    
    with patch("app.services.tool_optimizer.get_db") as mock_get_db:
        mock_get_db.return_value.__enter__.return_value = mock_db
        
        filtered, prompt = await optimize_agent_tools(
            agent_name="test_agent",
            initial_tools=initial_tools,
            system_prompt="Base prompt."
        )
        
        # tool3 is pruned, so it should be removed from the returned schemas
        assert len(filtered) == 2
        assert filtered[0]["function"]["name"] == "tool1"
        assert filtered[1]["function"]["name"] == "tool2"
        
        # tool2 is highlighted, so we expect the warning nudge in system prompt
        assert "UNDERUSED TOOLS WARNING" in prompt
        assert "tool2" in prompt
        assert "tool3" not in prompt

@pytest.mark.asyncio
async def test_record_tool_optimization_usage():
    mock_db = MagicMock()
    # Mock query showing tool1 had 1 unused count, tool2 had 3 unused counts
    mock_db.fetchall.return_value = [
        ("tool1", 1, "active"),
        ("tool2", 3, "highlighted")
    ]
    
    offered_tools = [
        {"function": {"name": "tool1"}},
        {"function": {"name": "tool2"}},
        {"function": {"name": "tool3"}}  # new tool, not in DB yet
    ]
    
    # tool1 was used during run, tool2 and tool3 were NOT used
    used_tool_names = ["tool1"]
    
    with patch("app.services.tool_optimizer.get_db") as mock_get_db:
        mock_get_db.return_value.__enter__.return_value = mock_db
        
        await record_tool_optimization_usage(
            agent_name="test_agent",
            offered_tools=offered_tools,
            used_tool_names=used_tool_names
        )
        
        # Verify db insert/update was called
        assert mock_db.execute.call_count > 0

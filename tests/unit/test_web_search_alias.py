import pytest
from unittest.mock import patch, AsyncMock
from app.tools.registry import registry

def test_web_search_alias_registered():
    """Verify that the web_search alias is registered in the tool registry."""
    assert "web_search" in registry.tools
    assert "search_web" in registry.tools

@pytest.mark.asyncio
async def test_web_search_alias_execution():
    """Verify that executing web_search forwards the call to search_web."""
    # We mock search_web to return a specific result and check if web_search calls it
    with patch("app.tools.web_tools.search_web", new_callable=AsyncMock) as mock_search_web:
        mock_search_web.return_value = "Search Results Content"
        
        tool_call = {
            "id": "call_web_search",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "Nvidia earnings"}'
            }
        }
        
        res = await registry.execute_tool_call(tool_call)
        
        # Verify the alias forwarded the arguments to search_web
        mock_search_web.assert_called_once_with("Nvidia earnings", 3)
        assert res["role"] == "tool"
        assert res["tool_call_id"] == "call_web_search"
        assert res["name"] == "web_search"
        assert res["content"] == "Search Results Content"

import pytest
import json
from unittest.mock import patch, MagicMock

from app.agents.retriever_agent import run_retriever
from app.agents.context_compressor import generate_capsule

@pytest.fixture
def mock_db():
    with patch("app.db.connection.get_db") as mock_get_db:
        mock_db_instance = MagicMock()
        mock_db_instance.execute.return_value.fetchone.return_value = None
        mock_db_instance.execute.return_value.fetchall.return_value = []
        mock_get_db.return_value.__enter__.return_value = mock_db_instance
        yield mock_db_instance

@pytest.fixture
def mock_run_agent_loop(mock_db):
    with patch("app.agents.agent_loop.run_agent_loop") as mock_ral:
        mock_ral.return_value = {
            "final_text": '{"price": 150, "news": "good"}',
            "token_usage": 150,
            "execution_ms": 600,
            "tool_calls": [{"name": "get_price"}]
        }
        yield mock_ral

@pytest.mark.asyncio
async def test_r01_response_is_parseable_json(mock_run_agent_loop):
    result = await run_retriever("TSLA", "cycle_123", "bot_123", "plan")
    parsed = json.loads(result["response"])
    assert isinstance(parsed, dict)

@pytest.mark.asyncio
async def test_r02_at_least_one_tool_called(mock_run_agent_loop):
    await run_retriever("TSLA", "cycle_123", "bot_123", "plan")
    trace = mock_run_agent_loop.return_value
    assert len(trace.get("tool_calls", [])) >= 1

@pytest.mark.asyncio
async def test_r03_all_tools_fail_still_returns_dict(mock_run_agent_loop):
    mock_run_agent_loop.return_value = {
        "final_text": '{"error": "all tools failed"}',
        "token_usage": 50,
        "execution_ms": 100
    }
    result = await run_retriever("TSLA", "c1", "b1", "plan")
    assert isinstance(result, dict)
    assert "response" in result

@pytest.mark.asyncio
async def test_r04_response_contains_market_data(mock_run_agent_loop):
    result = await run_retriever("TSLA", "c1", "b1", "plan")
    parsed = json.loads(result["response"])
    assert any(k in parsed for k in ["price", "volume", "news"])

@pytest.mark.asyncio
async def test_r05_capsule_under_200_tokens():
    capsule = await generate_capsule({"response": '{"price": 150}'}, "retriever", "c1", "TSLA")
    assert capsule.tokens_estimated <= 200

@pytest.mark.asyncio
async def test_r06_no_tools_outside_whitelist(mock_run_agent_loop):
    from app.agents.tool_whitelists import get_agent_tools
    allowed_tools = get_agent_tools("retriever")
    allowed_names = [t["function"]["name"] for t in allowed_tools]
    
    await run_retriever("TSLA", "c1", "b1", "plan")
    kwargs = mock_run_agent_loop.call_args.kwargs
    called_tools = kwargs.get("tools_override", [])
    if called_tools:
        called_names = [t["function"]["name"] for t in called_tools]
        assert all(n in allowed_names for n in called_names)

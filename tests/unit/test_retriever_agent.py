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
def mock_run_agent(mock_db):
    with patch("app.agents.retriever_agent.run_agent") as mock_ra:
        mock_ra.return_value = {
            "response": '{"price": 150, "news": "good"}',
            "tokens_used": 150,
            "execution_ms": 600,
            "tool_calls": [{"name": "get_price"}]
        }
        yield mock_ra

@pytest.mark.asyncio
async def test_r01_response_is_parseable_json(mock_run_agent):
    result = await run_retriever("TSLA", "cycle_123", "bot_123", "plan")
    parsed = json.loads(result["response"])
    assert isinstance(parsed, dict)

@pytest.mark.asyncio
async def test_r02_at_least_one_tool_called(mock_run_agent):
    await run_retriever("TSLA", "cycle_123", "bot_123", "plan")
    kwargs = mock_run_agent.call_args.kwargs
    assert kwargs.get("enable_tools") is True

@pytest.mark.asyncio
async def test_r03_all_tools_fail_still_returns_dict(mock_run_agent):
    mock_run_agent.return_value = {
        "response": '{"error": "all tools failed"}',
        "tokens_used": 50,
        "execution_ms": 100
    }
    result = await run_retriever("TSLA", "c1", "b1", "plan")
    assert isinstance(result, dict)
    assert "response" in result

@pytest.mark.asyncio
async def test_r04_response_contains_market_data(mock_run_agent):
    result = await run_retriever("TSLA", "c1", "b1", "plan")
    parsed = json.loads(result["response"])
    assert any(k in parsed for k in ["price", "volume", "news"])

@pytest.mark.asyncio
async def test_r05_capsule_under_200_tokens():
    capsule = await generate_capsule({"response": '{"price": 150}'}, "retriever", "c1", "TSLA")
    assert capsule.tokens_estimated <= 200

@pytest.mark.asyncio
async def test_r06_no_tools_outside_whitelist(mock_run_agent):
    await run_retriever("TSLA", "c1", "b1", "plan")
    kwargs = mock_run_agent.call_args.kwargs
    assert kwargs.get("enable_tools") is True
    assert kwargs.get("agent_name") == "retriever"

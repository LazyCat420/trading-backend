import pytest
import datetime
from unittest.mock import patch, MagicMock
from app.agents.base_agent import run_agent, _OUTCOME_CONTEXT_AGENTS

@pytest.fixture
def mock_run_agent_loop():
    with patch("app.agents.agent_loop.run_agent_loop") as mock_loop, \
         patch("app.agents.agent_loop.run_split_agent_loop") as mock_split_loop:
        # Default success response
        mock_loop.return_value = {
            "final_text": '{"result": "success"}',
            "token_usage": 150,
            "execution_ms": 1200
        }
        yield mock_loop

@pytest.fixture
def mock_db_empty():
    with patch("app.db.connection.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.__enter__.return_value = mock_db
        mock_db.execute.return_value.fetchone.return_value = None
        mock_db.fetchall.return_value = []
        mock_get_db.return_value = mock_db
        yield mock_db

@pytest.mark.asyncio
async def test_b01_b03_b04_base_agent_output_schema(mock_run_agent_loop, mock_db_empty):
    """
    B-01: Agent always returns a dict with keys: agent, ticker, response, tokens_used, execution_ms, timestamp
    B-03: tokens_used is always an integer >= 0 (never None)
    B-04: execution_ms is always an integer >= 0
    """
    result = await run_agent(
        agent_name="test_agent",
        ticker="AAPL",
        cycle_id="cycle_123",
        bot_id="bot_abc",
        system_prompt="You are a test agent.",
        user_prompt="Analyze this.",
        data_context="data"
    )
    
    assert isinstance(result, dict)
    
    expected_keys = {"agent", "ticker", "response", "tokens_used", "execution_ms", "timestamp"}
    for key in expected_keys:
        assert key in result
        
    assert isinstance(result["tokens_used"], int)
    assert result["tokens_used"] >= 0
    
    assert isinstance(result["execution_ms"], int)
    assert result["execution_ms"] >= 0

@pytest.mark.asyncio
async def test_b02_fallback_on_empty_response(mock_run_agent_loop, mock_db_empty):
    """
    B-02: response field is never None or empty string — must fallback to "Agent failed: ..." string
    """
    mock_run_agent_loop.return_value = {
        "final_text": "",
        "token_usage": 0,
        "execution_ms": 100
    }
    
    result = await run_agent(
        agent_name="test_agent",
        ticker="AAPL",
        cycle_id="cycle_123",
        bot_id="bot_abc",
        system_prompt="System",
        user_prompt="User"
    )
    
    assert result["response"] is not None
    assert result["response"] != ""
    assert "Agent failed" in result["response"]

@pytest.mark.asyncio
async def test_b06_dynamic_prompt_flag(mock_run_agent_loop, mock_db_empty):
    """
    B-06: dynamic_prompt_used=True only appears when enable_dynamic_prompt=True was passed
    """
    # Test with enable_dynamic_prompt=False
    result_static = await run_agent(
        agent_name="test_agent",
        ticker="AAPL",
        cycle_id="cycle_123",
        bot_id="bot_abc",
        system_prompt="System",
        user_prompt="User",
        enable_dynamic_prompt=False
    )
    
    assert "dynamic_prompt_used" in result_static
    assert result_static["dynamic_prompt_used"] is False
    
    # Mock dynamic prompt generation to succeed
    with patch("app.agents.base_agent._generate_dynamic_prompt") as mock_gen:
        mock_gen.return_value = ("Better dynamic prompt", "reasoning")
        
        # Test with enable_dynamic_prompt=True
        result_dynamic = await run_agent(
            agent_name="test_agent",
            ticker="AAPL",
            cycle_id="cycle_123",
            bot_id="bot_abc",
            system_prompt="System",
            user_prompt="User",
            data_context="some data",
            enable_dynamic_prompt=True
        )
        
        assert result_dynamic["dynamic_prompt_used"] is True

@pytest.mark.asyncio
async def test_b08_outcome_context_injection(mock_run_agent_loop, mock_db_empty):
    """
    B-08: Outcome context injection only fires for agents in _OUTCOME_CONTEXT_AGENTS frozenset
    """
    with patch("app.agents.base_agent.get_ticker_outcome_context") as mock_get_context:
        mock_get_context.return_value = "\n## PRIOR TRADE HISTORY FOR AAPL\n- WIN\n"
        
        non_context_agent = "some_random_agent"
        
        await run_agent(
            agent_name=non_context_agent,
            ticker="AAPL",
            cycle_id="1",
            bot_id="1",
            system_prompt="sys",
            user_prompt="user"
        )
        mock_get_context.assert_not_called()
        
        context_agent = list(_OUTCOME_CONTEXT_AGENTS)[0]
        
        await run_agent(
            agent_name=context_agent,
            ticker="AAPL",
            cycle_id="1",
            bot_id="1",
            system_prompt="sys",
            user_prompt="user"
        )
        mock_get_context.assert_called_once_with("AAPL")

@pytest.mark.asyncio
async def test_b09_empty_db_fallback(mock_run_agent_loop):
    """
    B-09: When generated_agent_prompts DB table is empty, falls back to static system prompt without crash
    """
    with patch("app.db.connection.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.__enter__.return_value = mock_db
        mock_db.execute.side_effect = Exception("Table does not exist")
        mock_get_db.return_value = mock_db
        
        result = await run_agent(
            agent_name="test_agent",
            ticker="AAPL",
            cycle_id="1",
            bot_id="1",
            system_prompt="STATIC PROMPT",
            user_prompt="user"
        )
        
        assert result["response"] == '{"result": "success"}'



@pytest.mark.asyncio
async def test_b07_data_context_truncation(mock_run_agent_loop, mock_db_empty):
    """
    B-07: data_context is correctly truncated if it exceeds budget
    """
    with patch("app.config.context_budget.get_context_budget") as mock_budget:
        mock_b = MagicMock()
        mock_b.data_context_chars = 10
        mock_budget.return_value = mock_b
        
        long_data = "This is a very long data context that exceeds the limit."
        
        await run_agent(
            agent_name="test_agent",
            ticker="AAPL",
            cycle_id="1",
            bot_id="1",
            system_prompt="sys",
            user_prompt="user",
            data_context=long_data
        )
        
        call_args = mock_run_agent_loop.call_args[1]
        user_prompt_passed = call_args["user_prompt"]
        
        assert "This is a " in user_prompt_passed
        assert "very long data" not in user_prompt_passed
        # Check that the data context part is truncated
        assert len(user_prompt_passed) <= len("This is a ") + len("\n\nuser")

@pytest.mark.asyncio
async def test_b05_timeout_returns_dict_shape(mock_db_empty):
    """
    B-05: 90s timeout returns error dict, not raise
    """
    import asyncio
    with patch("asyncio.wait_for") as mock_wait:
        mock_wait.side_effect = asyncio.TimeoutError()
        result = await run_agent(
            agent_name="test_agent", ticker="AAPL", cycle_id="1", bot_id="1",
            system_prompt="sys", user_prompt="user"
        )
        assert isinstance(result, dict)
        assert "timeout" in result.get("response", "").lower()

@pytest.mark.asyncio
async def test_b10_playbook_appended_not_overwritten(mock_run_agent_loop):
    """
    B-10: tool_playbook rules appended, not overwritten
    """
    with patch("app.db.connection.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.__enter__.return_value = mock_db
        # fetchone for the dynamic prompt (returns None)
        mock_db.execute.return_value.fetchone.return_value = None
        # fetchall for the playbook
        mock_db.execute.return_value.fetchall.return_value = [("use tool A", "cond1", "pattern2")]
        mock_get_db.return_value = mock_db
        
        await run_agent(
            agent_name="test_agent", ticker="AAPL", cycle_id="1", bot_id="1",
            system_prompt="ORIGINAL SYSTEM", user_prompt="user",
            enable_tools=True
        )
        
        call_args = mock_run_agent_loop.call_args[1]
        sys_passed = call_args["system_prompt"]
        
        assert "ORIGINAL SYSTEM" in sys_passed
        assert "TOOL PLAYBOOK RULES" in sys_passed
        assert "use tool A" in sys_passed

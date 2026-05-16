import sys
import os
import pytest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agents.planner_agent import run_planner
from app.agents.retriever_agent import run_retriever
from app.agents.verifier_agent import run_verifier
from app.pipeline.analysis.agent_execution import run_specialist_agents

pytestmark = pytest.mark.asyncio

class TestPlannerAgent:
    @patch("app.agents.planner_agent.run_agent", new_callable=AsyncMock)
    async def test_planner_agent_run(self, mock_run_agent):
        mock_run_agent.return_value = {"response": "plan JSON"}
        
        result = await run_planner(
            ticker="AAPL",
            cycle_id="test_cycle",
            bot_id="test_bot"
        )
        
        assert result["response"] == "plan JSON"
        mock_run_agent.assert_called_once()
        kwargs = mock_run_agent.call_args.kwargs
        assert kwargs["agent_name"] == "planner"
        assert kwargs["enable_tools"] is False
        assert "Create an evidence gathering plan for AAPL." in kwargs["user_prompt"]

    @patch("app.agents.planner_agent.run_agent", new_callable=AsyncMock)
    async def test_planner_ontology_injection(self, mock_run_agent):
        mock_run_agent.return_value = {"response": "plan JSON"}
        
        await run_planner(
            ticker="AAPL",
            cycle_id="test_cycle",
            bot_id="test_bot",
            ontology_context="AAPL is in tech sector."
        )
        
        kwargs = mock_run_agent.call_args.kwargs
        assert "KNOWN CONTEXT" in kwargs["user_prompt"]
        assert "AAPL is in tech sector." in kwargs["user_prompt"]

class TestRetrieverAgent:
    @patch("app.agents.retriever_agent.run_agent", new_callable=AsyncMock)
    async def test_retriever_agent_run(self, mock_run_agent):
        mock_run_agent.return_value = {"response": "retriever JSON"}
        
        result = await run_retriever(
            ticker="AAPL",
            cycle_id="test_cycle",
            bot_id="test_bot",
            capsule_context="Plan capsule"
        )
        
        assert result["response"] == "retriever JSON"
        mock_run_agent.assert_called_once()
        kwargs = mock_run_agent.call_args.kwargs
        assert kwargs["agent_name"] == "retriever"
        assert kwargs["enable_tools"] is True
        assert "Execute this plan for AAPL" in kwargs["user_prompt"]
        assert "Plan capsule" in kwargs["user_prompt"]

class TestVerifierAgent:
    @patch("app.agents.verifier_agent.run_agent", new_callable=AsyncMock)
    async def test_verifier_agent_run(self, mock_run_agent):
        mock_run_agent.return_value = {"response": "verifier JSON"}
        
        result = await run_verifier(
            ticker="AAPL",
            cycle_id="test_cycle",
            bot_id="test_bot",
            capsule_context="Retrieved data capsule"
        )
        
        assert result["response"] == "verifier JSON"
        mock_run_agent.assert_called_once()
        kwargs = mock_run_agent.call_args.kwargs
        assert kwargs["agent_name"] == "verifier"
        assert kwargs["enable_tools"] is True
        assert "Verify this evidence for AAPL" in kwargs["user_prompt"]
        assert "Retrieved data capsule" in kwargs["user_prompt"]



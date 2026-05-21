import pytest
import json
from unittest.mock import patch, MagicMock

from app.agents.planner_agent import run_planner
from app.pipeline.analysis.agent_execution import run_specialist_agents

@pytest.fixture
def mock_run_agent():
    with patch("app.agents.planner_agent.run_agent") as mock_run:
        mock_run.return_value = {
            "response": '{"plan": ["gather price", "gather news"]}',
            "agent": "planner",
            "ticker": "TSLA"
        }
        yield mock_run

@pytest.fixture
def mock_build_relationship_map():
    with patch("app.graph.graph_queries.build_relationship_map") as mock_brm:
        mock_brm.return_value = {"ontology_context": "test context"}
        yield mock_brm

@pytest.fixture
def mock_write_capsule_to_db():
    with patch("app.agents.context_compressor.write_capsule_to_db") as mock_wctd:
        yield mock_wctd

@pytest.mark.asyncio
async def test_p01_response_is_parseable_json(mock_run_agent):
    result = await run_planner("TSLA", "cycle_123", "bot_123")
    parsed = json.loads(result["response"])
    assert isinstance(parsed, dict)

@pytest.mark.asyncio
async def test_p02_plan_array_nonempty(mock_run_agent):
    result = await run_planner("TSLA", "cycle_123", "bot_123")
    parsed = json.loads(result["response"])
    assert len(parsed.get("plan", [])) >= 1

@pytest.mark.asyncio
async def test_p03_ontology_failure_graceful_fallback(mock_build_relationship_map):
    mock_build_relationship_map.side_effect = Exception("Ontology down")
    
    with patch("app.pipeline.analysis.agent_execution._run_with_timeout") as mock_run:
        mock_run.return_value = {"response": '{"plan": []}', "agent": "planner"}
        with patch("app.agents.context_compressor.generate_capsule") as mock_gen:
            mock_capsule = MagicMock()
            mock_capsule.tokens_estimated = 100
            mock_gen.return_value = mock_capsule
            with patch("app.agents.context_compressor.write_capsule_to_db"):
                result = await run_specialist_agents("TSLA", "c_1", "b_1")
                assert result is not None
                assert "planner" in result

@pytest.mark.asyncio
async def test_p04_tools_disabled(mock_run_agent):
    await run_planner("TSLA", "cycle_123", "bot_123")
    assert mock_run_agent.call_args.kwargs.get("enable_tools") is False

@pytest.mark.asyncio
async def test_p05_capsule_written_to_db(mock_write_capsule_to_db):
    with patch("app.agents.planner_agent.run_planner") as mock_run_planner:
        mock_run_planner.return_value = {"response": '{"plan": []}', "agent": "planner"}
        with patch("app.agents.retriever_agent.run_retriever") as mock_run_retriever:
            mock_run_retriever.return_value = {"response": '{"data": "none"}', "agent": "retriever"}
            with patch("app.agents.verifier_agent.run_verifier") as mock_run_verifier:
                mock_run_verifier.return_value = {"response": '{"verified": true}', "agent": "verifier"}
                with patch("app.agents.base_agent.run_agent") as mock_run_synth:
                    mock_run_synth.return_value = {"response": '{"signal": "HOLD"}', "agent": "synthesizer"}
                    with patch("app.graph.graph_queries.build_relationship_map"):
                        await run_specialist_agents("TSLA", "c_1", "b_1")
    
    assert mock_write_capsule_to_db.called

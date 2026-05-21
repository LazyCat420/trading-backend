import pytest
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.fixture
def mock_run_agent():
    with patch("app.agents.base_agent.run_agent") as mock_agent:
        async def mock_run(*args, **kwargs):
            return {"response": '{"signal": "HOLD", "confidence": 50, "rationale": "test"}'}
        mock_agent.side_effect = mock_run
        yield mock_agent

@pytest.fixture
def mock_generate_capsule():
    with patch("app.agents.context_compressor.generate_capsule") as mock_cap:
        from app.agents.capsule import AgentCapsule
        async def mock_run(*args, **kwargs):
            return AgentCapsule(
                agent_name="mock_agent", cycle_id="1", ticker="AAPL",
                summary="test", signal="HOLD", confidence=0.5,
                tokens_estimated=10
            )
        mock_cap.side_effect = mock_run
        yield mock_cap

@pytest.fixture
def mock_write_capsule():
    with patch("app.agents.context_compressor.write_capsule_to_db") as mock_write:
        async def mock_run(*args, **kwargs):
            pass
        mock_write.side_effect = mock_run
        yield mock_write

@pytest.fixture
def mock_run_planner():
    with patch("app.agents.planner_agent.run_planner") as mock_p:
        async def mock_run(*args, **kwargs):
            return {"response": "planned"}
        mock_p.side_effect = mock_run
        yield mock_p

@pytest.fixture
def mock_run_retriever():
    with patch("app.agents.retriever_agent.run_retriever") as mock_r:
        async def mock_run(*args, **kwargs):
            return {"response": "retrieved"}
        mock_r.side_effect = mock_run
        yield mock_r

@pytest.fixture
def mock_run_verifier():
    with patch("app.agents.verifier_agent.run_verifier") as mock_v:
        async def mock_run(*args, **kwargs):
            return {"response": "verified"}
        mock_v.side_effect = mock_run
        yield mock_v

@pytest.fixture
def mock_build_relationship_map():
    with patch("app.graph.graph_queries.build_relationship_map") as mock_b:
        async def mock_run(*args, **kwargs):
            return {"ontology_context": ""}
        mock_b.side_effect = mock_run
        yield mock_b

@pytest.mark.asyncio
async def test_preserves_quantitative_facts(
    mock_run_agent, 
    mock_run_planner, mock_run_retriever, mock_run_verifier,
    mock_generate_capsule, mock_write_capsule, mock_build_relationship_map
):
    """
    Test that the Synthesizer's system prompt strictly requires it to preserve
    hard quantitative facts from upstream capsules.
    """
    from app.pipeline.analysis.agent_execution import run_specialist_agents
    
    await run_specialist_agents("AAPL", "cycle_1", "bot_1")
    
    # Extract the call to the synthesizer
    synth_call = [call for call in mock_run_agent.call_args_list if call.kwargs.get("agent_name") == "synthesizer"][0]
    system_prompt = synth_call.kwargs.get("system_prompt")
    
    assert "ANTI-DEGRADATION RULE" in system_prompt, "Synthesizer is missing the anti-degradation rule in its prompt."
    assert "preserve hard quantitative facts" in system_prompt, "Synthesizer is not explicitly instructed to preserve facts."

@pytest.mark.asyncio
async def test_contradiction_surfacing(
    mock_run_agent, 
    mock_run_planner, mock_run_retriever, mock_run_verifier,
    mock_generate_capsule, mock_write_capsule, mock_build_relationship_map
):
    """
    Test that the Synthesizer's system prompt strictly requires it to surface
    contradictions between upstream reports rather than averaging them.
    """
    from app.pipeline.analysis.agent_execution import run_specialist_agents
    
    await run_specialist_agents("AAPL", "cycle_1", "bot_1")
    
    synth_call = [call for call in mock_run_agent.call_args_list if call.kwargs.get("agent_name") == "synthesizer"][0]
    system_prompt = synth_call.kwargs.get("system_prompt")
    
    assert "CONTRADICTION RULE" in system_prompt, "Synthesizer is missing the contradiction rule in its prompt."
    assert "explicitly surface and explain the contradiction" in system_prompt, "Synthesizer is not instructed to surface conflicts."
    assert "Do NOT average them out" in system_prompt, "Synthesizer is missing the guardrail against averaging conflicts."

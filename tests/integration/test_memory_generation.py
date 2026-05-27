import pytest
from unittest.mock import MagicMock
import asyncio

from app.pipeline.phases.phase6_post import run_phase6_post
from app.pipeline.core import PipelineContext

@pytest.mark.asyncio
async def test_memory_generation_on_cycle_close(monkeypatch, mock_db):
    """Verify that semantic and episodic memories are persisted when a cycle finishes."""
    import app.pipeline.analysis.purge_pass
    import app.cognition.ontology.knowledge_purge
    import app.pipeline.analysis.agent_maintenance
    import app.services.bot_manager
    import app.cycle.phases.phase6_post
    import app.trading.paper_trader
    import app.pipeline.subsystem_benchmarks
    import app.agents.meta_audit_agent
    import app.agents.quant_research_agent

    from contextlib import contextmanager

    @contextmanager
    def mock_get_db():
        yield mock_db

    # Systematically patch get_db in all loaded modules that have it
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod and hasattr(mod, "get_db"):
            try:
                monkeypatch.setattr(f"{mod_name}.get_db", mock_get_db)
            except Exception:
                pass

    # Patch get_db at the source
    monkeypatch.setattr("app.db.connection.get_db", mock_get_db)
    monkeypatch.setattr("app.cycle.phases.phase6_post.get_db", mock_get_db)
    monkeypatch.setattr("app.trading.paper_trader.get_db", mock_get_db)
    
    # Mock bot manager get_active_bot_id
    monkeypatch.setattr("app.services.bot_manager.get_active_bot_id", lambda: "test-bot")

    # Mock the return values for DB queries
    def mock_execute(query, *args, **kwargs):
        cursor = MagicMock()
        if "SELECT cash FROM bots" in query:
            cursor.fetchone.return_value = [10000.0]
        elif "SELECT close FROM price_history" in query:
            cursor.fetchone.return_value = [150.0]
        elif "SELECT result_json FROM analysis_results" in query:
            cursor.fetchone.return_value = ['{}']
        else:
            cursor.fetchall.return_value = []
            cursor.fetchone.return_value = None
        return cursor

    mock_db.execute.side_effect = mock_execute

    # Mock out the purge passes and subsystems to speed up tests and avoid side effects
    async def mock_purge_pass(*args, **kwargs):
        return []
        
    async def mock_purge_knowledge(*args, **kwargs):
        return {"nodes": 0, "edges": 0}
        
    async def mock_maintenance(*args, **kwargs):
        pass

    async def mock_agent(*args, **kwargs):
        pass

    monkeypatch.setattr("app.cycle.phases.phase6_post.run_purge_pass", mock_purge_pass)
    monkeypatch.setattr("app.cycle.phases.phase6_post.purge_stale_knowledge", mock_purge_knowledge)
    monkeypatch.setattr("app.cycle.phases.phase6_post.run_janitor_tasks", mock_maintenance)
    monkeypatch.setattr("app.cycle.phases.phase6_post.record_all", lambda *a, **kw: None)
    monkeypatch.setattr("app.cycle.phases.phase6_post.run_meta_audit", mock_agent)
    monkeypatch.setattr("app.cycle.phases.phase6_post.run_quant_research", mock_agent)

    # Mock get_portfolio to avoid NoneType error
    monkeypatch.setattr("app.cycle.phases.phase6_post._get_pf", lambda x: {"cash": 10000.0})

    ctx = PipelineContext(
        tickers=["AAPL"],
        collect=True,
        analyze=True,
        trade=True,
        cycle_id="test-cycle-123",
    )

    results = [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 80,
            "rationale": "Strong earnings report and MACD crossover.",
            "trade_executed": {
                "fill_qty": 10,
                "fill_price": 150.0
            }
        }
    ]

    trade_result = {
        "executed": [
            {
                "ticker": "AAPL",
                "fill_qty": 10,
                "fill_price": 150.0
            }
        ],
        "skipped": []
    }

    state = {
        "execution_mode": "v1_production"
    }

    emit = MagicMock()

    await run_phase6_post(ctx, "test-bot", results, trade_result, emit, state)

    # Verify semantic memory was inserted
    semantic_inserts = [c for c in mock_db.execute.call_args_list if "INSERT INTO semantic_memory" in str(c)]
    assert len(semantic_inserts) == 1
    assert "thesis_insight" in semantic_inserts[0].args[1]  # mem_type
    assert "Strong earnings report" in semantic_inserts[0].args[1][3]  # content

    # Verify episodic memory was inserted
    episodic_inserts = [c for c in mock_db.execute.call_args_list if "INSERT INTO episodic_memory" in str(c)]
    assert len(episodic_inserts) == 1
    assert "AAPL" in episodic_inserts[0].args[1]
    assert "Strong earnings report" in episodic_inserts[0].args[1][4]  # summary

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.analyze = True
    ctx.trade = True
    ctx.cycle_id = "test_cycle"
    ctx.tickers = ["AAPL"]
    return ctx


@pytest.mark.asyncio
@patch("app.cycle.phases.phase6_post.get_db")
@patch("app.services.memory.cycle_closer.cycle_closer.close_cycle", new_callable=AsyncMock)
@patch("app.trading.paper_trader.get_portfolio")
@patch("app.cycle.trading_phase.estimate_trade")
@patch("app.pipeline.analysis.purge_pass.run_purge_pass", new_callable=AsyncMock)
@patch("app.cognition.ontology.knowledge_purge.purge_stale_knowledge", new_callable=AsyncMock)
@patch("app.pipeline.analysis.agent_maintenance.run_janitor_tasks", new_callable=AsyncMock)
@patch("app.pipeline.subsystem_benchmarks.record_all")
@patch("app.agents.meta_audit_agent.run_meta_audit", new_callable=AsyncMock)
@patch("app.agents.quant_research_agent.run_quant_research", new_callable=AsyncMock)
@patch("app.services.bot_manager.get_active_bot_id", return_value="test_bot")
async def test_run_phase6_post_success(
    mock_bot_id, mock_quant, mock_audit, mock_record,
    mock_janitor, mock_purge_stale, mock_purge_pass,
    mock_estimate, mock_get_pf, mock_close_cycle, mock_get_db, mock_ctx
):
    from app.pipeline.phases.phase6_post import run_phase6_post

    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db

    mock_get_pf.return_value = {"cash": 10000}
    mock_estimate.return_value = {"qty": 10, "cost": 1000}

    # Mock DB fetchall for existing analysis_results
    mock_db.execute.return_value.fetchall.side_effect = [
        [("AAPL", json.dumps({"ticker": "AAPL", "action": "BUY"}))],  # existing_rows
        [("AAPL", 150.0)]  # price_rows
    ]

    results = [{"ticker": "AAPL", "action": "BUY", "confidence": 80}]
    trade_result = {"executed": [{"ticker": "AAPL", "qty": 10}], "skipped": []}

    state = {}
    emit = MagicMock()

    await run_phase6_post(mock_ctx, "bot1", results, trade_result, emit, state)

    # Verify memory closer was called
    mock_close_cycle.assert_called_once()

    # Verify DB update was executed (enrichment)
    assert mock_db.executemany.called

    # Verify maintenance routines were called
    mock_purge_pass.assert_called_once()
    mock_purge_stale.assert_called_once()
    mock_janitor.assert_called_once()

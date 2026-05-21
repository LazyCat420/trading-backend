import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.pipeline.phases.phase1_health import run_phase1_health

@pytest.fixture
def mock_emit():
    return MagicMock()

@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.cycle_id = "test-cycle"
    ctx.tickers = ["AAPL", "MSFT"]
    ctx.trade = True
    return ctx

@pytest.fixture
def mock_state():
    return {"position_tickers": ["AAPL"]}

@pytest.fixture
def mock_cycle_summary():
    return {}

@pytest.mark.asyncio
@patch("app.services.vllm_client.llm")
async def test_run_phase1_health_all_bots_down(mock_llm, mock_ctx, mock_emit, mock_cycle_summary, mock_state):
    # Both jetson and dgx are False
    mock_llm.health_all = AsyncMock(return_value={"jetson": False, "dgx_spark_1": False})
    
    with pytest.raises(RuntimeError, match="All LLM endpoints unreachable"):
        await run_phase1_health(mock_ctx, "bot123", mock_emit, mock_cycle_summary, mock_state)
    
    assert mock_cycle_summary["no_trade_reason"] == "all_bots_down"

@pytest.mark.asyncio
@patch("app.services.vllm_client.llm")
@patch("app.trading.paper_trader.check_stop_losses")
@patch("app.trading.paper_trader.check_take_profits")
@patch("app.cycle.phases.phase1_health.get_db")
async def test_run_phase1_health_success(mock_get_db, mock_ctp, mock_csl, mock_llm, mock_ctx, mock_emit, mock_cycle_summary, mock_state):
    mock_llm.health_all = AsyncMock(return_value={"jetson": True, "dgx_spark_1": True})
    
    mock_csl.return_value = [{"ticker": "TSLA"}]
    mock_ctp.return_value = []
    
    mock_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db
    
    # Mock position/watchlist rows
    # Position: AAPL, TSLA
    # Watchlist: AAPL
    # Orphan: TSLA
    def execute_mock(query, *args, **kwargs):
        cur = MagicMock()
        if "FROM positions" in query:
            cur.fetchall.return_value = [("AAPL",), ("TSLA",)]
        elif "FROM watchlist" in query:
            if "source = 'user'" in query:
                cur.fetchall.return_value = []
            else:
                cur.fetchall.return_value = [("AAPL",)]
        elif "SELECT directive_type" in query:
            cur.fetchall.return_value = []
        return cur
    mock_db.execute.side_effect = execute_mock
    
    with patch("app.pipeline.phases.phase1_health.settings") as mock_settings:
        mock_settings.TRIAGE_ENABLED = False
        
        await run_phase1_health(mock_ctx, "bot123", mock_emit, mock_cycle_summary, mock_state)
    
    assert mock_cycle_summary["jetson_healthy_start"] is True
    assert mock_cycle_summary["stop_loss_triggered"] == 1
    assert "TSLA" in mock_cycle_summary["stop_loss_tickers"]
    
    # Check that TSLA was added to tickers since it was an orphan
    assert "TSLA" in mock_ctx.tickers

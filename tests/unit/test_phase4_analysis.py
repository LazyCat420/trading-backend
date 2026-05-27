import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from app.pipeline.phases.phase4_analysis import run_phase4_analysis

@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.analyze = True
    ctx.tickers = ["AAPL", "NVDA"]
    ctx.cycle_id = "test_cycle"
    return ctx

@pytest.mark.asyncio
@patch("app.cycle.phases.phase4_analysis.execute_v2_pipeline")
async def test_run_phase4_analysis_success(mock_execute, mock_ctx):
    mock_execute.side_effect = [
        {"ticker": "AAPL", "action": "BUY", "confidence": 0.8},
        {"ticker": "NVDA", "action": "HOLD", "confidence": 0.5}
    ]
    
    cycle_summary = {"buy_count": 0, "sell_count": 0, "hold_count": 0, "review_count": 0}
    state = {}
    emit = MagicMock()
    
    results = await run_phase4_analysis(mock_ctx, "bot1", "macro", emit, cycle_summary, state)
    
    assert len(results) == 2
    assert cycle_summary["buy_count"] == 1
    assert cycle_summary["hold_count"] == 1
    
@pytest.mark.asyncio
@patch("app.cycle.phases.phase4_analysis.execute_v2_pipeline")
@patch("app.cycle.phases.phase4_analysis.settings.ANALYSIS_WORKER_TIMEOUT_SECONDS", 0.1)
async def test_run_phase4_analysis_timeout_fallback(mock_execute, mock_ctx):
    # Set tickers to at least 3 to trigger the all-crash gate abort
    mock_ctx.tickers = ["AAPL", "NVDA", "MSFT"]
    
    # Make the execute function sleep to force a timeout
    async def slow_execute(*args, **kwargs):
        await asyncio.sleep(0.5)
        return {}
        
    mock_execute.side_effect = slow_execute
    
    cycle_summary = {"buy_count": 0, "sell_count": 0, "hold_count": 0, "review_count": 0}
    state = {}
    emit = MagicMock()
    
    with pytest.raises(RuntimeError, match="All 3 tickers crashed"):
        await run_phase4_analysis(mock_ctx, "bot1", "macro", emit, cycle_summary, state)

@pytest.mark.asyncio
@patch("app.cycle.phases.phase4_analysis.execute_v2_pipeline")
async def test_run_phase4_analysis_zero_results(mock_execute, mock_ctx):
    # Return empty lists or None
    mock_execute.return_value = None
    
    cycle_summary = {"buy_count": 0, "sell_count": 0, "hold_count": 0, "review_count": 0}
    state = {}
    emit = MagicMock()
    
    with pytest.raises(RuntimeError, match="Analysis produced zero results"):
        await run_phase4_analysis(mock_ctx, "bot1", "macro", emit, cycle_summary, state)

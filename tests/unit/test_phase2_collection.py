import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.pipeline.phases.phase2_collection import run_phase2_collection

@pytest.fixture
def mock_emit():
    return MagicMock()

@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.tickers = ["AAPL", "MSFT"]
    return ctx

@pytest.fixture
def mock_state():
    return {}

@pytest.mark.asyncio
@patch("app.cycle.phases.phase2_collection.run_data")
async def test_run_phase2_collection_success(mock_run_data, mock_ctx, mock_emit, mock_state):
    # Mock data_results
    mock_run_data.return_value = {
        "tickers": ["AAPL", "MSFT", "NVDA"],
        "collected_count": 2
    }
    
    result = await run_phase2_collection(mock_ctx, mock_emit, mock_state)
    
    assert result == ["AAPL", "MSFT", "NVDA"]
    assert mock_ctx.tickers == ["AAPL", "MSFT", "NVDA"]
    assert mock_state["data_coverage_pct"] == round((2 / 3) * 100, 1)

@pytest.mark.asyncio
@patch("app.cycle.phases.phase2_collection.run_data")
async def test_run_phase2_collection_exception_handling(mock_run_data, mock_ctx, mock_emit, mock_state):
    # Mock exception
    mock_run_data.side_effect = Exception("API timeout")
    
    # Should not raise, should return original tickers gracefully
    result = await run_phase2_collection(mock_ctx, mock_emit, mock_state)
    
    assert result == ["AAPL", "MSFT"]
    mock_emit.assert_any_call("collecting", "fatal", "Collection crashed: API timeout", status="error")

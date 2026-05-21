import pytest
import asyncio
from unittest.mock import patch, MagicMock

from app.pipeline.phases.phase3_macro import run_phase3_macro

@pytest.fixture
def mock_emit():
    return MagicMock()

@pytest.mark.asyncio
@patch("app.cycle.phases.phase3_macro.settings")
async def test_run_phase3_macro_disabled(mock_settings, mock_emit):
    mock_settings.MACRO_SCOUT_ENABLED = False
    
    result = await run_phase3_macro(mock_emit)
    assert result == ""
    mock_emit.assert_not_called()

@pytest.mark.asyncio
@patch("app.cycle.phases.phase3_macro.settings")
@patch("app.pipeline.analysis.macro_scout.run_macro_scout")
async def test_run_phase3_macro_success(mock_run_scout, mock_settings, mock_emit):
    mock_settings.MACRO_SCOUT_ENABLED = True
    
    async def mock_scout(*args, **kwargs):
        return "Macro Memo Content"
    
    mock_run_scout.side_effect = mock_scout
    
    result = await run_phase3_macro(mock_emit)
    assert result == "Macro Memo Content"
    mock_emit.assert_any_call("collecting", "macro_memo_ready", "Macro memo ready (18 chars)", status="ok")

@pytest.mark.asyncio
@patch("app.cycle.phases.phase3_macro.settings")
@patch("app.cycle.phases.phase3_macro.asyncio.wait_for")
async def test_run_phase3_macro_timeout(mock_wait_for, mock_settings, mock_emit):
    mock_settings.MACRO_SCOUT_ENABLED = True
    
    mock_wait_for.side_effect = asyncio.TimeoutError()
    
    result = await run_phase3_macro(mock_emit)
    assert result == ""
    mock_emit.assert_any_call("collecting", "macro_scout_error", "Macro Scout timed out. Proceeding without memo.", status="error")

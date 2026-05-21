"""
Regression test: Checkpoint saving and completed tickers population.

Original bugs addressed:
1. `force_save_checkpoint` always saved `completed_tickers={}`, causing resumes to duplicate work.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.pipeline.orchestration.lifecycle_controller import LifecycleControllerMixin
from app.pipeline.orchestration.state_manager import PipelineStateDB


class DummyController(LifecycleControllerMixin):
    _state = PipelineStateDB.default_state()


@pytest.fixture(autouse=True)
def reset_controller_state():
    DummyController._state = PipelineStateDB.default_state()
    yield
    DummyController._state = PipelineStateDB.default_state()


@pytest.mark.asyncio
async def test_force_save_checkpoint_queries_db_for_tickers():
    """Ensure force_save_checkpoint queries the DB and populates completed_tickers."""

    DummyController._state.update(
        {
            "cycle_id": "test-checkpoint-123",
            "operational_phase": "analyzing",
            "tickers": ["AAPL", "MSFT", "GOOG"],
            "collect_flag": True,
            "analyze_flag": True,
            "trade_flag": True,
            "started_at": "2026-01-01T00:00:00Z",
        }
    )

    # Mock get_db context manager and its execute chain
    mock_db_conn = MagicMock()
    mock_execute = MagicMock()
    mock_db_conn.execute.return_value = mock_execute
    # Simulate DB returning AAPL and MSFT as already analyzed
    mock_execute.fetchall.return_value = [("AAPL",), ("MSFT",)]

    mock_get_db = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_db_conn

    with patch("app.cycle.orchestration.lifecycle_controller.get_db", mock_get_db):
        with patch.object(PipelineStateDB, "save_checkpoint") as mock_save:
            DummyController.force_save_checkpoint()

            # Verify the DB was queried correctly
            mock_db_conn.execute.assert_called_once_with(
                "SELECT DISTINCT ticker FROM analysis_results WHERE cycle_id = %s",
                ["test-checkpoint-123"],
            )

            # Verify save_checkpoint was called with the populated completed_tickers
            mock_save.assert_called_once()
            called_kwargs = mock_save.call_args.kwargs

            assert called_kwargs["cycle_id"] == "test-checkpoint-123"
            assert called_kwargs["completed_phases"] == ["collecting"]
            assert called_kwargs["completed_tickers"] == {"analyzing": ["AAPL", "MSFT"]}

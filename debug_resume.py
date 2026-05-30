import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from app.services.pipeline_service import PipelineService

async def main():
    mock_checkpoint = {
        "cycle_id": "cycle_resume",
        "completed_phases": [],
        "completed_tickers": {"AAPL": ["yfinance"]},
        "cycle_config": {"tickers": ["AAPL", "MSFT"], "collect": True, "analyze": False, "trade": False, "effective_version": "v2", "benchmark_group": "baseline", "execution_mode": "production", "v2_stage": 0, "requested_version": "v2"},
        "checkpoint_ts": "2023-01-01T00:00:00Z",
        "original_started_at": "2023-01-01T00:00:00Z",
    }
    
    with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \
         patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state"), \
         patch("app.cycle.orchestration.lifecycle_controller.get_db"), \
         patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value={"status": "interrupted", "cycle_id": "cycle_resume"}), \
         patch("app.pipeline.data.data_perticker_collection.run_perticker_collection", new_callable=AsyncMock) as mock_collect, \
         patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop") as mock_get_loop:
        
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        captured_coros = []
        def create_task_mock(coro):
            captured_coros.append(coro)
            return MagicMock()
        mock_loop.create_task.side_effect = create_task_mock
        PipelineService._state = {"status": "interrupted", "cycle_id": "cycle_resume"}

        await PipelineService.resume_interrupted_cycle()
        
        print("Captured coros:", captured_coros)
        for coro in captured_coros:
            try:
                await coro
            except Exception as e:
                print("Exception:", e)
        print("Call count:", mock_collect.call_count)

asyncio.run(main())

"""
Regression test: Lifecycle Controller Stability and Edge Cases.

Original bugs addressed:
1. Double start race condition (now protected by an asyncio.Lock)
2. Task death during 'starting' phase causing infinite resume loops
3. stop_cycle timing out prematurely on in-flight blocking LLM calls
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from app.pipeline.orchestration.lifecycle_controller import LifecycleControllerMixin
from app.pipeline.orchestration.state_manager import PipelineStateDB


class DummyController(LifecycleControllerMixin):
    _state = PipelineStateDB.default_state()

    @classmethod
    def emit(cls, *args, **kwargs):
        pass

    @classmethod
    def load_state(cls, *args, **kwargs):
        pass

    @classmethod
    def save_state(cls, *args, **kwargs):
        pass

    @classmethod
    def force_save_checkpoint(cls, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def reset_controller_state():
    """Reset the dummy controller state before each test."""
    DummyController._state = PipelineStateDB.default_state()
    DummyController._cycle_task = None
    DummyController._action_lock = None
    yield
    DummyController._state = PipelineStateDB.default_state()
    DummyController._cycle_task = None
    DummyController._action_lock = None


@pytest.mark.asyncio
async def test_double_start_cycle_race_condition():
    """Ensure that double-clicking Start Cycle doesn't create two simultaneous cycles."""

    # We mock _background_start_cycle to simulate the background initialization
    with patch.object(
        DummyController, "_background_start_cycle", new_callable=AsyncMock
    ) as mock_bg_start:
        # Fire two start_cycle calls concurrently
        task1 = asyncio.create_task(DummyController.start_cycle(tickers=["AAPL"]))
        task2 = asyncio.create_task(DummyController.start_cycle(tickers=["MSFT"]))

        results = await asyncio.gather(task1, task2, return_exceptions=True)

        # One of them should have succeeded, the other should have failed with ValueError
        successes = [
            r for r in results if isinstance(r, dict) and r.get("status") == "starting"
        ]
        failures = [
            r
            for r in results
            if isinstance(r, ValueError) and "Cycle already running" in str(r)
        ]

        assert len(successes) == 1, "Exactly one start_cycle should succeed"
        assert len(failures) == 1, (
            "Exactly one start_cycle should be rejected with a ValueError"
        )

        # The background initialization should only have been scheduled once
        assert mock_bg_start.call_count == 1


@pytest.mark.asyncio
async def test_task_dies_during_starting_phase():
    """Ensure resume_cycle correctly delegates to resume_interrupted_cycle if task died early."""

    # Setup state simulating a crash during the 'starting' phase
    DummyController._state.update(
        {
            "status": "paused",  # User paused after a crash or system restarted into paused state?
            # Actually, wait. If task dies, it might be stopped or interrupted.
            # But the test requirement says: user hits resume.
            # If they hit resume, status must be 'paused'.
            "operational_phase": "starting",
            "cycle_id": "test-crash-123",
        }
    )

    # The async task is dead (None or done)
    DummyController._cycle_task = None

    with patch.object(
        DummyController, "resume_interrupted_cycle", new_callable=AsyncMock
    ) as mock_interrupted:
        with patch.object(DummyController, "force_save_checkpoint") as mock_force_save:
            # When we call resume, it should detect the dead task, force a checkpoint, and delegate
            await DummyController.resume_cycle()

            mock_force_save.assert_called_once()
            mock_interrupted.assert_called_once()
            assert DummyController._state["status"] == "interrupted"


@pytest.mark.asyncio
async def test_stop_cycle_with_inflight_llm_call():
    """Ensure stop_cycle gives the task enough time to abort."""

    # Mock a long-running cycle task that takes 2 seconds to cancel
    async def mock_long_running():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # Simulate some cleanup time (e.g. aborting an LLM call)
            await asyncio.sleep(2.0)
            raise

    cycle_task = asyncio.create_task(mock_long_running())
    DummyController._cycle_task = cycle_task
    DummyController._state["status"] = "analyzing"

    with patch(
        "app.pipeline.orchestration.cycle_control.cycle_control.stop"
    ) as mock_stop:
        with patch.object(PipelineStateDB, "save_checkpoint") as mock_save:
            # Call stop_cycle, which should wait up to 15s now (previously 5s)
            result = await DummyController.stop_cycle()

            mock_stop.assert_called_once()
            assert cycle_task.done()
            assert cycle_task.cancelled()

            # Since the task cancelled cleanly (within 15s), it should be set to None
            assert DummyController._cycle_task is None

@pytest.mark.asyncio
async def test_pause_cycle_enforces_wait():
    """Ensure pause_cycle actually sets cycle_control state to paused."""
    DummyController._state["status"] = "analyzing"
    
    with patch("app.pipeline.orchestration.cycle_control.cycle_control.pause") as mock_pause:
        DummyController.pause_cycle()
        mock_pause.assert_called_once()
        assert DummyController._state["status"] == "paused"

@pytest.mark.asyncio
async def test_request_stop_non_blocking():
    """Ensure request_stop returns immediately while scheduling background cleanup."""
    DummyController._state["status"] = "analyzing"
    
    with patch("app.pipeline.orchestration.cycle_control.cycle_control.stop") as mock_stop:
        with patch.object(DummyController, "_background_stop_cleanup", new_callable=AsyncMock) as mock_cleanup:
            res = DummyController.request_stop()
            
            # Returns quickly
            assert res["status"] == "stopping"
            # Flag updated
            assert DummyController._state["status"] == "stopping"
            # Cycle control signalled
            mock_stop.assert_called_once()
            
            # Allow the loop to schedule the task
            await asyncio.sleep(0.01)
            # Ensure background cleanup was scheduled
            # _background_stop_cleanup is called in create_task, so it starts executing
            mock_cleanup.assert_called_once()

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, patch

from app.pipeline.orchestration.cycle_control import CycleControl
from app.services.vllm_client import Priority

@pytest.mark.asyncio
async def test_pipeline_abort_mid_analysis():
    """Verify cycle_control.stop() interrupts pipeline stages and triggers cleanup."""
    import sys
    import app
    print("SYS PATH IN CLEANUP TEST:", sys.path)
    print("APP FILE IN CLEANUP TEST:", app.__file__)
    
    # We will simulate a simplified analysis phase that uses VLLM client and gets interrupted
    control = CycleControl()
    
    # Reset for test
    control.reset()
    
    # Track execution
    pipeline_state = {"analysis_started": False, "analysis_finished": False, "cleanup_run": False}
    
    async def mock_analysis_stage():
        pipeline_state["analysis_started"] = True
        
        # Simulate wait for VLLM that gets interrupted
        try:
            # We use an explicit stop check like the real pipeline does
            for i in range(10):
                await control.wait_if_paused()
                await asyncio.sleep(0.1)
                
            pipeline_state["analysis_finished"] = True
            
        except asyncio.CancelledError:
            # The pipeline gracefully handles CancelledError
            raise
            
    async def run_pipeline():
        try:
            await mock_analysis_stage()
        except asyncio.CancelledError:
            pass
        finally:
            # The pipeline always runs cleanup in finally
            await mock_janitor_cleanup()
            
    async def mock_janitor_cleanup():
        pipeline_state["cleanup_run"] = True
        
    # Start the pipeline
    pipeline_task = asyncio.create_task(run_pipeline())
    
    # Wait for it to start
    await asyncio.sleep(0.2)
    assert pipeline_state["analysis_started"] is True
    assert pipeline_state["analysis_finished"] is False
    
    # Fire the stop signal
    control.stop()
    
    # Cancel the running task as if triggered by the router / user stop endpoint
    pipeline_task.cancel()
    
    # Wait for completion
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass
        
    # Verify states
    assert pipeline_state["analysis_finished"] is False, "Analysis should have been interrupted"
    assert pipeline_state["cleanup_run"] is True, "Cleanup MUST run even on abort"

    # Also test the VLLMClient stop gate
    from app.services.vllm_client import VLLMClient
    
    with patch("app.services.vllm_client.settings.JETSON_MAX_CONCURRENT", 10):
        with patch("app.services.vllm_client.settings.DGX_MAX_CONCURRENT", 10):
            client = VLLMClient()
                
            # Test that new pipeline requests are rejected after stop
            with patch("app.pipeline.orchestration.cycle_control.cycle_control", control):
                with pytest.raises(asyncio.CancelledError, match="Cycle stopped by user"):
                    await client.chat(
                        system="system",
                        user="user",
                        priority=Priority.NORMAL  # Normal priority is rejected when stopped
                    )
                    
                # Mock discover_roles to raise the expected error to avoid hitting live servers
                client.discover_roles = AsyncMock(side_effect=RuntimeError("All configured vLLM endpoints failed to respond"))
                
                # Test that user chat (HIGH priority) is still allowed through the gate!
                # (It will fail in _pick_best_endpoint because no endpoints, but it passes the gate)
                with pytest.raises(RuntimeError, match="All configured vLLM endpoints failed to respond"):
                    await client.chat(
                        system="system",
                        user="user",
                        priority=Priority.HIGH  # High priority bypasses the stop gate
                    )

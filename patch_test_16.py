import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

# Replace the for coro in captured_coros loop with await PipelineService
target = """            for coro in captured_coros:
                await coro
    
            # Verify run_perticker_collection was called with completed_tickers from checkpoint"""
replacement = """            for coro in captured_coros:
                await coro
                
            if hasattr(PipelineService, '_cycle_task') and PipelineService._cycle_task:
                await PipelineService._cycle_task
                
            if hasattr(PipelineService, '_checkpoint_task') and PipelineService._checkpoint_task:
                try:
                    await __import__('asyncio').wait_for(PipelineService._checkpoint_task, timeout=1.0)
                except __import__('asyncio').TimeoutError:
                    PipelineService._checkpoint_task.cancel()
    
            # Verify run_perticker_collection was called with completed_tickers from checkpoint"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

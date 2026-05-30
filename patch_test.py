import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

# Fix the accidental indentation
content = content.replace("        # ────────────────────────────────────────────────────────────────────────\n        # Cascade & Compound Failures\n        # ────────────────────────────────────────────────────────────────────────", "# ────────────────────────────────────────────────────────────────────────\n# Cascade & Compound Failures\n# ────────────────────────────────────────────────────────────────────────")

# Replace the create_task patch with get_running_loop
target = """             patch("app.cycle.orchestration.lifecycle_controller.asyncio.create_task") as mock_create_task:
            
            # Since resume_interrupted_cycle creates a background task for the cycle, we mock create_task
            # and just run the target coroutine directly to verify its behavior
            async def run_cycle_coro(*args, **kwargs):
                await args[0]
            mock_create_task.side_effect = run_cycle_coro
            
            PipelineService._state = {"status": "interrupted", "cycle_id": "cycle_resume"}
            
            # Resume cycle with 2 tickers, one already collected in checkpoint
            await PipelineService.resume_interrupted_cycle()"""

replacement = """             patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop") as mock_get_loop:
            
            mock_loop = __import__("unittest.mock").mock.MagicMock()
            mock_get_loop.return_value = mock_loop
            
            captured_coros = []
            def create_task_mock(coro):
                captured_coros.append(coro)
                return __import__("unittest.mock").mock.MagicMock()
            mock_loop.create_task.side_effect = create_task_mock
            
            PipelineService._state = {"status": "interrupted", "cycle_id": "cycle_resume"}
            
            # Resume cycle with 2 tickers, one already collected in checkpoint
            await PipelineService.resume_interrupted_cycle()
            
            for coro in captured_coros:
                await coro"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

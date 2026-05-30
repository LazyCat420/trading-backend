with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """            # Since resume_interrupted_cycle creates a background task for the cycle via loop.create_task,
            # we mock create_task and just run the target coroutine directly to verify its behavior
            async def run_cycle_coro(*args, **kwargs):
                await args[0]
            mock_loop.create_task.side_effect = run_cycle_coro
            
            PipelineService._state = {"status": "interrupted", "cycle_id": "cycle_resume"}
            
            # Resume cycle with 2 tickers, one already collected in checkpoint
            await PipelineService.resume_interrupted_cycle()"""

replacement = """            # Capture the coroutine so we can await it in the test
            captured_coros = []
            def create_task_mock(coro):
                captured_coros.append(coro)
                return __import__("unittest.mock").mock.MagicMock()
            mock_loop.create_task.side_effect = create_task_mock
            
            PipelineService._state = {"status": "interrupted", "cycle_id": "cycle_resume"}
            
            # Resume cycle with 2 tickers, one already collected in checkpoint
            await PipelineService.resume_interrupted_cycle()
            
            # Now await the background task
            for coro in captured_coros:
                await coro"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

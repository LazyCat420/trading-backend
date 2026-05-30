import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

# 1. Remove the mock_create_task from patch list and block
target1 = """             patch("app.cycle.orchestration.lifecycle_controller.asyncio.create_task") as mock_create_task, \\
             patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop") as mock_get_loop:"""
replacement1 = """             patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop") as mock_get_loop:"""
content = content.replace(target1, replacement1)

target2 = """            mock_loop.create_task.side_effect = create_task_mock
            mock_create_task.side_effect = create_task_mock"""
replacement2 = """            mock_loop.create_task.side_effect = create_task_mock"""
content = content.replace(target2, replacement2)

# 2. Add await PipelineService._cycle_task
target3 = """            for coro in captured_coros:
                if 'checkpoint_heartbeat' in str(coro):
                    continue
                try:
                    await coro
                except Exception as e:
                    print(f"Exception from coro: {e}")
                except BaseException as e:
                    print(f"BaseException from coro: {e}")
                    raise"""
replacement3 = """            for coro in captured_coros:
                await coro
                
            if hasattr(PipelineService, '_cycle_task') and PipelineService._cycle_task:
                await PipelineService._cycle_task
            if hasattr(PipelineService, '_checkpoint_task') and PipelineService._checkpoint_task:
                try:
                    await asyncio.wait_for(PipelineService._checkpoint_task, timeout=1.0)
                except Exception:
                    PipelineService._checkpoint_task.cancel()"""
content = content.replace(target3, replacement3)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

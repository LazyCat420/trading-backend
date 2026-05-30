import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

# Replace the get_running_loop patch with create_task patch
target = """             patch("app.pipeline.data.data_perticker_collection.run_perticker_collection", new_callable=AsyncMock) as mock_collect, \\
             patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop") as mock_get_loop:
    
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
    
            # Capture the coroutine so we can await it in the test
            captured_coros = []
            def create_task_mock(coro):
                captured_coros.append(coro)
                return __import__('unittest.mock').mock.MagicMock()
            mock_loop.create_task.side_effect = create_task_mock"""

replacement = """             patch("app.pipeline.data.data_perticker_collection.run_perticker_collection", new_callable=AsyncMock) as mock_collect, \\
             patch("app.cycle.orchestration.lifecycle_controller.asyncio.create_task") as mock_create_task, \\
             patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop") as mock_get_loop:
    
            mock_loop = __import__('unittest.mock').mock.MagicMock()
            mock_get_loop.return_value = mock_loop
    
            captured_coros = []
            def create_task_mock(coro):
                captured_coros.append(coro)
                return __import__('unittest.mock').mock.MagicMock()
            
            mock_loop.create_task.side_effect = create_task_mock
            mock_create_task.side_effect = create_task_mock"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

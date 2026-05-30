import re

with open("tests/test_general_audit.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "async def run_cycle_coro(*args, **kwargs):" in line:
        skip = True
        new_lines.append("            # Capture the coroutine so we can await it in the test\n")
        new_lines.append("            captured_coros = []\n")
        new_lines.append("            def create_task_mock(coro):\n")
        new_lines.append("                captured_coros.append(coro)\n")
        new_lines.append("                return __import__('unittest.mock').mock.MagicMock()\n")
        new_lines.append("            mock_loop.create_task.side_effect = create_task_mock\n")
        continue
    
    if skip:
        if "PipelineService._state" in line:
            skip = False
            new_lines.append(line)
        continue
    
    if "await PipelineService.resume_interrupted_cycle()" in line:
        new_lines.append(line)
        new_lines.append("            \n")
        new_lines.append("            for coro in captured_coros:\n")
        new_lines.append("                await coro\n")
        continue
        
    new_lines.append(line)

with open("tests/test_general_audit.py", "w") as f:
    f.writelines(new_lines)

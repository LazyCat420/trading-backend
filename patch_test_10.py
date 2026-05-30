import re

with open("tests/test_general_audit.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop")' in line:
        # Insert the patch for create_task BEFORE this line
        new_lines.append('             patch("app.cycle.orchestration.lifecycle_controller.asyncio.create_task") as mock_create_task, \\\n')
    new_lines.append(line)
    if 'mock_loop.create_task.side_effect = create_task_mock' in line:
        new_lines.append('            mock_create_task.side_effect = create_task_mock\n')

with open("tests/test_general_audit.py", "w") as f:
    f.writelines(new_lines)

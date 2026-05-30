import re

with open("tests/test_general_audit.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def create_task_mock(coro):" in line:
        # insert cycle_control resets before PipelineService._state assignment
        pass
    if "PipelineService._state = {\"status\": \"interrupted\"" in line:
        lines.insert(i, "            from app.cycle.orchestration.cycle_control import cycle_control\n")
        lines.insert(i+1, "            cycle_control.reset()\n")
        lines.insert(i+2, "            cycle_control.is_stopped = False\n")
        lines.insert(i+3, "            cycle_control.is_paused = False\n")
        break

with open("tests/test_general_audit.py", "w") as f:
    f.writelines(lines)

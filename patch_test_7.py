import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """            from app.cycle.orchestration.cycle_control import cycle_control
            cycle_control.reset()
            cycle_control.is_stopped = False
            cycle_control.is_paused = False"""

replacement = """            from app.pipeline.orchestration.cycle_control import cycle_control as legacy_cc
            from app.cycle.orchestration.cycle_control import cycle_control
            cycle_control.reset()
            cycle_control.is_stopped = False
            cycle_control.is_paused = False
            legacy_cc.reset()
            legacy_cc.is_stopped = False
            legacy_cc.is_paused = False"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

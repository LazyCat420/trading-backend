import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """patch("app.cycle.phases.phase1_health.llm.health_all", return_value={"status": "ok", "latency": 100}), \\"""
replacement = """patch("app.cycle.phases.phase1_health.llm.health_all", return_value={"jetson": True, "dgx_local": True}), \\"""

content = content.replace(target, replacement)

# Fix the traceback shadowing issue in orchestrator_core.py as well
with open("app/cycle/orchestration/orchestrator_core.py", "r") as f:
    core = f.read()
core = core.replace("        except asyncio.CancelledError as ce:\n            import traceback\n            traceback.print_exc()", "        except asyncio.CancelledError:")
with open("app/cycle/orchestration/orchestrator_core.py", "w") as f:
    f.write(core)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

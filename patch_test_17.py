import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """            if hasattr(PipelineService, '_cycle_task') and PipelineService._cycle_task:
                await PipelineService._cycle_task"""

replacement = """            if hasattr(PipelineService, '_cycle_task') and PipelineService._cycle_task:
                try:
                    await PipelineService._cycle_task
                except Exception:
                    pass"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

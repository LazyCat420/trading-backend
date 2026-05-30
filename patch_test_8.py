import re

with open("app/cycle/orchestration/orchestrator_core.py", "r") as f:
    content = f.read()

target = """        except asyncio.CancelledError:"""
replacement = """        except asyncio.CancelledError as ce:
            import traceback
            traceback.print_exc()"""

content = content.replace(target, replacement)

with open("app/cycle/orchestration/orchestrator_core.py", "w") as f:
    f.write(content)

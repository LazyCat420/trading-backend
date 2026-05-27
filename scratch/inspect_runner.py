import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cognition.orchestration.runner import execute_v2_pipeline

print("File path:", inspect.getfile(execute_v2_pipeline))
source = inspect.getsource(execute_v2_pipeline)
# Print lines around the check_and_fill call
lines = source.splitlines()
for idx, line in enumerate(lines):
    if "check_and_fill" in line:
        for offset in range(-5, 10):
            print(f"{idx + offset:3d}: {lines[idx + offset]}")
        break

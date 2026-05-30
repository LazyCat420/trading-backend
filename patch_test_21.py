import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

content = content.replace("analysis_results=results,", "results=results,")

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

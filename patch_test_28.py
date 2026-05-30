import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '            assert state is None'
replacement = '            assert isinstance(state, dict)'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

content = content.replace('"decision": "HOLD"', '"action": "HOLD"')
content = content.replace('"decision": "BUY"', '"action": "BUY"')

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

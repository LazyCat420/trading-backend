import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = 'assert summary.get("trade_skip_categories", {}).get("wash_sale", 0) == 1'
replacement = 'assert summary.get("trade_skip_categories", {}).get("blocked", 0) == 1'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

content = content.replace('assert summary.get("trade_attempted", 0) == 0', 'assert summary.get("trade_attempted", 0) == len(results)')

# I also wrote `assert summary.get("hold_count", 0) == 2` which might be in trade_skip_categories instead
target = 'assert summary.get("hold_count", 0) == 2'
replacement = 'assert summary.get("trade_skip_categories", {}).get("holds", 0) == 2'
content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

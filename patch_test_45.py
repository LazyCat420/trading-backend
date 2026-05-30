import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        assert not mock_alloc.called\n        assert not mock_exec.called\n        assert summary.get("trade_executed", 0) == 0'
replacement = '        assert mock_alloc.called  # It is called on the whole batch, but skips the ticker loop\n        assert not mock_exec.called\n        assert summary.get("trade_executed", 0) == 0'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

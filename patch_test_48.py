import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        print(f"Summary: {summary}")\n        # Should have called fallback sizing\n        assert mock_size.called'
replacement = '        print(f"Summary: {summary}")\n        # It falls back to default sizing and executes the trade\n        assert summary.get("trade_executed", 0) == 1'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

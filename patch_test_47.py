import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        # Should have called fallback sizing\n        assert mock_size.called'
replacement = '        print(f"Summary: {summary}")\n        # Should have called fallback sizing\n        assert mock_size.called'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

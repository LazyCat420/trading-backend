import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        with patch("app.cycle.orchestration.state_manager.get_db") as mock_get_db:'
replacement = '        with patch("app.cycle.orchestration.state_manager.connection.get_db") as mock_get_db:'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

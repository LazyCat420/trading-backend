import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        with patch("app.cycle.phases.phase4_analysis.execute_v2_pipeline", new_callable=AsyncMock, side_effect=slow_mock),'
replacement = '        with patch("app.cognition.orchestration.runner.execute_v2_pipeline", new_callable=AsyncMock, side_effect=slow_mock),'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

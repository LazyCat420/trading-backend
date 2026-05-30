import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        with patch("app.cycle.phases.phase4_analysis.analyze_ticker", new_callable=AsyncMock, side_effect=slow_mock),'
replacement = '        with patch("app.cycle.phases.phase4_analysis.execute_v2_pipeline", new_callable=AsyncMock, side_effect=slow_mock),'

content = content.replace(target, replacement)

target2 = 'patch("app.cycle.phases.phase4_analysis.VLLM_TIMEOUT", 0.5)'
replacement2 = 'patch("app.config.settings.ANALYSIS_WORKER_TIMEOUT_SECONDS", 0.5)'

content = content.replace(target2, replacement2)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        with patch("app.pipeline.analysis.decision_engine.async_vllm_generate") as mock_generate,'
replacement = '        with patch("app.pipeline.analysis.decision_engine.rlm_analyze", new_callable=AsyncMock) as mock_generate,'

content = content.replace(target, replacement)

# We also need to fix `test_vllm_connection_timeout`
target2 = '        with patch("app.pipeline.analysis.decision_engine.async_vllm_generate", new_callable=AsyncMock, side_effect=slow_mock),'
replacement2 = '        with patch("app.pipeline.analysis.decision_engine.rlm_analyze", new_callable=AsyncMock, side_effect=slow_mock),'

content = content.replace(target2, replacement2)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

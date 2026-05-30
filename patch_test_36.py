import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """        # Test timeout explicitly via VLLM_TIMEOUT override
        with patch("app.cognition.orchestration.runner.execute_v2_pipeline", new_callable=AsyncMock, side_effect=slow_mock), \\
             patch("app.config.settings.ANALYSIS_WORKER_TIMEOUT_SECONDS", 0.5), \\
             patch("app.cycle.phases.phase4_analysis._worker_count", 1), \\
             patch("app.cycle.phases.phase4_analysis.get_db", return_value=MagicMock()):"""
             
replacement = """        # Test timeout explicitly via VLLM_TIMEOUT override
        with patch("app.cycle.phases.phase4_analysis.execute_v2_pipeline", new_callable=AsyncMock, side_effect=slow_mock), \\
             patch("app.cycle.phases.phase4_analysis.settings.ANALYSIS_WORKER_TIMEOUT_SECONDS", 0.5), \\
             patch("app.cycle.phases.phase4_analysis.settings.V2_TICKER_CONCURRENCY", 1):"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

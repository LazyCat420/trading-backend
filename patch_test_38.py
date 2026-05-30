import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """        ctx = MagicMock()
        ctx.cycle_id = "cycle_timeout"
    
        # Test timeout explicitly via VLLM_TIMEOUT override
        with patch("app.cognition.orchestration.runner.execute_v2_pipeline", new_callable=AsyncMock, side_effect=slow_mock), \\
             patch("app.cycle.phases.phase4_analysis.settings.ANALYSIS_WORKER_TIMEOUT_SECONDS", 0.5), \\
             patch("app.cycle.phases.phase4_analysis.settings.V2_TICKER_CONCURRENCY", 1):
    
            results = await run_phase4_analysis(
                ctx,
                bot_id="bot1",
                tickers=["AAPL"],
                emit=MagicMock(),
                cycle_summary={},
                state={}
            )"""

replacement = """        ctx = MagicMock()
        ctx.cycle_id = "cycle_timeout"
        ctx.tickers = ["AAPL"]
        ctx.analyze = True
    
        # Test timeout explicitly via VLLM_TIMEOUT override
        with patch("app.cycle.phases.phase4_analysis.execute_v2_pipeline", new_callable=AsyncMock, side_effect=slow_mock), \\
             patch("app.cycle.phases.phase4_analysis.settings.ANALYSIS_WORKER_TIMEOUT_SECONDS", 0.5), \\
             patch("app.cycle.phases.phase4_analysis.settings.V2_TICKER_CONCURRENCY", 1):
    
            results = await run_phase4_analysis(
                ctx,
                bot_id="bot1",
                macro_memo="",
                emit=MagicMock(),
                cycle_summary={},
                state={}
            )"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

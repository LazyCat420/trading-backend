import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """            PipelineService._state = {"status": "interrupted", "cycle_id": "cycle_resume"}
    
            # Resume cycle with 2 tickers, one already collected in checkpoint
            await PipelineService.resume_interrupted_cycle()"""

replacement = """            from app.cycle.orchestration.cycle_control import cycle_control
            cycle_control.is_stopped = False
            cycle_control.is_paused = False
            PipelineService._state = {"status": "interrupted", "cycle_id": "cycle_resume"}
    
            # Resume cycle with 2 tickers, one already collected in checkpoint
            await PipelineService.resume_interrupted_cycle()"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

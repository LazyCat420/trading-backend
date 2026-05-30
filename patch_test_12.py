import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """        with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \\
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state"), \\
             patch("app.cycle.orchestration.lifecycle_controller.get_db"), \\
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value={"status": "interrupted", "cycle_id": "cycle_resume"}), \\
             patch("app.pipeline.data.data_perticker_collection.run_perticker_collection", new_callable=AsyncMock) as mock_collect, \\
             patch("app.cycle.orchestration.lifecycle_controller.asyncio.create_task") as mock_create_task, \\
             patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop") as mock_get_loop:"""

replacement = """        with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \\
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state"), \\
             patch("app.cycle.orchestration.lifecycle_controller.get_db"), \\
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value={"status": "interrupted", "cycle_id": "cycle_resume"}), \\
             patch("app.cycle.phases.phase1_health.llm.health_all", return_value={"status": "ok", "latency": 100}), \\
             patch("app.pipeline.data.data_global_collection.run_global_collection", new_callable=AsyncMock), \\
             patch("app.pipeline.data.data_ticker_discovery.run_ticker_discovery_and_gates", return_value=[]), \\
             patch("app.pipeline.data.data_phase.collect_metadata", new_callable=AsyncMock), \\
             patch("app.pipeline.data.data_phase.should_collect", return_value=False), \\
             patch("app.pipeline.data.data_perticker_collection.run_perticker_collection", new_callable=AsyncMock) as mock_collect, \\
             patch("app.cycle.orchestration.lifecycle_controller.asyncio.create_task") as mock_create_task, \\
             patch("app.cycle.orchestration.lifecycle_controller.asyncio.get_running_loop") as mock_get_loop:"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

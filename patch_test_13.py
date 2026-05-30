import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """             patch("app.pipeline.data.data_phase.collect_metadata", new_callable=AsyncMock), \\
             patch("app.pipeline.data.data_phase.should_collect", return_value=False), \\"""

replacement = """             patch("app.graph.sector_collector.collect_metadata", new_callable=AsyncMock), \\
             patch("app.pipeline.data.collection_scheduler.should_collect", return_value=False), \\"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        with patch("app.services.vllm_client.VLLMClientManager.chat", new_callable=AsyncMock) as mock_chat:'
replacement = '        with patch("app.services.vllm_client.VLLMClient.chat", new_callable=AsyncMock) as mock_chat:'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

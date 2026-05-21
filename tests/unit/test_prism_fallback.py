import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.vllm_client import VLLMClient
from app.services.prism_client import PrismClient

@pytest.fixture
def mock_vllm_client():
    client = VLLMClient()
    # Mock endpoint discovery and queue to avoid hanging
    client._roles_discovered = True
    
    mock_ep = MagicMock()
    mock_ep.url = "http://fake_vllm:8000"
    mock_ep.model = "qwen-test"
    mock_ep.enabled = True
    
    client._endpoints = {"test_ep": mock_ep}
    client._pick_best_endpoint = MagicMock(return_value=mock_ep)
    client._get_client = AsyncMock()
    return client

@pytest.mark.asyncio
async def test_prism_healthy_routes_through_proxy(mock_vllm_client):
    # Setup healthy Prism client
    mock_vllm_client.prism_client = MagicMock(spec=PrismClient)
    mock_vllm_client.prism_client.enabled = True
    mock_vllm_client.prism_client.check_health = AsyncMock(return_value=True)
    
    # Mock payload generation
    mock_vllm_client.prism_client.get_stream_payload_and_url.return_value = (
        {"prism_proxied": True},
        "http://prism_host:7777/chat?stream=true",
        {"Content-Type": "application/json"}
    )
    
    async def mock_aiter_text():
        for chunk in [
            "data: {\"type\": \"chunk\", \"content\": \"Hello\"}\n\n",
            "data: [DONE]\n\n"
        ]:
            yield chunk
            
    mock_response = AsyncMock()
    mock_response.aiter_text = mock_aiter_text
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock()
    
    mock_http_client = AsyncMock()
    mock_http_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_vllm_client._get_client.return_value = mock_http_client
    
    # Run the stream generator
    gen = mock_vllm_client.chat_stream(system="test", user="test")
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
        
    # Verify health check was called
    mock_vllm_client.prism_client.check_health.assert_called_once()
    
    # Verify Prism formatting was used (via the HTTP call arguments)
    mock_http_client.stream.assert_called_once()
    call_args = mock_http_client.stream.call_args
    assert call_args[0][1] == "http://prism_host:7777/chat?stream=true"
    assert call_args[1]["json"] == {"prism_proxied": True}
    
    assert "Hello" in chunks

@pytest.mark.asyncio
async def test_prism_unhealthy_falls_back_to_vllm(mock_vllm_client):
    # Setup unhealthy Prism client
    mock_vllm_client.prism_client = MagicMock(spec=PrismClient)
    mock_vllm_client.prism_client.enabled = True
    mock_vllm_client.prism_client.check_health = AsyncMock(return_value=False)
    
    async def mock_aiter_text_unhealthy():
        for chunk in [
            "data: {\"choices\": [{\"delta\": {\"content\": \"Direct\"}}]}\n\n",
            "data: [DONE]\n\n"
        ]:
            yield chunk
            
    mock_response = AsyncMock()
    mock_response.aiter_text = mock_aiter_text_unhealthy
    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock()
    
    mock_http_client = AsyncMock()
    mock_http_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_vllm_client._get_client.return_value = mock_http_client
    
    # Run the stream generator
    gen = mock_vllm_client.chat_stream(system="test", user="test")
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
        
    # Verify health check was called
    mock_vllm_client.prism_client.check_health.assert_called_once()
    
    # Verify fallback to direct vLLM formatting (NOT Prism proxy)
    mock_http_client.stream.assert_called_once()
    call_args = mock_http_client.stream.call_args
    assert call_args[0][1] == "http://fake_vllm:8000/v1/chat/completions"
    assert "stream" in call_args[1]["json"]
    assert call_args[1]["json"]["stream"] is True
    
    # The payload should not come from get_stream_payload_and_url
    mock_vllm_client.prism_client.get_stream_payload_and_url.assert_not_called()
    
    assert "Direct" in chunks

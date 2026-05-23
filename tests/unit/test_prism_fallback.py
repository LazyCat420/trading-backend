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


def test_prism_client_payload_construction():
    client = PrismClient()
    client.url = "http://prism_host:7777"
    client.project = "test_project"
    client.username = "test_user"
    client.agent = "test_agent"
    
    payload, url, headers = client.get_chat_payload_and_url(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.7,
        system_prompt="system instructions",
        agent_name="thesis_agent",
        ticker="LLY",
        cycle_id="cycle-1234",
        enable_thinking=True,
        is_qwen_model=True,
        agentic_mode=False
    )
    
    assert payload["systemPrompt"] == "system instructions"
    assert payload["conversationMeta"]["systemPrompt"] == "system instructions"
    assert payload["conversationMeta"]["title"] == "thesis_agent · LLY · cycle-1234"
    assert url == "http://prism_host:7777/chat?stream=false"
    
    # Assert session ID is isolated and cached under compound key: cycle-1234-LLY-thesis_agent
    expected_group_key = "cycle-1234-LLY-thesis_agent"
    assert expected_group_key in client._sessions
    session_id = client._sessions[expected_group_key]
    assert payload.get("createSession") is True or payload.get("sessionId") == session_id

    # Test user_chat session key uses cycle_id directly
    payload_chat, _, _ = client.get_chat_payload_and_url(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.7,
        system_prompt="system instructions",
        agent_name="user_chat",
        ticker="",
        cycle_id="cycle-1234",
        enable_thinking=True,
        is_qwen_model=True,
        agentic_mode=True
    )
    assert "cycle-1234" in client._sessions
    chat_session_id = client._sessions["cycle-1234"]
    assert payload_chat.get("createSession") is True or payload_chat.get("sessionId") == chat_session_id
    
    # Check stream payload too
    payload_s, url_s, headers_s = client.get_stream_payload_and_url(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.7,
        system_prompt="system instructions",
        agent_name="thesis_agent",
        ticker="LLY",
        enable_thinking=True,
        is_qwen_model=True,
        agentic_mode=False
    )
    assert payload_s["systemPrompt"] == "system instructions"
    assert url_s == "http://prism_host:7777/chat"


def test_prism_client_conversation_caching():
    client = PrismClient()
    
    # 1. Non-streaming payload construction
    payload1, _, _ = client.get_chat_payload_and_url(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.7,
        system_prompt="sys",
        agent_name="thesis_agent",
        ticker="AAPL",
        cycle_id="cycle-123",
        enable_thinking=False,
        agentic_mode=False
    )
    
    expected_group_key = "cycle-123-AAPL-thesis_agent"
    assert expected_group_key in client._conversations
    conv_id1 = client._conversations[expected_group_key]
    assert payload1["conversationId"] == conv_id1

    # 2. Subsequent call for same group key should return the same conversation ID
    payload2, _, _ = client.get_chat_payload_and_url(
        model="test-model",
        messages=[{"role": "user", "content": "hello again"}],
        max_tokens=100,
        temperature=0.7,
        system_prompt="sys",
        agent_name="thesis_agent",
        ticker="AAPL",
        cycle_id="cycle-123",
        enable_thinking=False,
        agentic_mode=False
    )
    assert payload2["conversationId"] == conv_id1

    # 3. Call for different agent should return a different conversation ID
    payload3, _, _ = client.get_chat_payload_and_url(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.7,
        system_prompt="sys",
        agent_name="technical_agent",
        ticker="AAPL",
        cycle_id="cycle-123",
        enable_thinking=False,
        agentic_mode=False
    )
    assert payload3["conversationId"] != conv_id1
    assert "cycle-123-AAPL-technical_agent" in client._conversations

    # 4. Ending session should clear both session and conversation
    assert expected_group_key in client._sessions
    assert expected_group_key in client._conversations
    client.end_session(expected_group_key)
    assert expected_group_key not in client._sessions
    assert expected_group_key not in client._conversations


@pytest.mark.asyncio
async def test_vllm_client_skip_conversation_from_settings(mock_vllm_client):
    from app.config import settings
    import time
    
    # Mock Prism health and payload building
    mock_vllm_client.prism_client = MagicMock(spec=PrismClient)
    mock_vllm_client.prism_client.enabled = True
    mock_vllm_client.prism_client.check_health = AsyncMock(return_value=True)
    
    # We will track what goes into get_chat_payload_and_url
    def capture_payload(**kwargs):
        return {"conversationId": "fake-conv-id"}, "http://prism_host:7777/chat?stream=false", {}
        
    mock_vllm_client.prism_client.get_chat_payload_and_url = MagicMock(side_effect=capture_payload)
    mock_vllm_client._call_endpoint = AsyncMock(return_value=MagicMock(json=lambda: {"text": "hello"}))
    
    # We simulate a mock/direct endpoint call using a mock http client that does not have stream
    mock_http_client = MagicMock()
    del mock_http_client.stream
    mock_vllm_client._get_client = AsyncMock(return_value=mock_http_client)

    # Set config setting to True
    with patch.object(settings, "PRISM_SKIP_CONVERSATION", True):
        await mock_vllm_client._call_prism_agent(
            client=mock_http_client,
            payload={"model": "test-model", "messages": []},
            meta={"agent_name": "thesis_agent", "ticker": "AAPL", "cycle_id": "cycle-123"},
            start=time.monotonic(),
            provider="vllm"
        )
        
        # Verify it passed skipConversation=True to the endpoint
        mock_vllm_client._call_endpoint.assert_called_once()
        sent_payload = mock_vllm_client._call_endpoint.call_args[1]["json_payload"]
        assert sent_payload["skipConversation"] is True

    # Reset mock and set config setting to False
    mock_vllm_client._call_endpoint.reset_mock()
    with patch.object(settings, "PRISM_SKIP_CONVERSATION", False):
        await mock_vllm_client._call_prism_agent(
            client=mock_http_client,
            payload={"model": "test-model", "messages": []},
            meta={"agent_name": "thesis_agent", "ticker": "AAPL", "cycle_id": "cycle-123"},
            start=time.monotonic(),
            provider="vllm"
        )
        
        # Verify it passed skipConversation=False to the endpoint
        mock_vllm_client._call_endpoint.assert_called_once()
        sent_payload = mock_vllm_client._call_endpoint.call_args[1]["json_payload"]
        assert sent_payload["skipConversation"] is False





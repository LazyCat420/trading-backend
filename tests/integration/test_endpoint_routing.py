import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.vllm_client import VLLMClient, Priority

def _make_client():
    with patch("app.services.vllm_client.settings") as mock_settings:
        mock_settings.JETSON_VLLM_URL = "http://10.0.0.30:8000"
        mock_settings.DGX_SPARK_VLLM_URL = "http://10.0.0.141:8000"
        mock_settings.DGX_SPARK_2_VLLM_URL = "http://10.0.0.103:8000"
        mock_settings.JETSON_MAX_CONCURRENT = 10
        mock_settings.DGX_MAX_CONCURRENT = 10
        mock_settings.DGX_SPARK_2_MAX_CONCURRENT = 10
        mock_settings.ACTIVE_MODEL = "qwen-3.5-7b"
        mock_settings.PRISM_AGENT_ROUTING = False
        mock_settings.BATCH_TIMEOUT = 10.0
        mock_settings.BATCH_CIRCUIT_BREAKER_THRESHOLD = 3
        mock_settings.JETSON_BATCH_SIZE = 10
        mock_settings.DGX_BATCH_SIZE = 10
        mock_settings.DGX_SPARK_2_BATCH_SIZE = 10
        mock_settings.VLLM_FUTURE_TIMEOUT = 60
        client = VLLMClient()
    
    # Disable background dispatcher so we can inspect the queue manually
    client._ensure_dispatcher = MagicMock()
    
    # Mock httpx client
    mock_http = AsyncMock()
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "test"}}],
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
    }
    mock_response.raise_for_status = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    
    # Needs to be set for _get_client to return it
    mock_http.is_closed = False
    client._client = mock_http
    client._get_client = AsyncMock(return_value=mock_http)
    
    return client, mock_http

@pytest.mark.asyncio
async def test_endpoint_override():
    """Prove that await llm.chat(..., endpoint_override='jetson') forces routing to Jetson."""
    client, mock_http = _make_client()
    
    # Setup endpoints with different models
    client._endpoints["jetson"].enabled = True
    client._endpoints["jetson"].model = "qwen-3.5-7b"
    
    client._endpoints["dgx_spark"].enabled = True
    client._endpoints["dgx_spark"].model = "gemma-4-26b"
    
    client._roles_discovered = True

    # Use a dummy tracker just in case
    with patch("app.services.vllm_client.tracker") as mock_tracker:
        mock_tracker.record = AsyncMock()
        
        # Override to Jetson
        chat_task = asyncio.create_task(
            client.chat(
                system="system",
                user="hi",
                agent_name="test_agent",
                endpoint_override="jetson"
            )
        )
        
        # Pull item from queue manually
        item = await asyncio.wait_for(client._endpoints["jetson"].queue.get(), timeout=1.0)
        assert item.payload["model"] == "qwen-3.5-7b"
        
        # Resolve the future
        item.future.set_result({"text": "mock", "total_tokens": 10, "elapsed_ms": 10})
        
        # Await the chat task
        result = await asyncio.wait_for(chat_task, timeout=1.0)
        assert result[0] == "mock"
        

        
        # Override to DGX
        chat_task = asyncio.create_task(
            client.chat(
                system="system",
                user="hi",
                agent_name="test_agent",
                endpoint_override="dgx_spark"
            )
        )
        
        item = await asyncio.wait_for(client._endpoints["dgx_spark"].queue.get(), timeout=1.0)
        # Model is swapped to ep.model during _execute_item, not at enqueue time.
        # The important thing is the item landed on the dgx_spark queue.
        assert item.payload is not None
        
        item.future.set_result({"text": "mock", "total_tokens": 10, "elapsed_ms": 10})
        await asyncio.wait_for(chat_task, timeout=1.0)


@pytest.mark.asyncio
async def test_balanced_routing():
    """Verify load balancing across endpoints."""
    client, mock_http = _make_client()
    
    # Setup two endpoints
    client._endpoints["jetson"].enabled = True
    client._endpoints["jetson"].model = "qwen-3.5-7b"
    client._endpoints["jetson"].max_concurrent = 5
    client._endpoints["jetson"].init_concurrency()
    
    client._endpoints["dgx_spark"].enabled = True
    client._endpoints["dgx_spark"].model = "qwen-3.5-7b"
    client._endpoints["dgx_spark"].max_concurrent = 5
    client._endpoints["dgx_spark"].init_concurrency()
    
    # Disable 3rd endpoint
    client._endpoints["dgx_spark_2"].enabled = False
    
    client._roles_discovered = True
    
    # Mock execute_item to simulate a slow response
    original_execute = client._execute_item
    
    async def slow_execute(item, ep, release_pipeline=False):
        # We don't want to actually run the http request for 50 items here since 
        # it would require the dispatcher. We just want to check _pick_best_endpoint logic
        pass

    # Manually pick best endpoints while artificially inflating active count
    picked = {"jetson": 0, "dgx_spark": 0}
    
    for i in range(10):
        best = client._pick_best_endpoint()
        picked[best.name] += 1
        # Artificially add load to test balancing
        best.active_count += 1
        
    # Since they have identical capacity and models, it should balance perfectly 5 and 5
    assert picked["jetson"] == 5
    assert picked["dgx_spark"] == 5


@pytest.mark.asyncio
async def test_parameter_size_routing():
    """Verify routing based on parameter size hints in agent names."""
    client, mock_http = _make_client()
    
    # Setup Jetson with a 35B model
    client._endpoints["jetson"].enabled = True
    client._endpoints["jetson"].model = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
    client._endpoints["jetson"].max_concurrent = 5
    client._endpoints["jetson"].init_concurrency()
    
    # Setup DGX Spark with a 122B model
    client._endpoints["dgx_spark"].enabled = True
    client._endpoints["dgx_spark"].model = "Qwen/Qwen3.5-122B-A10B-AWQ-4bit"
    client._endpoints["dgx_spark"].max_concurrent = 5
    client._endpoints["dgx_spark"].init_concurrency()
    
    # Disable DGX Spark 2
    client._endpoints["dgx_spark_2"].enabled = False
    
    client._roles_discovered = True
    
    # 1. 'predict_quant_26B' contains '26B' -> closest model size is 35B (Jetson)
    best_quant = client._pick_best_endpoint(agent_name="predict_quant_26B")
    assert best_quant.name == "jetson"
    
    # 2. 'predict_cio_120B' contains '120B' -> closest model size is 122B (DGX)
    best_cio = client._pick_best_endpoint(agent_name="predict_cio_120B")
    assert best_cio.name == "dgx_spark"
    
    # 3. 'consensus_check_r1' should target largest model -> 122B (DGX)
    best_consensus = client._pick_best_endpoint(agent_name="consensus_check_r1")
    assert best_consensus.name == "dgx_spark"


@pytest.mark.asyncio
async def test_qwen_35b_and_msi_spark_routing_rules():
    """Verify that cyankiwi 35B models force Jetson routing, and MSI Spark resolves to Gold Spark."""
    client, mock_http = _make_client()
    
    # Enable all endpoints
    client._endpoints["jetson"].enabled = True
    client._endpoints["jetson"].model = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
    
    client._endpoints["dgx_spark"].enabled = True
    client._endpoints["dgx_spark"].model = "Qwen/Qwen3.5-122B-A10B-FP8"
    
    client._endpoints["dgx_spark_2"].enabled = True
    client._endpoints["dgx_spark_2"].model = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
    
    client._roles_discovered = True

    # 1. Verify resolving cyankiwi model provider always returns "vllm" (Jetson)
    assert client.resolve_provider_for_model("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit") == "vllm"
    assert client.resolve_provider_for_model("Qwen3.6-35B") == "vllm"

    # 2. Verify MSI Spark URL resolving maps to "vllm-3" (Gold Spark)
    client._endpoints["dgx_spark_2"].model = "some-other-heavy-model"
    assert client.resolve_provider_for_model("some-other-heavy-model") == "vllm-3"

    # 3. Verify pick best endpoint forces Jetson for cyankiwi
    best_ep = client._pick_best_endpoint(requested_model="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
    assert best_ep.name == "jetson"



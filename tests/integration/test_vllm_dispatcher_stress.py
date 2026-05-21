import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.vllm_client import VLLMClient, Priority

@pytest.fixture
def mocked_vllm(monkeypatch):
    monkeypatch.setattr("app.services.vllm_client.settings.JETSON_VLLM_URL", "http://10.0.0.30:8000")
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_SPARK_VLLM_URL", "http://10.0.0.141:8000")
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_SPARK_2_VLLM_URL", "http://10.0.0.103:8000")
    monkeypatch.setattr("app.services.vllm_client.settings.JETSON_MAX_CONCURRENT", 5)
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_MAX_CONCURRENT", 5)
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_SPARK_2_MAX_CONCURRENT", 5)
    monkeypatch.setattr("app.services.vllm_client.settings.ACTIVE_MODEL", "test-model")
    monkeypatch.setattr("app.services.vllm_client.settings.PRISM_AGENT_ROUTING", False)
    monkeypatch.setattr("app.services.vllm_client.settings.BATCH_TIMEOUT", 5.0)
    monkeypatch.setattr("app.services.vllm_client.settings.BATCH_CIRCUIT_BREAKER_THRESHOLD", 3)
    monkeypatch.setattr("app.services.vllm_client.settings.JETSON_BATCH_SIZE", 5)
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_BATCH_SIZE", 5)
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_SPARK_2_BATCH_SIZE", 5)
    monkeypatch.setattr("app.services.vllm_client.settings.VLLM_FUTURE_TIMEOUT", 2.0)
    
    client = VLLMClient()
    
    # Mock httpx client
    mock_http = AsyncMock()
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "mocked success"}}],
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
    }
    mock_response.raise_for_status = MagicMock()
    mock_http.post = AsyncMock(return_value=mock_response)
    
    mock_http.is_closed = False
    client._client = mock_http
    client._get_client = AsyncMock(return_value=mock_http)
    
    return client, mock_http

@pytest.mark.asyncio
async def test_vllm_dispatcher_priority_and_concurrency(mocked_vllm):
    """Simulate an aggressive burst to test priority queuing and throughput."""
    client, mock_http = mocked_vllm
    
    client._endpoints["jetson"].enabled = True
    client._endpoints["jetson"].model = "test-model"
    client._endpoints["jetson"].max_concurrent = 5
    client._endpoints["jetson"].init_concurrency()
    
    client._endpoints["dgx_spark"].enabled = False
    client._endpoints["dgx_spark_2"].enabled = False
    client._roles_discovered = True

    # Slow down HTTP mock to allow queue to build up
    completion_order = []
    
    async def slow_post(url, json=None, headers=None, timeout=None):
        await asyncio.sleep(0.1)
        # message 0 is system, message 1 is user
        priority_tag = json["messages"][1]["content"] if len(json["messages"]) > 1 else ""
        completion_order.append(priority_tag)
        return mock_http.post.return_value
    
    mock_http.post.side_effect = slow_post

    with patch("app.services.vllm_client.tracker") as mock_tracker:
        mock_tracker.record = AsyncMock()
        
        # Don't start dispatcher yet, let the queue fill up
        tasks = []
        for i in range(20):
            # Interleave HIGH and NORMAL priority
            priority = Priority.HIGH if i % 2 == 0 else Priority.NORMAL
            tasks.append(
                asyncio.create_task(
                    client.chat(
                        system="system",
                        user=f"HIGH_{i}" if priority == Priority.HIGH else f"NORMAL_{i}",
                        priority=priority
                    )
                )
            )
            
        # Give chat() tasks a moment to enqueue
        await asyncio.sleep(0.1)
        
        # Now start dispatcher to process the queued items
        dispatcher_task = asyncio.create_task(client._dispatch_loop("jetson"))

        # Wait for all
        results = await asyncio.gather(*tasks, return_exceptions=True)
        dispatcher_task.cancel()
        
        successes = [r for r in results if isinstance(r, tuple) and r[0] == "mocked success"]
        assert len(successes) == 20, f"Expected 20 successes, got {len(successes)}"
        
        # Because HIGH priority was queued alongside NORMAL priority before the dispatcher
        # started, the PriorityQueue should process HIGH items first.
        high_completions = [tag for tag in completion_order if tag.startswith("HIGH")]
        normal_completions = [tag for tag in completion_order if tag.startswith("NORMAL")]
        
        assert len(high_completions) == 10
        assert len(normal_completions) == 10
        
        # Check that mostly HIGH priority was processed first
        # (The very first few might be a mix if concurrency is 5, but overall HIGH is front-loaded)
        first_half = completion_order[:10]
        assert len([tag for tag in first_half if tag.startswith("HIGH")]) >= 8, \
            "HIGH priority items should be processed before NORMAL priority"


@pytest.mark.asyncio
async def test_queue_timeout_and_eviction(mocked_vllm):
    """Verify futures timeout correctly and active semaphores are released."""
    client, mock_http = mocked_vllm
    
    client._endpoints["jetson"].enabled = True
    client._endpoints["jetson"].model = "test-model"
    client._endpoints["jetson"].max_concurrent = 2
    client._endpoints["jetson"].init_concurrency()
    client._roles_discovered = True

    # Block the HTTP mock forever
    async def forever_post(*args, **kwargs):
        await asyncio.sleep(10.0)
        return mock_http.post.return_value
    
    mock_http.post.side_effect = forever_post

    with patch("app.services.vllm_client.tracker") as mock_tracker:
        mock_tracker.record = AsyncMock()
        
        dispatcher_task = asyncio.create_task(client._dispatch_loop("jetson"))
        
        start_time = time.monotonic()
        
        t1 = asyncio.create_task(client.chat(system="system", user="1"))
        t2 = asyncio.create_task(client.chat(system="system", user="2"))
        t3 = asyncio.create_task(client.chat(system="system", user="3"))
        
        results = await asyncio.gather(t1, t2, t3, return_exceptions=True)
        duration = time.monotonic() - start_time
        dispatcher_task.cancel()

        # They should all fail with RuntimeError representing a VLLM future timeout
        for r in results:
            assert isinstance(r, RuntimeError)
            assert "vLLM future timeout" in str(r)
            
        assert 2.0 <= duration <= 2.5, f"Duration was {duration}s"
        assert client._endpoints["jetson"].queue.empty()


@pytest.mark.asyncio
async def test_hot_swap_model_stress(mocked_vllm):
    """Simulate a live model change mid-cycle.
    Verify queued items adapt and inherit the new model name before dispatching.
    """
    client, mock_http = mocked_vllm
    
    client._endpoints["jetson"].enabled = True
    client._endpoints["jetson"].model = "old-model"
    client._endpoints["jetson"].max_concurrent = 5
    client._endpoints["jetson"].init_concurrency()
    
    client._roles_discovered = True

    # Capture the payload sent to HTTP mock
    dispatched_models = []
    
    async def capture_post(url, json=None, headers=None, timeout=None):
        await asyncio.sleep(0.2)  # Block long enough to ensure max_concurrency stalls the queue
        dispatched_models.append(json.get("model", "unknown"))
        return mock_http.post.return_value
    
    mock_http.post.side_effect = capture_post

    with patch("app.services.vllm_client.tracker") as mock_tracker:
        mock_tracker.record = AsyncMock()
        
        # Enqueue requests asking for "old-model"
        tasks = []
        for i in range(10):
            tasks.append(
                asyncio.create_task(
                    client.chat(
                        system="system",
                        user=f"user_{i}",
                        model_override="old-model"
                    )
                )
            )
            
        await asyncio.sleep(0.05)
        
        # Simulating hot swap mid-queue: Background probe updates the endpoint model
        client._endpoints["jetson"].model = "new-dynamic-model"
        
        # Start dispatcher
        dispatcher_task = asyncio.create_task(client._dispatch_loop("jetson"))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        dispatcher_task.cancel()
        
        # Because the background dispatcher loop might process some items before the hot swap,
        # we should see BOTH models in the dispatched payload list. This proves that items
        # remaining in the queue successfully adapted their payloads to the new model mid-flight.
        assert len(dispatched_models) == 10
        assert "new-dynamic-model" in dispatched_models, \
            f"Expected at least some to adapt to 'new-dynamic-model', got {dispatched_models}"
        assert "old-model" in dispatched_models, \
            f"Expected some to execute before swap with 'old-model', got {dispatched_models}"


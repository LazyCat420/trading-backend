import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.vllm_client import VLLMClient, Priority

@pytest.fixture
def mocked_vllm_cb(monkeypatch):
    monkeypatch.setattr("app.services.vllm_client.settings.JETSON_VLLM_URL", "http://10.0.0.30:8000")
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_SPARK_VLLM_URL", "http://10.0.0.141:8000")
    monkeypatch.setattr("app.services.vllm_client.settings.JETSON_MAX_CONCURRENT", 10)
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_MAX_CONCURRENT", 10)
    monkeypatch.setattr("app.services.vllm_client.settings.ACTIVE_MODEL", "test-model")
    monkeypatch.setattr("app.services.vllm_client.settings.PRISM_AGENT_ROUTING", False)
    monkeypatch.setattr("app.services.vllm_client.settings.BATCH_TIMEOUT", 5.0)
    monkeypatch.setattr("app.services.vllm_client.settings.BATCH_CIRCUIT_BREAKER_THRESHOLD", 3)
    monkeypatch.setattr("app.services.vllm_client.settings.JETSON_BATCH_SIZE", 10)
    monkeypatch.setattr("app.services.vllm_client.settings.DGX_BATCH_SIZE", 10)
    monkeypatch.setattr("app.services.vllm_client.settings.VLLM_FUTURE_TIMEOUT", 60.0)
    
    client = VLLMClient()
    
    mock_http = AsyncMock()
    mock_http.is_closed = False
    client._client = mock_http
    client._get_client = AsyncMock(return_value=mock_http)
    
    return client, mock_http

@pytest.mark.asyncio
async def test_circuit_breaker_race_conditions(mocked_vllm_cb):
    """Test concurrent failures increment failure count safely and trip exactly once."""
    print("\n[DEBUG] Starting test_circuit_breaker_race_conditions")
    client, mock_http = mocked_vllm_cb
    
    ep = client._endpoints["jetson"]
    ep.enabled = True
    ep.model = "test-model"
    ep.max_concurrent = 50
    ep.batch_size = 50
    ep.queue = None
    ep.slots = None
    ep.pipeline_slots = None
    ep.init_concurrency()
    client._roles_discovered = True

    from httpx import RequestError, Request
    
    # 50 simultaneous requests hitting a network error
    async def fail_post(*args, **kwargs):
        await asyncio.sleep(0.01)
        raise RequestError("Connection reset", request=Request("POST", "http://test"))

    mock_http.post.side_effect = fail_post

    with patch("app.services.vllm_client.tracker") as mock_tracker:
        mock_tracker.record = AsyncMock()
        
        # Fire 50 requests
        print("[DEBUG] Enqueueing 50 requests...")
        tasks = []
        for i in range(50):
            tasks.append(
                asyncio.create_task(client.chat(system="system", user=f"msg {i}"))
            )
        print("[DEBUG] 50 requests enqueued.")
            
        print("[DEBUG] Starting manual dispatcher task...")
        dispatcher_task = asyncio.create_task(client._dispatch_loop(client._endpoints["jetson"]))
        print("[DEBUG] Manual dispatcher task started.")
        
        # Wait for all to fail
        print("[DEBUG] Gathering 50 request tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"[DEBUG] Gathering complete. Results size: {len(results)}")
        dispatcher_task.cancel()
        # Yield control to let the background run_and_release tasks finish processing their errors
        await asyncio.sleep(0.2)
        
        # All 50 should be RequestError (raised up from the dispatcher)
        for r in results:
            assert isinstance(r, RequestError)
            
        ep = client._endpoints["jetson"]
        
        # The batch size was 50, so this was exactly 1 batch failure!
        # wait, if max_concurrent is 50, batch_size is 50. 1 batch failure means consecutive_batch_failures=1.
        # But wait, what if they were processed in smaller batches because gather doesn't guarantee atomic batches?
        # Actually, let's verify circuit breaker threshold logic.
        assert ep.consecutive_batch_failures == 1
        assert ep.circuit_open_until < time.monotonic()  # Circuit should NOT be open yet (threshold is 3)

        # Now let's force the circuit breaker to trip by doing 2 more batches
        for batch_idx in range(2):
            print(f"[DEBUG] Starting batch {batch_idx + 1}...")
            tasks = []
            for i in range(10):
                tasks.append(asyncio.create_task(client.chat(system="system", user=f"msg {i}")))
            
            d_task = asyncio.create_task(client._dispatch_loop(client._endpoints["jetson"]))
            print(f"[DEBUG] Gathering batch {batch_idx + 1}...")
            await asyncio.gather(*tasks, return_exceptions=True)
            print(f"[DEBUG] Batch {batch_idx + 1} gathered.")
            d_task.cancel()
            # Yield control to let the background run_and_release tasks finish processing their errors
            await asyncio.sleep(0.2)
            
        # Total batch failures = 3. Circuit should now be open!
        assert ep.consecutive_batch_failures == 0  # Resets to 0 after tripping
        assert ep.circuit_open_until > time.monotonic()  # Open!
        assert ep.load_score == float('inf')
        print("[DEBUG] test_circuit_breaker_race_conditions completed successfully")


@pytest.mark.asyncio
async def test_dynamic_role_discovery_under_load(mocked_vllm_cb):
    """Simulate discover_roles running mid-flight and updating the routing table."""
    client, mock_http = mocked_vllm_cb
    
    # Start with only Jetson
    jetson = client._endpoints["jetson"]
    jetson.enabled = True
    jetson.model = "test-model"
    jetson.queue = None
    jetson.slots = None
    jetson.pipeline_slots = None
    jetson.init_concurrency()
    
    # DGX Spark is initially disabled
    dgx = client._endpoints["dgx_spark"]
    dgx.enabled = False
    dgx.queue = None
    dgx.slots = None
    dgx.pipeline_slots = None
    dgx.init_concurrency()
    
    client._roles_discovered = True

    # We will use this to track where requests actually went
    routing_log = []
    
    # We patch _call_vllm_direct to bypass the actual HTTP mock but track routing
    original_call = client._call_vllm_direct
    
    async def mock_call_direct(client_http, payload, meta, start, ep=None):
        routing_log.append(ep.name)
        await asyncio.sleep(0.05)
        return "mocked", 10, 50
        
    with patch.object(client, "_call_vllm_direct", side_effect=mock_call_direct):
        with patch("app.services.vllm_client.tracker") as mock_tracker:
            mock_tracker.record = AsyncMock()
            
            # Start Jetson dispatcher
            jetson_dispatcher = asyncio.create_task(client._dispatch_loop(client._endpoints["jetson"]))
            # Start DGX dispatcher (even though disabled, it loops waiting for items)
            dgx_dispatcher = asyncio.create_task(client._dispatch_loop(client._endpoints["dgx_spark"]))
            
            # Send 5 requests (should all go to Jetson)
            t1 = [asyncio.create_task(client.chat("sys", "u")) for _ in range(5)]
            await asyncio.sleep(0.1) # let them enqueue and route
            
            # Simulating background discovery task turning DGX online with the same model
            # This makes DGX eligible for load balancing
            client._endpoints["dgx_spark"].enabled = True
            client._endpoints["dgx_spark"].model = "test-model"
            
            # Send 10 more requests
            t2 = [asyncio.create_task(client.chat("sys", "u")) for _ in range(10)]
            
            # Wait for all to finish
            await asyncio.gather(*t1, *t2)
            
            jetson_dispatcher.cancel()
            dgx_dispatcher.cancel()
            
            # The first 5 MUST have gone to Jetson
            assert all(name == "jetson" for name in routing_log[:5])
            
            # The remaining 10 should be load balanced since DGX came online
            # Because DGX had 0 active and 0 queued, it should receive the bulk of the initial burst
            dgx_count = sum(1 for name in routing_log[5:] if name == "dgx_spark")
            jetson_count = sum(1 for name in routing_log[5:] if name == "jetson")
            
            assert dgx_count > 0, "DGX should have received requests after discovery"
            # It should be roughly balanced (e.g. 5 and 5, or 6 and 4)
            assert 3 <= dgx_count <= 7
            assert 3 <= jetson_count <= 7

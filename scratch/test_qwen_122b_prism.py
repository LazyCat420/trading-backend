import httpx
import asyncio
import traceback

async def test_prism():
    prism_url = "http://10.0.0.16:7777/chat?stream=false"
    headers = {
        "Content-Type": "application/json",
        "x-project": "vllm-trading-bot",
        "x-username": "lazy-trader",
    }
    payload = {
        "provider": "vllm-2", # let's try vllm-2 or vllm
        "model": "Qwen/Qwen3.5-122B-A10B-FP8",
        "messages": [{"role": "user", "content": "Hello"}],
        "maxTokens": 100,
        "temperature": 0.3,
        "conversationId": "test-conv-id-dgx-qwen122b",
        "project": "vllm-trading-bot",
        "username": "lazy-trader",
        "agent": "CUSTOM_MARKET_ALPHA",
        "functionCallingEnabled": False,
        "agenticLoopEnabled": False,
        "systemPrompt": "You are a helpful assistant.",
    }
    async with httpx.AsyncClient() as client:
        try:
            print("Trying provider=vllm-2...")
            r = await client.post(prism_url, json=payload, headers=headers, timeout=60.0)
            print("Status Code:", r.status_code)
            print("Body:", r.text)
            
            print("\nTrying provider=vllm...")
            payload["provider"] = "vllm"
            r = await client.post(prism_url, json=payload, headers=headers, timeout=60.0)
            print("Status Code:", r.status_code)
            print("Body:", r.text)
        except Exception as e:
            print("Error occurred:")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_prism())

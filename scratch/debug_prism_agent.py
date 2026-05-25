import os
import sys
import asyncio
import httpx
import json

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.config import settings
from app.services.prism_client import PrismClient

async def main():
    prism = PrismClient()
    print("Prism URL:", prism.url)
    
    # Simple message payload
    messages = [
        {"role": "user", "content": "Hello, this is a test. Answer with 'OK' and nothing else."}
    ]
    
    payload, url, headers = prism.get_chat_payload_and_url(
        model="Qwen/Qwen3.5-122B-A10B-FP8",  # raw model name
        messages=messages,
        max_tokens=50,
        temperature=0.1,
        system_prompt="You are a helpful assistant.",
        agent_name="test_agent",
        ticker="AAPL",
        cycle_id="test-session",
        enable_thinking=False,
        tools=None,
        agentic_mode=True,
        provider="vllm-2",  # specify vllm-2 provider
    )
    
    print("\n--- Request ---")
    print(f"POST {url}")
    print(f"Headers: {headers}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=20.0)
            print(f"\n--- Response status: {resp.status_code} ---")
            print(f"Headers: {resp.headers}")
            print(f"Body: {resp.text}")
    except httpx.HTTPStatusError as hse:
        print(f"\nHTTPStatusError: {hse}")
        print(f"Response: {hse.response.text}")
    except Exception as e:
        import traceback
        print(f"\nException: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

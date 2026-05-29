import httpx
import time
import json

URL = "http://10.0.0.141:8000/v1/chat/completions"
MODEL = "cyankiwi/MiniMax-M2.7-AWQ-4bit"

async def test_speed():
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short poem about trading stocks in 3 sentences."}
        ],
        "temperature": 0.3,
        "max_tokens": 100
    }
    
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(URL, json=payload)
            r.raise_for_status()
            elapsed = time.monotonic() - start
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            tps = comp_tokens / elapsed if elapsed > 0 else 0
            print(f"Success! Elapsed: {elapsed:.2f}s | Completion tokens: {comp_tokens} | Speed: {tps:.2f} tok/s")
            print(f"Response: {text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_speed())

import asyncio
import aiohttp
import time

async def test_ping(url):
    print(f"Testing ping to {url} ...")
    payload = {
        "model": "hermes-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    headers = {
        "Authorization": "Bearer change-me-local-dev",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        start = time.monotonic()
        try:
            async with session.post(url, json=payload, headers=headers, timeout=10.0) as resp:
                elapsed = time.monotonic() - start
                print(f"  POST {url} -> status {resp.status} in {elapsed:.2f}s")
                if resp.status == 200:
                    data = await resp.json()
                    print(f"    response: {data}")
                else:
                    text = await resp.text()
                    print(f"    error text: {text}")
        except Exception as e:
            elapsed = time.monotonic() - start
            print(f"  POST {url} -> failed in {elapsed:.2f}s: {e}")

async def main():
    endpoints = [
        "http://10.0.0.30:8642/v1/chat/completions",
        "http://10.0.0.141:8642/v1/chat/completions"
    ]
    for ep in endpoints:
        await test_ping(ep)

if __name__ == "__main__":
    asyncio.run(main())

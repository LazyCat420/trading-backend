import asyncio
import aiohttp

async def test_health(url):
    print(f"Testing {url} ...")
    async with aiohttp.ClientSession() as session:
        # Test /health
        health_url = url.replace("/v1/chat/completions", "/health")
        try:
            async with session.get(health_url, timeout=3.0) as resp:
                print(f"  GET {health_url} -> status {resp.status}")
        except Exception as e:
            print(f"  GET {health_url} -> failed: {e}")
            
        # Test /v1/models
        models_url = url.replace("/v1/chat/completions", "/v1/models")
        try:
            async with session.get(models_url, timeout=3.0) as resp:
                print(f"  GET {models_url} -> status {resp.status}")
                if resp.status == 200:
                    data = await resp.json()
                    print(f"    models: {data}")
        except Exception as e:
            print(f"  GET {models_url} -> failed: {e}")

async def main():
    endpoints = [
        "http://10.0.0.30:8642/v1/chat/completions",
        "http://10.0.0.141:8642/v1/chat/completions",
        "http://10.0.0.103:8642/v1/chat/completions"
    ]
    for ep in endpoints:
        await test_health(ep)

if __name__ == "__main__":
    asyncio.run(main())

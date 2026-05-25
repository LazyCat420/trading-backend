import os
import sys
import asyncio
import httpx

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.config import settings

async def main():
    # Test GET /v1/models on Prism Gateway
    url = f"{settings.PRISM_URL}/v1/models"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
            print(f"GET {url} response status: {resp.status_code}")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                print(f"Found {len(models)} models on Prism:")
                for m in models:
                    print(f"  - {m.get('id')}")
            else:
                print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Failed to query {url}: {e}")

if __name__ == "__main__":
    asyncio.run(main())

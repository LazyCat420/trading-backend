import os
import sys
import asyncio
import httpx

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.config import settings

async def main():
    print(f"Prism URL from settings: {settings.PRISM_URL}")
    print(f"Prism Enabled: {settings.PRISM_ENABLED}")
    print(f"Prism Routing: {settings.PRISM_AGENT_ROUTING}")
    
    url = f"{settings.PRISM_URL}/health"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
            print(f"GET {url} response status: {resp.status_code}")
            print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Failed to check health of {url}: {e}")

if __name__ == "__main__":
    asyncio.run(main())

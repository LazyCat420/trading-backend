import asyncio
import sys
import os
import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.web_tools import get_healthy_hermes_endpoints, get_hermes_session
from app.services.boot_service import BootService
from app.config import settings

async def main():
    await BootService.startup()
    try:
        session = get_hermes_session()
        hermes_endpoints = list(settings.HERMES_ENDPOINT_MAP.values())
        hermes_key = settings.API_SERVER_KEY
        
        print("Endpoints:", hermes_endpoints)
        print("Key:", hermes_key)
        
        # We will also run our own explicit ping to see the exact errors
        for ep in hermes_endpoints:
            try:
                ping_payload = {
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "stream": False,
                }
                ping_headers = {
                    "Authorization": f"Bearer {hermes_key}",
                    "Content-Type": "application/json",
                }
                ping_timeout = aiohttp.ClientTimeout(total=15.0)
                async with session.post(
                    ep, json=ping_payload, headers=ping_headers, timeout=ping_timeout
                ) as response:
                    print(f"Direct ping to {ep} -> Status: {response.status}")
                    if response.status != 200:
                        print(f"  Error body: {await response.text()}")
            except Exception as e:
                print(f"Direct ping to {ep} failed: {e}")
                
        print("\nRunning get_healthy_hermes_endpoints...")
        res = await get_healthy_hermes_endpoints(hermes_endpoints, hermes_key, session)
        print("Healthy endpoints detected:", res)
        
    finally:
        await BootService.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

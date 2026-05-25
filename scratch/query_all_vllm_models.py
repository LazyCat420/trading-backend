import sys
import os
import asyncio
import httpx

local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(local_dir, ".."))

from app.config import settings

async def main():
    endpoints = {
        "jetson": settings.JETSON_VLLM_URL,
        "dgx_spark": settings.DGX_SPARK_VLLM_URL,
    }
    
    async with httpx.AsyncClient() as client:
        for name, url in endpoints.items():
            print(f"--- Endpoint {name} ({url}) ---")
            try:
                r = await client.get(f"{url}/v1/models", timeout=5.0)
                print("Status Code:", r.status_code)
                if r.status_code == 200:
                    data = r.json()
                    for m in data.get("data", []):
                        print(f"  - {m.get('id')}")
                else:
                    print("Body:", r.text)
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())

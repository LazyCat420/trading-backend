import asyncio
import httpx
import json

async def main():
    prism_url = "http://10.0.0.16:7777"
    
    headers = {
        "Content-Type": "application/json",
        "x-project": "vllm-trading-bot",
        "x-username": "lazy-trader",
    }
    
    async with httpx.AsyncClient() as client:
        # Check health
        try:
            r = await client.get(f"{prism_url}/health")
            print("Health Status Code:", r.status_code)
            print("Health Response:", r.text)
        except Exception as e:
            print("Health Error:", e)
            
        # Get active models list from Prism
        try:
            r = await client.get(f"{prism_url}/v1/models", headers=headers)
            print("\n/v1/models Status Code:", r.status_code)
            if r.status_code == 200:
                print("Models:")
                for m in r.json().get("data", []):
                    print(f"  - {m.get('id')}")
            else:
                print("Response:", r.text)
        except Exception as e:
            print("/v1/models Error:", e)

if __name__ == "__main__":
    asyncio.run(main())

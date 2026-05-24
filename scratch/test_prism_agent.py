import asyncio
import httpx
from app.config import settings

async def main():
    url = f"{settings.PRISM_URL}/agent?stream=false"
    headers = {
        "x-project": "vllm-trading-bot",
        "x-username": "lazy-trader",
        "Content-Type": "application/json"
    }
    payload = {
        "provider": "vllm",
        "model": settings.ACTIVE_MODEL or "auto",
        "messages": [
            {"role": "user", "content": "Please get the market data for ticker AAPL using the appropriate tool."}
        ],
        "agent": "CUSTOM_MARKET_ALPHA",
        "maxTokens": 1024,
        "temperature": 0.3,
        "skipConversation": True,
        "autoApprove": True,
        "systemPrompt": "You are Market Alpha. Use the get_market_data tool when asked."
    }
    
    print(f"Calling Prism agent at {url}...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(url, json=payload, headers=headers)
            print(f"Status Code: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                print("Response JSON:")
                import json
                print(json.dumps(data, indent=2))
            else:
                print(f"Failed: {r.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

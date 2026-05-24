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
    
    # We explicitly prefix tool names in the system and user prompts
    payload = {
        "provider": "vllm",
        "model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
        "messages": [
            {"role": "user", "content": "Please get the market data for ticker AAPL using the mcp__lazy-tool-service__get_market_data tool."}
        ],
        "agent": "CUSTOM_MARKET_ALPHA",
        "maxTokens": 1024,
        "temperature": 0.3,
        "skipConversation": True,
        "autoApprove": True,
        "enabledTools": [
            "mcp__lazy-tool-service__get_market_data",
            "execute_python",
            "read_file"
        ],
        "systemPrompt": (
            "You are Market Alpha. You have access to the following tools: mcp__lazy-tool-service__get_market_data. "
            "When asked to get market data, you MUST invoke mcp__lazy-tool-service__get_market_data with the correct parameters."
        )
    }
    
    print(f"Calling Prism agent at {url} with a 180-second timeout...")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
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

import httpx
import asyncio
import traceback
import json

async def test_prism():
    prism_url = "http://10.0.0.16:7777/chat?stream=false"
    headers = {
        "Content-Type": "application/json",
        "x-project": "vllm-trading-bot",
        "x-username": "lazy-trader",
    }
    payload = {
        "provider": "vllm",
        "model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
        "messages": [{"role": "user", "content": "What is the price of AAPL? Please call the get_stock_price tool."}],
        "maxTokens": 1000,
        "temperature": 0.0,
        "conversationId": "test-conv-id-tools",
        "project": "vllm-trading-bot",
        "username": "lazy-trader",
        "agent": "CUSTOM_MARKET_ALPHA",
        "functionCallingEnabled": False,
        "agenticLoopEnabled": False,
        "systemPrompt": "You are a helpful assistant.",
        "tools": [
            {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given ticker symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol, e.g. AAPL"
                        }
                    },
                    "required": ["ticker"]
                }
            }
        ]
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(prism_url, json=payload, headers=headers, timeout=60.0)
            print("Status Code:", r.status_code)
            print("Body:")
            print(json.dumps(r.json(), indent=2))
        except Exception as e:
            print("Error occurred:")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_prism())

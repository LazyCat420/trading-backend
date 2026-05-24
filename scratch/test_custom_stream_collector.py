import httpx
import asyncio
import traceback
import json

async def test_prism_stream():
    prism_url = "http://10.0.0.16:7777/chat"
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
        "conversationId": "test-conv-id-stream-tools",
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
            async with client.stream("POST", prism_url, json=payload, headers=headers, timeout=60.0) as response:
                print("Status Code:", response.status_code)
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                print("[Stream Finished]")
                                continue
                            try:
                                event = json.loads(data_str)
                                print(f"Event type: {event.get('type')}")
                                if event.get("type") == "toolCall":
                                    print(f"  Tool Call: {json.dumps(event, indent=2)}")
                            except Exception as e:
                                print(f"Error parsing line: {e}")
        except Exception as e:
            print("Error occurred:")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_prism_stream())

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
    
    # We use raw tool names (e.g. get_market_data).
    # The client payload builder will automatically prefix them.
    from app.services.prism_client import PrismClient
    client = PrismClient()
    
    # Mock parameters
    messages = [
        {"role": "user", "content": "Please get the market data for ticker AAPL using the get_market_data tool."}
    ]
    system_prompt = (
        "You are Market Alpha. You have access to the following tools: get_market_data. "
        "When asked to get market data, you MUST invoke get_market_data with the correct parameters."
    )
    
    from app.tools.registry import registry
    # Only expose get_market_data to make it faster
    active_tools = registry.get_schemas_by_names(["get_market_data"])
    
    payload, target_url, headers = client.get_chat_payload_and_url(
        model="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
        system_prompt=system_prompt,
        agent_name="CUSTOM_MARKET_ALPHA",
        ticker="AAPL",
        cycle_id="",
        enable_thinking=False,
        tools=active_tools,
        agentic_mode=True,
    )
    
    print("Generated Payload (truncated options/keys):")
    import pprint
    pprint.pprint({k: v for k, v in payload.items() if k not in ["tools", "systemPrompt"]})
    print("Generated Payload systemPrompt:", payload["systemPrompt"])
    print("Generated Payload messages:", payload["messages"])
    
    print(f"\nCalling Prism agent at {target_url} with a 180-second timeout...")
    try:
        async with httpx.AsyncClient(timeout=180.0) as http_client:
            r = await http_client.post(target_url, json=payload, headers=headers)
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

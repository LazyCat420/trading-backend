import httpx
import json
import asyncio

URL = "http://10.0.0.141:8000/v1/chat/completions"
MODEL = "cyankiwi/MiniMax-M2.7-AWQ-4bit"

async def test_selector():
    system_prompt = (
        "You are an expert Tool Routing Agent for a financial trading system. "
        "Given a task and a list of available tools, select the tools needed to "
        "complete the task. Choose only what is necessary (maximum 5 tools). "
        "Output ONLY a JSON object with no explanation. Format:\n"
        '{"selected_tools": ["tool_a", "tool_b"]}'
    )
    user_prompt = (
        "Task: Formulate a bull technical thesis for CBOE.\n\n"
        "Available Tools:\n"
        "- get_technical_indicators: Get RSI, MACD, etc.\n"
        "- get_market_data: Get price history.\n"
        "- query_technical_indicator: Query specific indicator details.\n"
        "- search_database_facts: Search internal fact database.\n"
        "- get_sec_filings: Get SEC filings.\n\n"
        "Select up to 5 tools needed for this task."
    )
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 256
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(URL, json=payload)
            r.raise_for_status()
            print("Status Code:", r.status_code)
            data = r.json()
            print("Full JSON Response:")
            print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_selector())

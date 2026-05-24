import httpx
import asyncio

async def test_vllm():
    vllm_url = "http://10.0.0.141:8000/v1/chat/completions"
    payload = {
        "model": "qwen3.5-122b-a10b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(vllm_url, json=payload, timeout=5.0)
            print("Status Code:", r.status_code)
            print("Body:", r.text)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_vllm())

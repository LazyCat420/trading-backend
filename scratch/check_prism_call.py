import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)

from app.services.vllm_client import llm
from app.config import settings

async def main():
    print("Prism Configured URL:", settings.PRISM_URL)
    print("Prism Enabled:", settings.PRISM_ENABLED)
    print("Prism Agent Routing:", settings.PRISM_AGENT_ROUTING)
    
    print("\nDiscovering endpoints...")
    await llm.discover_roles()
    
    print("\nMaking test chat call to DGX Spark...")
    try:
        resp, tokens, latency = await llm.chat(
            system="You are a helpful assistant.",
            user="Hello, reply with only the word 'pong'.",
            agent_name="test_agent_dgx",
            ticker="TEST",
            endpoint_override="dgx_spark"
        )
        print(f"\nResponse: {resp}")
        print(f"Tokens: {tokens}, Latency: {latency}ms")
    except Exception as e:
        print(f"Error calling LLM: {e}")

if __name__ == "__main__":
    asyncio.run(main())

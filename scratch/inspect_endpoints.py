import asyncio
from app.services.vllm_client import llm

async def main():
    await llm.discover_roles()
    print("=== Registered Endpoints ===")
    for name, ep in llm._endpoints.items():
        print(f"Name: {name}")
        print(f"  URL: {ep.url}")
        print(f"  Model: {ep.model}")
        print(f"  Max Model Len (raw_ctx): {ep.max_model_len}")
        print(f"  Enabled: {ep.enabled}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())

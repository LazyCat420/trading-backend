import asyncio
import logging
import traceback
from app.config import settings
from app.services.vllm_client import llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_prism_direct")

async def test_health():
    logger.info("Starting Prism client diagnostic test...")
    logger.info(f"settings.PRISM_ENABLED: {settings.PRISM_ENABLED}")
    logger.info(f"settings.PRISM_AGENT_ROUTING: {settings.PRISM_AGENT_ROUTING}")
    logger.info(f"settings.PRISM_URL: {settings.PRISM_URL}")
    
    prism = llm.prism_client
    logger.info(f"PrismClient properties url: {prism.url}")
    logger.info(f"PrismClient properties enabled: {prism.enabled}")
    
    # Run check_health
    try:
        healthy = await prism.check_health()
        logger.info(f"check_health() result: {healthy}")
    except Exception as e:
        logger.error(f"check_health() failed: {e}")
        traceback.print_exc()

    # Try manual request with httpx client
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            logger.info("Sending GET request to health endpoint...")
            r = await client.get(f"{settings.PRISM_URL}/health", timeout=5.0)
            logger.info(f"Response status: {r.status_code}")
            logger.info(f"Response body: {r.text}")
    except Exception as e:
        logger.error(f"Manual connection to health endpoint failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_health())

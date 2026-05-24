import asyncio
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.web_tools import query_hermes
from app.services.boot_service import BootService

async def main():
    await BootService.startup()
    try:
        print("Querying Hermes...")
        res = await query_hermes("Test prompt")
        print("Result:", res)
    except Exception as e:
        print("Error caught:")
        traceback.print_exc()
    finally:
        await BootService.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

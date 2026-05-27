import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.boot_service import BootService
from app.pipeline.data.data_completeness import check_and_fill

async def main():
    await BootService.startup()
    print("Booted.")
    
    t0 = time.monotonic()
    try:
        print("Starting check_and_fill with 10s timeout...")
        res = await asyncio.wait_for(check_and_fill("OKLO"), timeout=10.0)
        print("Success in", time.monotonic() - t0, "s:", res)
    except asyncio.TimeoutError:
        print("Timed out as expected in", time.monotonic() - t0, "s")
    except Exception as e:
        print("Failed with exception:", e)
    finally:
        await BootService.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

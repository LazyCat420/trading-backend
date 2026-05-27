import asyncio
import sys
import os
import time
import logging
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging before any imports that might get loggers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    stream=sys.stdout
)

from app.services.boot_service import BootService
from app.cognition.orchestration.runner import execute_v2_pipeline
from app.cycle.orchestration.cycle_control import cycle_control

async def main():
    try:
        print("Starting BootService.startup()...")
        await BootService.startup()
        print("Booted.")
        
        # Resume cycle control to prevent wait_if_paused from freezing execution
        cycle_control.resume()
        print("Resumed cycle control.")
        
        t0 = time.monotonic()
        print("Running execute_v2_pipeline for OKLO...")
        res = await execute_v2_pipeline("OKLO")
        print("Completed in", time.monotonic() - t0, "s:", res)
    except KeyboardInterrupt:
        print("INTERRUPTED (KeyboardInterrupt)!")
        traceback.print_exc()
    except BaseException as e:
        print(f"CRASHED/INTERRUPTED with BaseException: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        try:
            await BootService.shutdown()
        except Exception as se:
            print("Shutdown error:", se)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Outer KeyboardInterrupt!")
        traceback.print_exc()
    except BaseException as e:
        print(f"Outer BaseException: {type(e).__name__}: {e}")
        traceback.print_exc()

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.web_tools import hermes_web_research
from app.tools.db_tools import search_internal_database
from app.services.boot_service import BootService

async def main():
    await BootService.startup()
    try:
        print("--- Testing search_internal_database ---")
        res_db = await search_internal_database(query="earnings", ticker="AAPL")
        print("DB Result:", res_db)
        
        print("\n--- Testing hermes_web_research ---")
        res_hermes = await hermes_web_research(query="AAPL recent news and sentiment", ticker="AAPL")
        print("Hermes Result:", res_hermes)
    finally:
        await BootService.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

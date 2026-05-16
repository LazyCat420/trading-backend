import aiohttp
from typing import Tuple

async def check_wikipedia(ticker: str) -> bool:
    """
    Last resort company existence check via Wikipedia API.
    Returns True if a likely company page is found for the ticker.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f'"{ticker}" company OR "{ticker}" stock',
        "utf8": "1",
        "format": "json"
    }
    
    headers = {
        "User-Agent": "VllmTradingBot/1.0"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    search_results = data.get("query", {}).get("search", [])
                    # We consider it a pass if Wikipedia has at least one relevant hit
                    if len(search_results) > 0:
                        return True
    except Exception:
        # Fall back to False if Wikipedia API fails or times out
        pass
        
    return False

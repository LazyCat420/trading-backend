import aiohttp
from app.validation.models import QuarantineReason
from typing import Tuple, Optional

async def check_finviz(ticker: str) -> Tuple[bool, Optional[QuarantineReason]]:
    """
    Checks if a ticker exists on Finviz.
    Returns a tuple of (passed, quarantine_reason).
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 429:
                    return False, QuarantineReason.RATE_LIMIT_EXCEEDED
                elif response.status == 404:
                    return False, QuarantineReason.NO_DATA
                elif response.status == 200:
                    text = await response.text()
                    # Sometimes Finviz returns 200 but renders an error page for missing tickers
                    if "We couldn't find any match" in text or "not found" in text.lower():
                        return False, QuarantineReason.NO_DATA
                    return True, None
                else:
                    return False, QuarantineReason.NO_DATA
    except Exception as e:
        err = str(e).lower()
        # Treat connection timeouts as potential rate limiting/IP blocking
        if "timeout" in err or "429" in err or "connection" in err:
            return False, QuarantineReason.RATE_LIMIT_EXCEEDED
        return False, QuarantineReason.NO_DATA

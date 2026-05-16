import asyncio
import yfinance as yf
from app.validation.models import QuarantineReason
from typing import Tuple, Optional

def _sync_check_yfinance(ticker: str) -> Tuple[bool, Optional[QuarantineReason]]:
    try:
        ticker_obj = yf.Ticker(ticker)
        # 1mo period is generally safe and captures recent data
        hist = ticker_obj.history(period="1mo")
        if hist.empty:
            # Check if it's a known error from yfinance shared state
            import yfinance.shared as shared
            if ticker in shared._ERRORS:
                err = str(shared._ERRORS[ticker]).lower()
                if "429" in err or "rate limit" in err:
                    return False, QuarantineReason.RATE_LIMIT_EXCEEDED
            return False, QuarantineReason.NO_DATA
        return True, None
    except Exception as e:
        err = str(e).lower()
        if "429" in err or "rate limit" in err:
            return False, QuarantineReason.RATE_LIMIT_EXCEEDED
        return False, QuarantineReason.NO_DATA

async def check_yfinance(ticker: str) -> Tuple[bool, Optional[QuarantineReason]]:
    """
    Checks if a ticker exists on yfinance.
    Returns a tuple of (passed, quarantine_reason).
    """
    return await asyncio.to_thread(_sync_check_yfinance, ticker)

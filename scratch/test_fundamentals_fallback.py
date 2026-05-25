import os
import sys
import asyncio
import datetime
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db
from app.collectors.yfinance_collector import collect_fundamentals

async def run_fallback_test():
    print("=" * 100)
    print("TESTING FUNDAMENTALS FALLBACK CHAIN")
    print("=" * 100)

    ticker = "TSLA"

    # Clean up fundamentals for TSLA first
    print(f"Clearing database fundamentals for {ticker}...")
    with get_db() as db:
        db.execute("DELETE FROM fundamentals WHERE ticker = %s", [ticker])

    # 1. Mock fetch_fundamentals_dict to fail, and mock Finnhub Client to return dummy data
    # This forces fallback.
    print("Mocking yfinance failure & Finnhub success...")
    
    mock_profile = {
        "marketCapitalization": 850000.5, # 850B
        "beta": 1.45,
    }
    mock_basic_financials = {
        "metric": {
            "peNormalizedTTM": 45.2,
            "bookValuePerShareAnnual": 15.6,
            "psTTM": 8.2,
            "netProfitMarginTTM": 0.12,
            "roeTTM": 0.18,
            "roaTTM": 0.09,
            "debtEquityTTM": 12.5, # 12.5%
            "currentRatioAnnual": 1.65,
            "52WeekHigh": 280.0,
            "52WeekLow": 140.0,
        }
    }

    # Setup the Finnhub mock
    mock_finnhub_client = MagicMock()
    mock_finnhub_client.company_profile2.return_value = mock_profile
    mock_finnhub_client.company_basic_financials.return_value = mock_basic_financials

    # Run the collector with mocks in place
    with patch("app.collectors.yfinance_collector.fetch_fundamentals_dict", return_value=None), \
         patch("app.config.config.settings.FINNHUB_API_KEY", "dummy_key"), \
         patch("finnhub.Client", return_value=mock_finnhub_client):
        
        success = await collect_fundamentals(ticker)
        print(f"collect_fundamentals status: {success}")

    # Verify database insertion
    with get_db() as db:
        row = db.execute(
            "SELECT source, market_cap, pe_ratio, debt_to_equity, current_ratio, beta FROM fundamentals WHERE ticker = %s",
            [ticker]
        ).fetchone()

    if row:
        print(f"Database Record Found!")
        print(f"Source: {row[0]}")
        print(f"Market Cap: {row[1]} (Expected: 850000500000.0)")
        print(f"P/E Ratio: {row[2]} (Expected: 45.2)")
        print(f"Debt/Equity: {row[3]} (Expected: 12.5)")
        print(f"Current Ratio: {row[4]} (Expected: 1.65)")
        print(f"Beta: {row[5]} (Expected: 1.45)")
        
        assert row[0] == "finnhub"
        assert row[1] == 850000500000.0
        assert row[2] == 45.2
        assert row[3] == 12.5
        assert row[4] == 1.65
        assert row[5] == 1.45
        print("Fallback Verification PASSED!")
    else:
        print("ERROR: Fundamentals were not saved to database.")

if __name__ == "__main__":
    asyncio.run(run_fallback_test())

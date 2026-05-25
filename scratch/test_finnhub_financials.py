import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.config import settings

def test_finnhub():
    key = settings.FINNHUB_API_KEY
    if not key:
        print("FINNHUB_API_KEY not set!")
        return
    import finnhub
    client = finnhub.Client(api_key=key)
    try:
        # Get profile
        profile = client.company_profile2(symbol='AAPL')
        print("Profile keys:", profile.keys())
        print("Profile marketCapitalization:", profile.get("marketCapitalization"))
        print("Profile shareOutstanding:", profile.get("shareOutstanding"))
        
        # Get basic financials
        financials = client.company_basic_financials('AAPL', 'all')
        print("Financials keys:", financials.keys())
        metric = financials.get("metric", {})
        print("Metric sample:", {k: metric.get(k) for k in list(metric.keys())[:15]})
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_finnhub()

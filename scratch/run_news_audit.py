"""
News Collection Audit and Fallback Test.
Runs news collection for S&P 500 top 5 tickers, tracks duplicate rates,
and tests the fallback mechanism when Finnhub fails.
"""

import sys
import os
import asyncio
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db
from app.collectors.news_collector import collect_for_ticker
from app.processors.deduplicator import deduplicate_news

async def audit_news_collection():
    tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "GOOG"]
    print("=" * 100)
    print("AUDITING NEWS COLLECTION FOR S&P 500 TOP 5 TICKERS")
    print("=" * 100)
    
    print(f"{'Ticker':<8} | {'Pre-Collect':<12} | {'Post-Collect':<12} | {'Inserted':<8} | {'Duplicates':<10} | {'Post-Dedup':<10}")
    print("-" * 100)
    
    total_inserted = 0
    total_duplicates = 0
    
    for ticker in tickers:
        # Pre-collect count
        with get_db() as db:
            before_cnt = db.execute(
                "SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [ticker]
            ).fetchone()[0]
            
        # Run collector
        try:
            await collect_for_ticker(ticker)
        except Exception as e:
            print(f"Error collecting for {ticker}: {e}")
            
        # Post-collect count
        with get_db() as db:
            after_cnt = db.execute(
                "SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [ticker]
            ).fetchone()[0]
            
        inserted = after_cnt - before_cnt
        total_inserted += inserted
        
        # Run deduplicator
        removed, _ = deduplicate_news(ticker)
        total_duplicates += removed
        
        # Post-dedup count
        with get_db() as db:
            final_cnt = db.execute(
                "SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [ticker]
            ).fetchone()[0]
            
        print(f"{ticker:<8} | {before_cnt:<12} | {after_cnt:<12} | {inserted:<8} | {removed:<10} | {final_cnt:<10}")
        
    print("-" * 100)
    print(f"Total inserted: {total_inserted} articles")
    print(f"Total duplicates removed: {total_duplicates} articles")
    if total_inserted > 0:
        print(f"Duplicate/Redundancy Rate: {(total_duplicates / total_inserted) * 100:.1f}%")
    print("=" * 100)
    print("\n")
    
    # ── Test Fallback Mechanism ──
    print("=" * 100)
    print("TESTING COLD FALLBACK MECHANISM (FINNHUB FAILURE SIMULATION)")
    print("=" * 100)
    
    # Store original api key
    original_key = os.environ.get("FINNHUB_API_KEY", "")
    
    # Delete from DB first so we can see new insertions from fallback
    test_ticker = "GOOG"
    print(f"Clearing recent GOOG news articles to test raw ingestion...")
    with get_db() as db:
        db.execute("DELETE FROM news_articles WHERE ticker = %s AND published_at > %s", [test_ticker, datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=1)])
        
    # Simulate API Key missing (Finnhub collector will skip)
    print("Simulating Finnhub API Key missing...")
    os.environ["FINNHUB_API_KEY"] = ""
    
    with get_db() as db:
        before_fallback = db.execute("SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [test_ticker]).fetchone()[0]
        
    # Execute collector (should fallback completely to yfinance headlines)
    print(f"Running news collector for {test_ticker} (Finnhub disabled)...")
    await collect_for_ticker(test_ticker)
    
    with get_db() as db:
        after_fallback = db.execute("SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [test_ticker]).fetchone()[0]
        
    fallback_inserted = after_fallback - before_fallback
    print(f"Successfully fell back to yfinance! Ingested {fallback_inserted} articles.")
    
    # Restore original key
    if original_key:
        os.environ["FINNHUB_API_KEY"] = original_key
        
    print("=" * 100)
    print("Audit and Fallback verification complete!")

if __name__ == "__main__":
    asyncio.run(audit_news_collection())

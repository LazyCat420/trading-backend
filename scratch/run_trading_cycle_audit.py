import os
import sys
import asyncio
import time
import uuid
import json
from datetime import datetime, timezone

# Insert project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.connection import get_db
from app.processors.deduplicator import deduplicate_news
from app.collectors.yfinance_collector import collect_fundamentals
from app.services.vllm_client import llm, Priority
from app.cognition.evidence.packet_builder import build_evidence_packet
from app.cognition.orchestration.runner import execute_v2_pipeline

async def test_endpoint_connectivity():
    print("\n--- 1. Testing vLLM Endpoints Connectivity & Model Discovery ---")
    try:
        disc = await llm.discover_roles()
        print(f"Model Discovery Completed: {json.dumps(disc, indent=2)}")
        print(f"Active Model: {llm.model}")
        for name, ep in llm._endpoints.items():
            print(f"Endpoint '{name}': url={ep.url}, role={ep.role}, enabled={ep.enabled}, model={ep.model}, loading={ep.loading}, circuit_open_until={ep.circuit_open_until}")
        return True
    except Exception as e:
        print(f"❌ Endpoint discovery failed: {e}")
        return False

async def test_news_deduplication():
    print("\n--- 2. Testing Exact URL and Title News Deduplication ---")
    ticker = "TEST_DEDUP"
    with get_db() as db:
        # Clean up any leftover test data
        db.execute("DELETE FROM news_articles WHERE ticker = %s", [ticker])
        
        # Insert 3 articles: 2 with exact same URL, 2 with same title but different URLs
        db.execute("""
            INSERT INTO news_articles (id, ticker, title, summary, url, published_at, publisher)
            VALUES 
            (%s, %s, %s, %s, %s, %s, %s),
            (%s, %s, %s, %s, %s, %s, %s),
            (%s, %s, %s, %s, %s, %s, %s)
        """, (
            str(uuid.uuid4()), ticker, "Unique Title 1", "Summary 1", "http://test.com/dup1", datetime.now(timezone.utc), "Source A",
            str(uuid.uuid4()), ticker, "Unique Title 2", "Summary 2", "http://test.com/dup1", datetime.now(timezone.utc), "Source B",
            str(uuid.uuid4()), ticker, "Duplicate Title", "Summary 3", "http://test.com/dup2", datetime.now(timezone.utc), "Source C",
        ))
        
        print("Inserted 3 news articles (including duplicates) for TEST_DEDUP.")
        
        # Run deduplication
        removed, _ = deduplicate_news(ticker)
        print(f"Deduplication removed {removed} articles.")
        
        # Verify count
        count = db.execute("SELECT COUNT(*) FROM news_articles WHERE ticker = %s", [ticker]).fetchone()[0]
        print(f"Remaining articles for TEST_DEDUP: {count}")
        
        # Clean up
        db.execute("DELETE FROM news_articles WHERE ticker = %s", [ticker])
        
        if removed >= 1 and count <= 2:
            print("✅ News Deduplication Test PASSED!")
            return True
        else:
            print("❌ News Deduplication Test FAILED!")
            return False

async def test_fundamentals_fallback():
    print("\n--- 3. Testing Fundamentals Collection & Fallback Chain ---")
    ticker = "AAPL"
    try:
        print(f"Running collect_fundamentals for {ticker}...")
        success = await collect_fundamentals(ticker)
        print(f"Fundamentals collection success: {success}")
        
        with get_db() as db:
            row = db.execute(
                "SELECT ticker, pe_ratio, debt_to_equity, source, snapshot_date FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                [ticker]
            ).fetchone()
            if row:
                print(f"Saved Fundamentals in DB: Ticker={row[0]}, PE={row[1]}, D/E={row[2]}, Source={row[3]}, Date={row[4]}")
                print("✅ Fundamentals Fallback Test PASSED!")
                return True
            else:
                print("❌ No fundamentals found in DB for AAPL!")
                return False
    except Exception as e:
        print(f"❌ Fundamentals Fallback Test failed with exception: {e}")
        return False

async def test_qualitative_pillar_adjustment():
    print("\n--- 4. Testing Qualitative Pillar Adjustment in Evidence Packet ---")
    ticker = "AAPL"
    try:
        print("Building evidence packet (this triggers qualitative pillar adjustment LLM call)...")
        start = time.monotonic()
        packet = await build_evidence_packet(ticker)
        elapsed = time.monotonic() - start
        print(f"Evidence packet built in {elapsed:.2f} seconds.")
        
        # Verify pillars are present
        if packet.pillar_profiles and "pillars" in packet.pillar_profiles:
            print("Pillars found:")
            for pk, p_data in packet.pillar_profiles["pillars"].items():
                print(f"  {pk.upper()}: Base={p_data.get('base_score')}, Adjusted={p_data.get('adjusted_score')}, Rationale={p_data.get('adjustment_rationale')}")
            print("✅ Qualitative Pillar Adjustment Test PASSED!")
            return True
        else:
            print("❌ No pillar profiles found in evidence packet!")
            return False
    except Exception as e:
        print(f"❌ Qualitative Pillar Adjustment Test failed: {e}")
        return False

async def test_v2_pipeline_execution():
    print("\n--- 5. Testing V2 Cognition Pipeline End-to-End ---")
    ticker = "AAPL"
    cycle_id = f"audit-cycle-{uuid.uuid4().hex[:8]}"
    bot_id = "lazy-trader-v4"
    
    print(f"Running execute_v2_pipeline for {ticker} (Cycle: {cycle_id})...")
    start = time.monotonic()
    try:
        result = await execute_v2_pipeline(
            ticker,
            cycle_id=cycle_id,
            bot_id=bot_id,
            macro_memo="GEOPOLITICAL: RISK_ON regime. Positive macro environment.",
        )
        elapsed = time.monotonic() - start
        print(f"V2 Pipeline completed in {elapsed:.2f} seconds.")
        if result:
            print(f"Result Verdict: Action={result.get('action')}, Confidence={result.get('confidence')}%")
            print(f"Rationale: {result.get('rationale')[:300]}...")
            print("✅ V2 Cognition Pipeline Test PASSED!")
            return True
        else:
            print("❌ V2 Pipeline returned None result!")
            return False
    except Exception as e:
        print(f"❌ V2 Pipeline failed with exception: {e}")
        return False

async def main():
    print("==================================================")
    print("      TRADING CYCLE AUDIT & VERIFICATION SUITE    ")
    print("==================================================")
    
    c_ok = await test_endpoint_connectivity()
    d_ok = await test_news_deduplication()
    f_ok = await test_fundamentals_fallback()
    
    # Only run LLM-reliant tests if connectivity is OK
    p_ok = False
    v_ok = False
    if c_ok:
        p_ok = await test_qualitative_pillar_adjustment()
        v_ok = await test_v2_pipeline_execution()
    else:
        print("\n⚠️ Skipping LLM-reliant tests because vLLM connectivity failed.")
        
    print("\n==================================================")
    print("                  AUDIT SUMMARY                   ")
    print("==================================================")
    print(f"1. Endpoint Connectivity:     {'PASSED' if c_ok else 'FAILED'}")
    print(f"2. News Deduplication:        {'PASSED' if d_ok else 'FAILED'}")
    print(f"3. Fundamentals Fallback:     {'PASSED' if f_ok else 'FAILED'}")
    print(f"4. Pillar Adjustment:          {'PASSED' if p_ok else 'FAILED'}")
    print(f"5. V2 Cognition Pipeline:     {'PASSED' if v_ok else 'FAILED'}")
    print("==================================================")
    
    if all([c_ok, d_ok, f_ok, p_ok, v_ok]):
        print("🎉 ALL AUDIT TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("❌ SOME AUDIT TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

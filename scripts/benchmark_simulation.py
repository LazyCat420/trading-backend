"""
Benchmark Simulation — Run trading cycles under different simulated market scenarios
and evaluate decision accuracy and confidence.
"""

import sys
import os
import asyncio
import time
import json
import logging

# Ensure path is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from cycle_main import run_single_cycle
from app.db.connection import get_db

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_sim")

SCENARIOS = [
    {
        "name": "Bullish Trend + Positive News",
        "trend": "bullish",
        "news_sentiment": "positive",
        "expected_action": "BUY"
    },
    {
        "name": "Bearish Trend + Negative News",
        "trend": "bearish",
        "news_sentiment": "negative",
        "expected_action": "SELL"
    },
    {
        "name": "Neutral Trend + Neutral News",
        "trend": "neutral",
        "news_sentiment": "neutral",
        "expected_action": "HOLD"
    },
    {
        "name": "Volatile Trend + Negative News",
        "trend": "volatile",
        "news_sentiment": "negative",
        "expected_action": "SELL"
    }
]

async def run_benchmark(ticker: str = "NVDA"):
    # Force simulation mode
    settings.EXECUTION_MODE = "simulation"
    
    logger.info("==============================================================")
    logger.info("WORLD SIMULATOR BENCHMARK START | Ticker: %s", ticker)
    logger.info("==============================================================")
    
    results_summary = []
    
    for idx, sc in enumerate(SCENARIOS):
        logger.info("\n--- Scenario %d/%d: %s ---", idx + 1, len(SCENARIOS), sc["name"])
        
        # Configure simulation parameters
        settings.SIMULATION_TREND = sc["trend"]
        settings.SIMULATION_NEWS_SENTIMENT = sc["news_sentiment"]
        
        cycle_id = f"sim-{sc['trend']}-{sc['news_sentiment']}-{int(time.time())}"
        
        # Run a single cycle
        try:
            # We run the cycle for the specified ticker
            cycle_summary = await run_single_cycle(tickers=[ticker], cycle_id=cycle_id)
            
            # Fetch decision from database
            with get_db() as db:
                row = db.execute(
                    "SELECT result_json FROM analysis_results WHERE cycle_id = %s AND ticker = %s",
                    [cycle_id, ticker]
                ).fetchone()
                
            if row:
                decision_data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                action = decision_data.get("action", "HOLD")
                confidence = decision_data.get("confidence", 0)
                rationale = decision_data.get("rationale", "")
            else:
                action = "NO_DECISION"
                confidence = 0
                rationale = "No decision record found in analysis_results."
                
            status = "PASS" if action == sc["expected_action"] else "FAIL"
            
            results_summary.append({
                "scenario": sc["name"],
                "trend": sc["trend"],
                "news_sentiment": sc["news_sentiment"],
                "expected": sc["expected_action"],
                "actual": action,
                "confidence": confidence,
                "status": status,
                "rationale": rationale[:200] + "..."
            })
            
            logger.info("Scenario complete. Result Action: %s | Expected: %s | Status: %s", action, sc["expected_action"], status)
            
        except Exception as e:
            logger.exception("Failed to run scenario %s", sc["name"])
            results_summary.append({
                "scenario": sc["name"],
                "trend": sc["trend"],
                "news_sentiment": sc["news_sentiment"],
                "expected": sc["expected_action"],
                "actual": "ERROR",
                "confidence": 0,
                "status": "ERROR",
                "rationale": str(e)
            })
            
    # Print final scorecard
    print("\n" + "=" * 80)
    print("                    WORLD SIMULATOR BENCHMARK SCORECARD")
    print("=" * 80)
    print(f"{'Scenario Name':<35} | {'Trend':<10} | {'Sentiment':<10} | {'Expected':<8} | {'Actual':<8} | {'Status':<6}")
    print("-" * 80)
    
    passed_count = 0
    for r in results_summary:
        print(f"{r['scenario']:<35} | {r['trend']:<10} | {r['news_sentiment']:<10} | {r['expected']:<8} | {r['actual']:<8} | {r['status']:<6}")
        if r['status'] == "PASS":
            passed_count += 1
            
    print("=" * 80)
    print(f"Accuracy: {passed_count}/{len(results_summary)} ({passed_count/len(results_summary)*100:.1f}%)")
    print("=" * 80 + "\n")
    
    # Save scorecard to scratch for reference
    scratch_dir = "/home/lazycat/.gemini/antigravity-ide/brain/3b34cc5f-f299-4df3-a1de-e6e7752f438c"
    os.makedirs(scratch_dir, exist_ok=True)
    with open(os.path.join(scratch_dir, "benchmark_scorecard.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Scorecard saved to benchmark_scorecard.json")

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    asyncio.run(run_benchmark(ticker))

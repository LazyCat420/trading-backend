import sys
import os
import asyncio
import logging
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Make logging verbose for subagent harness to see the spawn actions
logging.getLogger("app.tools.subagent_tools").setLevel(logging.DEBUG)
logging.getLogger("app.pipeline.analysis.agent_execution").setLevel(logging.INFO)

from app.pipeline.analysis.agent_execution import run_specialist_agents

async def run_live_test():
    ticker = "NVDA"
    cycle_id = "test_live_cycle_001"
    bot_id = "test_bot"
    
    print(f"--- Starting Live Sequential Pipeline Test for {ticker} ---")
    
    try:
        # Run the full sequential agent execution without mocks
        results = await run_specialist_agents(
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id
        )
        
        print("\n=== PIPELINE RESULTS ===")
        for agent, result in results.items():
            if agent == "_capsules":
                print(f"\n[{agent.upper()} ({len(result)})]")
                for c in result:
                    print(f"  - {c.agent_name.upper()}: {c.summary[:150]}... [Signal: {c.signal}, Conf: {c.confidence}]")
            else:
                resp = result.get('response', '')
                if isinstance(resp, str) and len(resp) > 200:
                    resp = resp[:200] + "... (truncated)"
                elif isinstance(resp, dict):
                    resp = json.dumps(resp, indent=2)
                print(f"\n[{agent.upper()}]\n{resp}")
                
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(run_live_test())

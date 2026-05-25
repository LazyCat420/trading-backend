import os
import sys
import asyncio
import json

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.agents.pre_trade_agent import run_pre_trade
from app.agents.base_agent import run_agent
from app.agents.pre_trade_agent import PRE_TRADE_SYSTEM_PROMPT

async def main():
    ticker = "CRWV"
    confidence = 65
    cycle_id = "test-pre-trade-diag"
    bot_id = "lazy-trader-v4"
    rationale = "A BUY is justified as the technical setup meets the 'RSI < 65' requirement..."
    
    print(f"Running pre-trade agent for {ticker} via run_agent...")
    
    result = await run_agent(
        agent_name="pre_trade",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=PRE_TRADE_SYSTEM_PROMPT,
        user_prompt=(
            f"Run the full pre-trade risk calculation chain for {ticker}.\n"
            f"The decision engine assigned a confidence of {confidence}%.\n"
            f"Sizing details / rationale from the decision engine: {rationale}\n"
            f"Calculate the appropriate position size, stop-loss, and risk/reward "
            f"ratio using your calculator tools. Then decide: APPROVE or VETO."
        ),
        max_tokens=1024,
        enable_tools=True,
    )
    
    print("\n--- AGENT RESULT ---")
    print(f"Agent name: {result.get('agent')}")
    print(f"Tokens used: {result.get('tokens_used')}")
    print(f"Execution ms: {result.get('execution_ms')}")
    print("\nResponse:")
    print(result.get("response"))
    
    # Check parsing
    from app.utils.text_utils import parse_json_response
    parsed = parse_json_response(result.get("response"))
    print("\nParsed JSON:")
    print(parsed)

if __name__ == "__main__":
    asyncio.run(main())

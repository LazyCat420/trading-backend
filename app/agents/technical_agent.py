"""
Technical Analysis Agent — uses pre-computed indicators from technical_processor.
LLM analyzes labeled signals (RSI, MACD, BB, etc.) — never calculates them.
"""

from app.agents.base_agent import run_agent
from app.processors.technical_processor import compute_technicals, get_signals

SYSTEM_PROMPT = """You are a technical analysis expert. You will receive pre-computed 
technical indicators for a stock. Your job is to INTERPRET these signals — do NOT 
recalculate anything. The numbers given to you are authoritative.

CRITICAL: ONLY reference numeric values explicitly present in the data provided.
Do NOT infer, estimate, or recall historical values from training knowledge.
If a metric is not in the provided data, state 'not available'.

Analyze the indicators and respond with JSON:
{"trend": "bullish|bearish|neutral", "signal": "BUY|SELL|HOLD", 
"confidence": 0-100, "key_signals": ["signal1", "signal2"], 
"summary": "2-3 sentence analysis"}"""


async def run(ticker: str, cycle_id: str, bot_id: str) -> dict:
    # Compute technicals if not already done
    compute_technicals(ticker)

    # Get pre-formatted signal text
    signals = get_signals(ticker)

    return await run_agent(
        agent_name="technical",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=SYSTEM_PROMPT,
        data_context=signals,
        user_prompt=f"Based on the technical data above, provide your analysis for {ticker}.",
        max_tokens=512,
        enable_dynamic_prompt=True,
        enable_tools=True,
    )

"""
Fund Flow / Institutional Signal Agent — uses pre-computed data from institutional_processor.
LLM analyzes 13F holdings and congress trades — never queries SEC EDGAR.
"""

from app.agents.base_agent import run_agent
from app.processors.institutional_processor import get_signals

SYSTEM_PROMPT = """You are an institutional flow analyst. You will receive 13F filing data 
(which large funds hold this stock) and congressional trading data. Your job is to 
INTERPRET these signals — do NOT look up additional data.

Remember: 13F filings are 45-day lagging indicators. Treat them as confirmation signals, 
not predictive. Congressional trades may indicate informed trading.

CRITICAL: ONLY reference fund names, trade amounts, and dates explicitly present in the
data provided. Do NOT fabricate or recall fund holdings from training knowledge.

Analyze and respond with JSON:
{"institutional_signal": "accumulating|neutral|distributing",
"congress_signal": "buying|neutral|selling", "notable_holders": ["fund1", "fund2"],
"confidence": 0-100, "summary": "2-3 sentence analysis"}"""


async def run(ticker: str, cycle_id: str, bot_id: str) -> dict:
    signals = get_signals(ticker)

    return await run_agent(
        agent_name="fund_flow",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=SYSTEM_PROMPT,
        data_context=signals,
        user_prompt=f"Based on the institutional and congressional data above, analyze flow signals for {ticker}.",
        max_tokens=512,
        enable_dynamic_prompt=True,
        enable_tools=True,
    )

"""
Risk Assessment Agent — uses pre-computed data from risk_processor.
LLM analyzes volatility, drawdown, and macro context — never calculates.
"""

from app.agents.base_agent import run_agent
from app.processors.risk_processor import get_signals

SYSTEM_PROMPT = """You are a risk management expert. You will receive pre-computed risk 
metrics including volatility, drawdown, and macroeconomic indicators. Your job is to 
INTERPRET these signals — do NOT recalculate anything.

CRITICAL: ONLY reference numeric values explicitly present in the data provided.
Do NOT infer, estimate, or recall historical values from training knowledge.
If a metric is not in the provided data, state 'not available'.

Analyze the risk profile and respond with JSON:
{"risk_score": 0-100, "max_position_pct": 0.01-0.10,
"stop_loss_pct": 0.03-0.15, "risk_factors": ["factor1", "factor2"],
"confidence": 0-100, "summary": "2-3 sentence analysis"}"""


async def run(ticker: str, cycle_id: str, bot_id: str) -> dict:
    signals = get_signals(ticker)

    return await run_agent(
        agent_name="risk",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=SYSTEM_PROMPT,
        data_context=signals,
        user_prompt=f"Based on the risk data above, assess the risk profile for a potential {ticker} position.",
        max_tokens=512,
        enable_dynamic_prompt=True,
        enable_tools=True,
    )

"""
Comparative Analysis Agent — interprets pre-computed peer comparison data.

LLM analyzes labeled signals from peer_comparison_processor — never calculates.
Only runs for BUY/SELL candidates to save tokens.
"""

from app.agents.base_agent import run_agent
from app.processors.peer_comparison_processor import build_comparison_context

SYSTEM_PROMPT = """You are an expert comparative analyst. You will receive pre-computed 
quantitative comparison data between a stock, its sector peer, and an uncorrelated 
outperformer. All numbers are authoritative — do NOT recalculate anything.

Your job is to INTERPRET the comparison and determine:
1. Is this stock the best capital allocation vs its peer and outperformer?
2. Is the stock trending healthily or already exhausted (momentum regime)?
3. Does the risk/reward favor this stock over alternatives?

Key signals to watch:
- EXHAUSTED regime = the rally is likely over, high risk of reversal
- BEARISH_DIVERGENCE = price making new highs but RSI declining — weakness
- BUYING_CLIMAX = huge volume spike on up day — often marks a top
- DECELERATING = momentum fading, the easy gains are behind
- EARLY_TREND or MID_TREND = healthy, the move may continue

CRITICAL: ONLY reference numeric values and signals explicitly present in the data
provided. Do NOT fabricate price targets, P/E ratios, or momentum readings.

Respond with JSON:
{"relative_signal": "PREFER_TICKER|PREFER_PEER|PREFER_OUTPERFORMER|NEUTRAL",
 "confidence": 0-100,
 "momentum_assessment": "1 sentence on trend health",
 "capital_allocation": "1 sentence on whether this is the best use of capital",
 "key_risks": ["risk1", "risk2"]}"""


async def run(
    ticker: str,
    cycle_id: str,
    bot_id: str,
    watchlist: list[str] | None = None,
) -> dict:
    """Run comparative analysis for a ticker vs its peers."""
    if watchlist is None:
        watchlist = []

    # Build pre-computed comparison context (pure math, no LLM)
    comparison_text = build_comparison_context(ticker, watchlist)

    return await run_agent(
        agent_name="comparative",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=SYSTEM_PROMPT,
        data_context=comparison_text,
        user_prompt=(
            f"Based on the peer comparison data above, assess whether {ticker} "
            f"is the best capital allocation opportunity vs its peer and "
            f"outperformer. Is the momentum healthy or exhausted?"
        ),
        max_tokens=512,
        enable_dynamic_prompt=True,
        enable_tools=True,
    )

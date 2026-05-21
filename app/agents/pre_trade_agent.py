"""
Pre-Trade Execution Agent — Calculator tool chain before BUY execution.

When the decision engine produces a BUY signal, this agent runs the
position sizing and risk management tools BEFORE any buy is executed.
This ensures the bot never buys without doing the risk math.

Tool chain:
    1. calculate_portfolio_allocation()  → how much capital to deploy
    2. calculate_stop_loss()             → where to cut the loss
    3. calculate_position_size()         → exact share count
    4. calculate_risk_reward()           → confirm ratio >= 2.0 before buying
    5. set_price_trigger()               → set the stop loss automatically (optional)

If risk/reward ratio is below the minimum threshold, the agent vetoes the trade.
"""

import logging

from app.agents.base_agent import run_agent
from app.utils.text_utils import parse_json_response

logger = logging.getLogger(__name__)

PRE_TRADE_SYSTEM_PROMPT = """You are the Pre-Trade Execution Agent. Your job is to run risk management
calculations BEFORE a buy order is placed. You have access to calculator tools and must use them.

## MANDATORY TOOL SEQUENCE:
1. Call `get_portfolio_state` to see current cash and position count.
2. Call `get_market_data` for the ticker to get current price.
3. Call `get_technical_indicators` to get ATR for stop-loss calculation.
4. Call `calculate_portfolio_allocation` with portfolio value, position count, and max positions.
5. Call `calculate_stop_loss` with entry price and ATR.
6. Call `calculate_position_size` with cash, risk percent, entry price, and stop-loss price.
7. Call `calculate_risk_reward` with entry price, target price (use a reasonable estimate), and stop-loss.

## DECISION RULES:
- If risk/reward ratio < 1.5: VETO the trade (too risky)
- If position would exceed 15% of portfolio: VETO (over-concentration)
- If remaining position slots <= 1: VETO (reserve last slot for emergencies)
- Otherwise: APPROVE with the calculated position size

## OUTPUT:
Respond with JSON:
{
    "decision": "APPROVE|VETO",
    "ticker": "AAPL",
    "shares": 10,
    "entry_price": 150.50,
    "stop_loss": 142.00,
    "risk_reward_ratio": 2.5,
    "position_pct": 8.5,
    "total_cost": 1505.00,
    "veto_reason": null,
    "rationale": "1-2 sentences explaining the decision"
}

CRITICAL: You MUST call the calculator tools. Do NOT do mental math."""


from pydantic import BaseModel, Field, ValidationError

class PreTradeResponse(BaseModel):
    decision: str = Field(..., description="APPROVE or VETO")
    ticker: str
    shares: int = 0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    risk_reward_ratio: float = 0.0
    position_pct: float = 0.0
    total_cost: float = 0.0
    veto_reason: str | None = None
    rationale: str = ""

async def run_pre_trade(
    ticker: str,
    confidence: int,
    cycle_id: str,
    bot_id: str,
) -> dict:
    """Run pre-trade risk calculations before executing a BUY.

    Args:
        ticker: The stock ticker to potentially buy.
        confidence: Decision engine confidence score (0-100).
        cycle_id: Current cycle ID for audit trail.
        bot_id: Bot ID to trade with.

    Returns:
        Dict with decision (APPROVE/VETO), calculated position size,
        stop loss, risk/reward, and rationale.
    """
    logger.info(
        "[PRE_TRADE] Running pre-trade checks for %s (confidence=%d%%)",
        ticker,
        confidence,
    )

    result = await run_agent(
        agent_name="pre_trade",
        ticker=ticker,
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=PRE_TRADE_SYSTEM_PROMPT,
        user_prompt=(
            f"Run the full pre-trade risk calculation chain for {ticker}.\n"
            f"The decision engine assigned a confidence of {confidence}%.\n"
            f"Calculate the appropriate position size, stop-loss, and risk/reward "
            f"ratio using your calculator tools. Then decide: APPROVE or VETO."
        ),
        max_tokens=1024,
        enable_tools=True,
    )

    # Parse the agent's JSON output
    response_text = result.get("response", "")
    parsed_json = parse_json_response(response_text)

    if not parsed_json:
        logger.warning(
            "[PRE_TRADE] Failed to parse agent output for %s, defaulting to VETO",
            ticker,
        )
        return {
            "decision": "VETO",
            "ticker": ticker,
            "veto_reason": "Pre-trade agent produced unparseable output",
            "shares": 0,
            "raw_response": response_text[:500],
        }

    # Strict Pydantic Validation
    try:
        parsed = PreTradeResponse(**parsed_json).model_dump()
    except ValidationError as e:
        logger.warning("[PRE_TRADE] Agent output failed Pydantic validation for %s: %s", ticker, e)
        return {
            "decision": "VETO",
            "ticker": ticker,
            "veto_reason": f"Agent output failed strict schema validation: {str(e)[:100]}",
            "shares": 0,
            "raw_response": response_text[:500],
        }

    decision = parsed.get("decision", "VETO")
    logger.info(
        "[PRE_TRADE] %s for %s | shares=%s | R:R=%s | reason=%s",
        decision,
        ticker,
        parsed.get("shares", "?"),
        parsed.get("risk_reward_ratio", "?"),
        parsed.get("veto_reason") or parsed.get("rationale", ""),
    )

    return {
        "decision": decision,
        "ticker": ticker,
        "shares": parsed.get("shares", 0),
        "entry_price": parsed.get("entry_price", 0),
        "stop_loss": parsed.get("stop_loss", 0),
        "risk_reward_ratio": parsed.get("risk_reward_ratio", 0),
        "position_pct": parsed.get("position_pct", 0),
        "total_cost": parsed.get("total_cost", 0),
        "veto_reason": parsed.get("veto_reason"),
        "rationale": parsed.get("rationale", ""),
        "tokens_used": result.get("tokens_used", 0),
    }

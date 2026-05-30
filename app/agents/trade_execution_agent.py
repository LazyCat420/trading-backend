"""
Trade Execution Agent — Unified sizing and risk management for BUY/SELL/HOLD.

Replaces the separate Pre-Trade Agent with a single agent that handles all
action types. Each call selects a sector-specific system prompt for the ticker,
providing specialized analyst reasoning adapted to the stock's sector.

Action-specific behavior:
  BUY:  Calculates shares, entry price, stop-loss, R:R, position size
  SELL: Evaluates partial trim (25/50/100%) vs full exit based on P&L and thesis
  HOLD: Checks trailing stop adjustments, thesis health, and convert-to-SELL triggers

This agent is ALWAYS advisory — the pipeline executes the decision engine's
action regardless. The agent's output only affects sizing and stop-loss placement.

Usage:
    from app.agents.trade_execution_agent import run_trade_execution

    result = await run_trade_execution(
        ticker="AAPL", action="BUY", confidence=82,
        cycle_id="cycle-123", bot_id="bot-1",
        rationale="Strong earnings growth", sector="Technology",
    )

    if result["decision"] == "APPROVE":
        # Use result["shares"], result["total_cost"], etc.
        pass
"""

import logging

from app.agents.base_agent import run_agent
from app.utils.text_utils import parse_json_response

logger = logging.getLogger(__name__)


# ── System Prompts by Action Type ───────────────────────────────────────

BUY_SYSTEM_PROMPT = """You are the Trade Execution Agent (BUY mode). Your job is to calculate
precise position sizing and risk management parameters before a buy order.

## MANDATORY TOOL SEQUENCE:
1. Call `get_portfolio_state` to see current cash, position count, and whether this ticker is already held.
2. Call `get_market_data` for the ticker to get current price.
3. Call `get_technical_indicators` to get ATR for stop-loss calculation.
4. Call `calculate_portfolio_allocation` with portfolio value, position count, and max positions.
5. Call `calculate_stop_loss` with entry price and ATR.
6. Call `calculate_position_size` with cash, risk percent, entry price, and stop-loss price.
7. Call `calculate_risk_reward` with entry price, target price, and stop-loss.

## SECTOR-SPECIFIC GUIDANCE:
{sector_guidance}

## POSITION SIZING RULES:
- For NEW positions: Scale between 2% and 15% of total portfolio value.
- For ADDITIONS: Check current concentration. Total must not exceed 20% of portfolio.
- Default to APPROVE unless risk/reward ratio < 1.0 or over-concentration > 20%.

## OUTPUT:
Respond with JSON:
{{
    "decision": "APPROVE|VETO",
    "ticker": "{ticker}",
    "shares": 10,
    "entry_price": 150.50,
    "stop_loss": 142.00,
    "risk_reward_ratio": 2.5,
    "position_pct": 8.5,
    "total_cost": 1505.00,
    "veto_reason": null,
    "rationale": "1-2 sentences explaining the decision"
}}

CRITICAL: You MUST call the calculator tools. Do NOT do mental math."""

SELL_SYSTEM_PROMPT = """You are the Trade Execution Agent (SELL mode). Your job is to determine
the optimal exit strategy for an existing position.

## EVALUATION STEPS:
1. Call `get_portfolio_state` to see current position details (entry price, current P&L, hold duration).
2. Call `get_market_data` to get the current price.
3. Call `get_technical_indicators` to evaluate momentum indicators (RSI, MACD, support/resistance).
4. Evaluate whether this is a full exit or partial trim based on:
   - P&L percentage (take profits on large gains, cut losses early)
   - Momentum: Is the trend accelerating or decelerating?
   - Thesis status: Has the original buy thesis been invalidated?

## SECTOR-SPECIFIC GUIDANCE:
{sector_guidance}

## EXIT SIZING RULES:
- If P&L > +20% and momentum decelerating: Suggest 50% trim
- If P&L > +40%: Suggest 75-100% exit (lock in gains)
- If P&L < -10% and thesis broken: Suggest 100% exit
- If stop-loss triggered: Always 100% exit
- Otherwise: 100% exit (default for simplicity)

## OUTPUT:
Respond with JSON:
{{
    "decision": "APPROVE",
    "ticker": "{ticker}",
    "sell_pct": 100,
    "exit_reason": "thesis_invalidated|take_profit|stop_loss|momentum_reversal|rebalance",
    "current_pnl_pct": 12.5,
    "trailing_stop_adjustment": null,
    "rationale": "1-2 sentences explaining the exit decision"
}}

CRITICAL: Default to APPROVE with 100% sell. Only suggest partial exits when clearly beneficial."""

HOLD_SYSTEM_PROMPT = """You are the Trade Execution Agent (HOLD mode). Your job is to evaluate
whether an existing HOLD position needs any adjustments.

## EVALUATION STEPS:
1. Call `get_portfolio_state` to see current position details.
2. Call `get_market_data` and `get_technical_indicators` for the current state.
3. Check:
   - Should the trailing stop be tightened? (price moved significantly in favor)
   - Has the original thesis changed? (fundamental shifts)
   - Should this convert from HOLD to SELL? (deteriorating conditions)

## SECTOR-SPECIFIC GUIDANCE:
{sector_guidance}

## OUTPUT:
Respond with JSON:
{{
    "decision": "HOLD|CONVERT_SELL",
    "ticker": "{ticker}",
    "stop_adjustment": null,
    "thesis_status": "intact|weakening|invalidated",
    "rationale": "1-2 sentences"
}}"""


# ── Sector Guidance Templates ───────────────────────────────────────────

SECTOR_GUIDANCE = {
    "technology": (
        "This is a Technology stock. Focus on TAM expansion, R&D spend efficiency, "
        "AI/cloud adoption curves, competitive moat durability, and user growth metrics. "
        "Use wider stops (ATR × 2.5) to accommodate higher volatility. "
        "Watch for earnings surprise risk and multiple compression."
    ),
    "energy": (
        "This is an Energy stock. Focus on oil/gas price sensitivity, OPEC decisions, "
        "geopolitical war premium, capex cycles, breakeven production costs, and dividend yield. "
        "Use tighter stops (ATR × 1.5) and prioritize dividend protection. "
        "Watch for commodity price correlation and regulatory risk."
    ),
    "healthcare": (
        "This is a Healthcare/Biotech stock. Focus on FDA pipeline milestones, patent cliff dates, "
        "clinical trial phase progression, and reimbursement/pricing risk. "
        "For pre-revenue biotech: cap position size at 5% due to binary event risk. "
        "Watch for PDUFA dates and competitive drug approvals."
    ),
    "financial services": (
        "This is a Financial Services stock. Focus on net interest margin (NIM) sensitivity, "
        "credit quality trends, loan loss reserve adequacy, and regulatory capital ratios. "
        "Interest rate moves dominate: rising rates help NIM but hurt bond portfolios. "
        "Watch for credit cycle turns and stress test results."
    ),
    "consumer": (
        "This is a Consumer sector stock. Focus on same-store sales trends, brand strength, "
        "consumer sentiment indicators, pricing power vs inflation, and inventory levels. "
        "Seasonal patterns matter: Q4 retail, summer travel, back-to-school. "
        "Watch for consumer confidence shifts and competitive disruption."
    ),
    "default": (
        "Apply a balanced fundamental + technical analysis approach. "
        "Use standard position sizing (Kelly-inspired, 2-10% of portfolio). "
        "No sector-specific bias. Focus on value, momentum, and risk/reward fundamentals."
    ),
}


def _get_sector_guidance(sector: str | None) -> str:
    """Look up sector-specific guidance, defaulting to balanced."""
    if not sector:
        return SECTOR_GUIDANCE["default"]
    key = sector.lower().strip()
    # Try exact match first, then partial match
    if key in SECTOR_GUIDANCE:
        return SECTOR_GUIDANCE[key]
    for k, v in SECTOR_GUIDANCE.items():
        if k in key or key in k:
            return v
    return SECTOR_GUIDANCE["default"]


def _get_sector_for_ticker(ticker: str) -> str:
    """Look up the sector for a ticker from the fundamentals table."""
    try:
        from app.db.connection import get_db
        with get_db() as db:
            row = db.execute(
                "SELECT sector FROM fundamentals WHERE ticker = %s "
                "ORDER BY snapshot_date DESC LIMIT 1",
                [ticker.upper()],
            ).fetchone()
            if row and row[0]:
                return row[0]
    except Exception:
        pass
    return "default"


def _get_prompt_template(sector: str, action: str) -> str | None:
    """Try to load a prompt template from the DB (Phase 3 integration point).

    Returns the system prompt text if found, None otherwise.
    """
    try:
        from app.db.connection import get_db
        with get_db() as db:
            row = db.execute(
                "SELECT system_prompt FROM prompt_templates "
                "WHERE sector = %s AND status = 'active' "
                "ORDER BY win_rate DESC, total_trades DESC LIMIT 1",
                [sector.lower()],
            ).fetchone()
            if row and row[0]:
                return row[0]
    except Exception:
        pass  # Table might not exist yet — fall back to hardcoded prompts
    return None


def _build_system_prompt(action: str, ticker: str, sector: str | None) -> str:
    """Build the system prompt for the given action + sector."""
    sector = sector or _get_sector_for_ticker(ticker)
    sector_guidance = _get_sector_guidance(sector)

    # Try DB prompt template first (Phase 3)
    db_prompt = _get_prompt_template(sector, action)
    if db_prompt:
        return db_prompt.format(ticker=ticker, sector_guidance=sector_guidance)

    # Fall back to hardcoded action-specific prompts
    if action == "BUY":
        return BUY_SYSTEM_PROMPT.format(ticker=ticker, sector_guidance=sector_guidance)
    elif action == "SELL":
        return SELL_SYSTEM_PROMPT.format(ticker=ticker, sector_guidance=sector_guidance)
    elif action == "HOLD":
        return HOLD_SYSTEM_PROMPT.format(ticker=ticker, sector_guidance=sector_guidance)
    else:
        return BUY_SYSTEM_PROMPT.format(ticker=ticker, sector_guidance=sector_guidance)


# ── Pydantic Models ────────────────────────────────────────────────────

from pydantic import BaseModel, Field, ValidationError


class BuyResponse(BaseModel):
    decision: str = Field(..., description="APPROVE or VETO")
    ticker: str = ""
    shares: int = 0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    risk_reward_ratio: float = 0.0
    position_pct: float = 0.0
    total_cost: float = 0.0
    veto_reason: str | None = None
    rationale: str = ""


class SellResponse(BaseModel):
    decision: str = Field(default="APPROVE")
    ticker: str = ""
    sell_pct: int = 100
    exit_reason: str = ""
    current_pnl_pct: float = 0.0
    trailing_stop_adjustment: float | None = None
    rationale: str = ""


class HoldResponse(BaseModel):
    decision: str = Field(default="HOLD")
    ticker: str = ""
    stop_adjustment: float | None = None
    thesis_status: str = "intact"
    rationale: str = ""


# ── Main Entry Point ──────────────────────────────────────────────────

async def run_trade_execution(
    ticker: str,
    action: str,
    confidence: int,
    cycle_id: str,
    bot_id: str,
    rationale: str = "",
    sector: str | None = None,
) -> dict:
    """Unified trade execution agent — handles BUY/SELL/HOLD.

    Args:
        ticker: Stock ticker.
        action: BUY, SELL, or HOLD.
        confidence: Decision engine confidence (0-100).
        cycle_id: Current cycle ID.
        bot_id: Bot ID.
        rationale: Decision engine rationale/thesis.
        sector: Optional sector override (auto-detected if None).

    Returns:
        Dict with decision, sizing info, and rationale.
        Always advisory — never blocks the pipeline.
    """
    action = action.upper()
    if action not in ("BUY", "SELL", "HOLD"):
        logger.warning("[TRADE_EXEC] Unknown action '%s', treating as BUY", action)
        action = "BUY"

    logger.info(
        "[TRADE_EXEC] Running %s execution for %s (confidence=%d%%)",
        action, ticker, confidence,
    )

    system_prompt = _build_system_prompt(action, ticker, sector)

    user_prompt = (
        f"Run the trade execution analysis for {ticker} ({action} signal).\n"
        f"Confidence from decision engine: {confidence}%.\n"
        f"Rationale: {rationale}\n"
        f"Calculate the appropriate parameters using your tools, then decide."
    )

    try:
        result = await run_agent(
            agent_name="trade_execution",
            ticker=ticker,
            cycle_id=cycle_id,
            bot_id=bot_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1024,
            enable_tools=True,
        )
    except Exception as agent_err:
        logger.warning(
            "[TRADE_EXEC] Agent call failed for %s %s: %s — returning default APPROVE",
            action, ticker, agent_err,
        )
        return _default_response(action, ticker)

    # Parse the agent's JSON output
    response_text = result.get("response", "")
    try:
        parsed_json = parse_json_response(response_text)
    except Exception as parse_err:
        logger.warning(
            "[TRADE_EXEC] parse_json_response failed for %s %s: %s — returning default",
            action, ticker, parse_err,
        )
        parsed_json = {}

    if not parsed_json:
        logger.warning(
            "[TRADE_EXEC] Failed to parse agent output for %s %s — returning default",
            action, ticker,
        )
        return _default_response(action, ticker, response_text, result.get("tokens_used", 0))

    # Validate with Pydantic (fail-open)
    parsed = _validate_response(action, ticker, parsed_json)

    decision = parsed.get("decision", "APPROVE")
    logger.info(
        "[TRADE_EXEC] %s for %s %s | reason=%s",
        decision, action, ticker,
        parsed.get("veto_reason") or parsed.get("rationale", ""),
    )

    # Always include standard fields
    parsed["ticker"] = ticker
    parsed["action"] = action
    parsed["tokens_used"] = result.get("tokens_used", 0)

    return parsed


def _validate_response(action: str, ticker: str, parsed_json: dict) -> dict:
    """Validate parsed JSON against the action-specific model. Fail-open."""
    model_map = {"BUY": BuyResponse, "SELL": SellResponse, "HOLD": HoldResponse}
    model = model_map.get(action, BuyResponse)

    try:
        validated = model(**parsed_json).model_dump()
        return validated
    except ValidationError as e:
        logger.warning(
            "[TRADE_EXEC] Pydantic validation failed for %s %s: %s — using partial data",
            action, ticker, e,
        )
        # Use whatever we could parse, fill gaps with defaults
        defaults = model.model_construct().model_dump()
        defaults.update({k: v for k, v in parsed_json.items() if v is not None})
        defaults["decision"] = parsed_json.get("decision", "APPROVE")
        return defaults


def _default_response(
    action: str, ticker: str,
    raw_response: str = "", tokens_used: int = 0,
) -> dict:
    """Return a safe default response when the agent fails."""
    if action == "BUY":
        return {
            "decision": "APPROVE",
            "ticker": ticker,
            "action": "BUY",
            "shares": 0,
            "total_cost": 0,
            "stop_loss": 0,
            "risk_reward_ratio": 0,
            "position_pct": 0,
            "entry_price": 0,
            "veto_reason": None,
            "rationale": "Trade execution agent unavailable — approved with Kelly fallback",
            "raw_response": raw_response[:500] if raw_response else "",
            "tokens_used": tokens_used,
        }
    elif action == "SELL":
        return {
            "decision": "APPROVE",
            "ticker": ticker,
            "action": "SELL",
            "sell_pct": 100,
            "exit_reason": "agent_fallback",
            "rationale": "Trade execution agent unavailable — defaulting to full exit",
            "tokens_used": tokens_used,
        }
    else:  # HOLD
        return {
            "decision": "HOLD",
            "ticker": ticker,
            "action": "HOLD",
            "thesis_status": "unknown",
            "rationale": "Trade execution agent unavailable — maintaining hold",
            "tokens_used": tokens_used,
        }

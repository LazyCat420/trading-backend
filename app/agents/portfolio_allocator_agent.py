"""
Portfolio Allocator Agent — Agentic Capital Allocation with Regime Awareness.

The SOLE authority for position sizing decisions. Uses the Trading Constitution,
market regime classification, and brain graph connections to make informed,
adaptive sizing decisions.

Architecture:
  1. Calls `assess_risk_environment` (composite tool) → gets regime, portfolio,
     brain graph, and constitution rules in ONE tool call
  2. Uses regime-adjusted sizing bounds from constitution (not hardcoded)
  3. Calls `calculate_position_size` for deterministic math per ticker
  4. Outputs batch allocations with rationale

Feedback Loop (RLHF-style):
  P&L outcomes → AutoResearch → Benchmark Agent → Constitution amendments
  → sizing rules evolve automatically over cycles

Fallback: If the agent crashes or times out, the caller falls back to
the deterministic `get_size_pct()` formula in trading_phase.py.
"""

import logging
import json
from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError

from app.agents.base_agent import run_agent
from app.utils.text_utils import parse_json_response
from app.db.connection import get_db

logger = logging.getLogger(__name__)

PORTFOLIO_ALLOCATOR_SYSTEM_PROMPT = """You are the Portfolio Sizing Agent — the SOLE authority for position sizing decisions.

Your job is to evaluate proposed BUY trades as a batch and determine how much capital to allocate to each one.
You make decisions based on data, not hardcoded rules. You have tools to assess the full risk environment.

## WORKFLOW (follow this EXACTLY):

### Step 1: Assess the Risk Environment
Call `assess_risk_environment` with the list of tickers you're sizing. This composite tool returns:
- **Market Regime**: BULL/SIDEWAYS/BEAR classification with a position multiplier
- **Portfolio State**: Current cash, positions, sector breakdown
- **Brain Graph**: Sector connections and correlations between tickers
- **Constitution Rules**: Adaptive sizing parameters (min/max %, confidence thresholds)
- **Regime-Adjusted Bounds**: Pre-calculated sizing range factoring in the regime multiplier

### Step 2: Apply Regime-Adjusted Sizing
Use the `regime_adjusted_sizing` section from Step 1. The sizing range is calculated as:
- `min_pct` to `max_pct` (already regime-adjusted)
- Scale LINEARLY within this range based on signal confidence
- Higher confidence → higher allocation (up to regime-adjusted max)
- Lower confidence → lower allocation (down to min)

### Step 3: Check Graph Connections
Review the brain graph data. If two proposed BUY tickers are highly correlated (>0.70):
- Reduce the combined allocation to avoid concentration risk
- Prefer the ticker with higher conviction
If a proposed ticker is negatively correlated with existing holdings, it may serve as a hedge — factor this into sizing.

### Step 4: Calculate Exact Sizes
For each approved ticker, call `calculate_position_size` with the determined risk percentage.

### Step 5: Enforce Constraints
- Total cost of all BUY orders MUST NOT exceed available cash
- Single ticker concentration MUST NOT exceed 20% of portfolio value
- Sector concentration must respect constitution limits
- If cash is limited, prioritize highest-confidence signals first

## OUTPUT FORMAT:
Respond with JSON:
{
    "allocations": [
        {
            "ticker": "AAPL",
            "decision": "APPROVE|VETO",
            "adjusted_size_pct": 5.2,
            "shares": 10,
            "total_cost": 1505.00,
            "regime_context": "BULL @ 1.0x multiplier",
            "veto_reason": null,
            "rationale": "Why this sizing was chosen based on regime, graph, and constitution data."
        }
    ]
}

CRITICAL RULES:
1. You MUST call `assess_risk_environment` FIRST. Do not skip this step.
2. Do NOT do mental math for sizing — use the calculator tools for exact numbers.
3. Always explain HOW the regime affected your sizing in the rationale.
4. If the regime is BEAR, require EXTRA conviction for any BUY approval.
5. If two tickers share a sector or have high graph correlation, explain how you reduced combined exposure."""


class TickerAllocation(BaseModel):
    ticker: str
    decision: str = Field(..., description="APPROVE or VETO")
    adjusted_size_pct: float = Field(0.0, description="Allocated portfolio percentage")
    shares: int = Field(0, description="Calculated number of shares to trade")
    total_cost: float = Field(0.0, description="Total cost of the trade in USD")
    regime_context: str = Field("", description="Regime state that influenced this decision")
    veto_reason: Optional[str] = None
    rationale: str = ""


class PortfolioAllocatorResponse(BaseModel):
    allocations: List[TickerAllocation]


async def run_portfolio_allocator(
    decisions: list[dict],
    cycle_id: str,
    bot_id: str,
) -> dict:
    """Evaluate and adjust trade decisions as a batch using agentic sizing.

    The agent calls the composite `assess_risk_environment` tool to get
    regime, portfolio, brain graph, and constitution data in one shot,
    then makes sizing decisions based on all inputs.

    Args:
        decisions: Decisions from analysis phase.
        cycle_id: Current cycle ID for tracking.
        bot_id: Bot ID to query portfolio from.

    Returns:
        Dict mapping ticker -> allocation decisions.
    """
    buy_decisions = [d for d in decisions if d.get("action") == "BUY" and d.get("confidence", 0) > 0]
    if not buy_decisions:
        logger.info("[PORTFOLIO_ALLOCATOR] No BUY decisions to allocate capital for.")
        return {}

    tickers_to_alloc = [d["ticker"] for d in buy_decisions]
    logger.info(
        "[PORTFOLIO_ALLOCATOR] Evaluating allocations for: %s",
        ", ".join(tickers_to_alloc),
    )

    # Format the proposed trades context for the allocator
    proposed_trades_info = []
    for d in buy_decisions:
        proposed_trades_info.append(
            f"- {d['ticker']}: Action=BUY, Confidence={d['confidence']}%, "
            f"Rationale={d.get('rationale', 'N/A')}"
        )
    proposed_context = "\n".join(proposed_trades_info)

    result = await run_agent(
        agent_name="portfolio_allocator",
        ticker="_ALLOCATOR_",
        cycle_id=cycle_id,
        bot_id=bot_id,
        system_prompt=PORTFOLIO_ALLOCATOR_SYSTEM_PROMPT,
        user_prompt=(
            f"Evaluate these proposed BUY recommendations and allocate sizing:\n\n"
            f"{proposed_context}\n\n"
            f"Tickers to assess: {json.dumps(tickers_to_alloc)}\n\n"
            f"STEP 1: Call `assess_risk_environment` with the tickers list above.\n"
            f"STEP 2: Use the regime-adjusted sizing bounds from the response.\n"
            f"STEP 3: Check brain graph correlations between these tickers.\n"
            f"STEP 4: Call `calculate_position_size` for each approved ticker.\n"
            f"STEP 5: Output final allocations JSON."
        ),
        max_tokens=2000,
        enable_tools=True,
    )

    response_text = result.get("response", "")
    try:
        parsed_json = parse_json_response(response_text)
    except Exception as parse_err:
        logger.warning(
            "[PORTFOLIO_ALLOCATOR] parse_json_response failed: %s — returning empty allocations",
            parse_err,
        )
        parsed_json = {}

    if not parsed_json or "allocations" not in parsed_json:
        logger.warning(
            "[PORTFOLIO_ALLOCATOR] Agent failed to produce valid batch allocations, returning empty."
        )
        return {}

    try:
        validated = PortfolioAllocatorResponse(**parsed_json)
        allocations_map = {a.ticker: a.model_dump() for a in validated.allocations}
        logger.info(
            "[PORTFOLIO_ALLOCATOR] Batch allocation complete. Decisions: %s",
            {t: f"{a['decision']} @ {a['adjusted_size_pct']}%" for t, a in allocations_map.items()},
        )
        return allocations_map
    except ValidationError as e:
        logger.warning(
            "[PORTFOLIO_ALLOCATOR] Validation failed for allocator output: %s", e
        )
        return {}


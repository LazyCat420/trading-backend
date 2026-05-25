"""
Quant Toolset.

Provides dynamic mathematical and quantitative calculation tools for the analyst agents.
"""

import ast
import operator
import math
import logging

from pydantic import BaseModel, Field
from app.tools.registry import registry


class TickerInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g. AAPL)")

logger = logging.getLogger(__name__)

# Allowed operators for the safe math evaluator
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.BitXor: operator.xor,
}

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}


def _eval_expr(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        return _ALLOWED_OPERATORS[type(node.op)](
            _eval_expr(node.left), _eval_expr(node.right)
        )
    elif isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPERATORS[type(node.op)](_eval_expr(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS:
            args = [_eval_expr(arg) for arg in node.args]
            return _ALLOWED_FUNCTIONS[node.func.id](*args)
    raise ValueError(f"Unsupported mathematical expression logic: {type(node)}")


async def run_quant_equation(equation: str) -> str:
    """Safely execute a mathematical equation or quantitative model (e.g., DCF calculation steps, PEG ratio validation)."""
    try:
        # Prevent overly complex/dangerous exec
        if len(equation) > 200:
            return "Error: Equation too long. Break it down."

        tree = ast.parse(equation, mode="eval")
        result = _eval_expr(tree.body)

        # Format the result gracefully
        if isinstance(result, float):
            result = round(result, 4)

        logger.info(f"[QUANT] Evaluated: {equation} -> {result}")
        return f"Equation Result: {result}"

    except Exception as e:
        logger.error(f"[QUANT] Equation execution failed: {e}")
        return "Equation Error: Invalid math format. Example allowed: '((10.5 / 2.1) ** 2) * log(10)'"


registry.register(
    func=run_quant_equation,
    name="run_quant_equation",
    description="Run a specific quantitative/mathematical equation. Useful for verifying margins, P/E variants, or backing up claims with rigid calculations.",
    parameters={
        "type": "object",
        "properties": {
            "equation": {
                "type": "string",
                "description": "The math expression (e.g. '(2.45 - 1.1) / 5.0' or '15.4 * log(2)').",
            }
        },
        "required": ["equation"],
    },
    tier=0,
    source="computed",
)

@registry.register(
    name="execute_momentum_strategy",
    description="Evaluates the stock using a quantitative momentum strategy. Checks RSI, MACD, and moving averages to output a momentum score and signal.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol",
            }
        },
        "required": ["ticker"],
    },
    tier=0,
    source="computed",
    input_model=TickerInput,
)
async def execute_momentum_strategy(ticker: str) -> str:
    from app.db.connection import get_db
    with get_db() as db:
        # Fetch latest technicals + close price
        row = db.execute(
            """
            SELECT t.rsi_14, t.macd, t.macd_signal, t.sma_20, t.sma_50, t.sma_200, p.close
            FROM technicals t
            JOIN price_history p ON t.ticker = p.ticker AND t.date = p.date
            WHERE t.ticker = %s
            ORDER BY t.date DESC LIMIT 1
            """,
            [ticker]
        ).fetchone()

        if not row:
            return "Error: Insufficient technical data to evaluate momentum strategy."

        rsi, macd, macd_signal, sma_20, sma_50, sma_200, close_price = row
        
        score = 0
        reasons = []

        if rsi and 40 <= rsi <= 70:
            score += 1
            reasons.append(f"RSI is neutral-bullish ({rsi:.2f})")
        elif rsi and rsi > 70:
            score -= 1
            reasons.append(f"RSI is overbought ({rsi:.2f})")
        elif rsi and rsi < 30:
            score -= 1
            reasons.append(f"RSI is oversold/weak momentum ({rsi:.2f})")

        if macd and macd_signal and macd > macd_signal:
            score += 2
            reasons.append(f"MACD is bullish (MACD {macd:.2f} > Signal {macd_signal:.2f})")
        elif macd and macd_signal:
            score -= 1
            reasons.append(f"MACD is bearish (MACD {macd:.2f} < Signal {macd_signal:.2f})")

        if close_price and sma_20 and close_price > sma_20:
            score += 1
            reasons.append("Price is above 20-day SMA (Short-term uptrend)")
        if close_price and sma_50 and close_price > sma_50:
            score += 1
            reasons.append("Price is above 50-day SMA (Medium-term uptrend)")
        
        signal = "BUY" if score >= 3 else "HOLD" if score >= 0 else "SELL"

        report = f"### Momentum Strategy Evaluation for {ticker}\n"
        report += f"**Signal:** {signal} (Score: {score})\n"
        report += "**Factors:**\n- " + "\n- ".join(reasons)
        return report

@registry.register(
    name="execute_value_strategy",
    description="Evaluates the stock using a quantitative value strategy. Checks P/E, PEG, P/B, and Debt/Equity to output a value score and signal.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol",
            }
        },
        "required": ["ticker"],
    },
    tier=0,
    source="computed",
    input_model=TickerInput,
)
async def execute_value_strategy(ticker: str) -> str:
    from app.db.connection import get_db
    with get_db() as db:
        row = db.execute(
            """
            SELECT pe_ratio, forward_pe, peg_ratio, price_to_book, debt_to_equity
            FROM fundamentals
            WHERE ticker = %s
            ORDER BY snapshot_date DESC LIMIT 1
            """,
            [ticker]
        ).fetchone()

        if not row:
            return "Error: Insufficient fundamental data to evaluate value strategy."

        pe, fwd_pe, peg, pb, de = row
        
        score = 0
        reasons = []

        if pe and 0 < pe < 20:
            score += 1
            reasons.append(f"P/E ratio is attractive ({pe:.2f})")
        elif pe and pe >= 20:
            score -= 1
            reasons.append(f"P/E ratio is high ({pe:.2f})")

        if peg and 0 < peg < 1.5:
            score += 2
            reasons.append(f"PEG ratio indicates undervalued growth ({peg:.2f})")
        elif peg and peg >= 1.5:
            score -= 1
            reasons.append(f"PEG ratio indicates overvalued growth ({peg:.2f})")

        if pb and 0 < pb < 3:
            score += 1
            reasons.append(f"P/B ratio is attractive ({pb:.2f})")
            
        if de and de < 1.5:
            score += 1
            reasons.append(f"Debt to Equity is healthy ({de:.2f})")
        elif de and de >= 2.0:
            score -= 1
            reasons.append(f"Debt to Equity is high/risky ({de:.2f})")
            
        signal = "BUY" if score >= 3 else "HOLD" if score >= 0 else "SELL"

        report = f"### Value Strategy Evaluation for {ticker}\n"
        report += f"**Signal:** {signal} (Score: {score})\n"
        report += "**Factors:**\n- " + "\n- ".join(reasons)
        return report

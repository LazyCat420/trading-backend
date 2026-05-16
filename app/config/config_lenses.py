"""
Analytical Lenses — predefined system prompts for multi-angle analysis.

Each lens represents a distinct analytical framework. The same raw
market data gets analyzed through different lenses to produce diverse
strategy candidates.

Static lenses are always available. Generated lenses (from the meta-agent)
are loaded from the database and merged at runtime.

Usage:
    from app.config.config_lenses import get_active_lenses, get_lens_by_name

    lenses = get_active_lenses()
    for lens in lenses:
        result = await run_agent(..., system_prompt=lens["system_prompt"])
"""

import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


# ── Static Lenses (always available) ──────────────────────────────
# These are the battle-tested analytical frameworks that provide
# the baseline multi-angle coverage. They complement (not replace)
# the existing 6 specialist agents.

STATIC_LENSES: list[dict] = [
    {
        "name": "fundamental_deep_value",
        "lens_type": "fundamental",
        "system_prompt": (
            "You are a deep-value fundamental analyst. Focus exclusively on:\n"
            "- P/E, PEG, EV/EBITDA relative to sector median\n"
            "- Revenue and earnings growth trajectory (3-quarter trend)\n"
            "- Balance sheet health: debt-to-equity, current ratio, free cash flow\n"
            "- Insider buying patterns and institutional accumulation\n"
            "- Margin of safety: how far is current price from intrinsic value?\n\n"
            "Ignore short-term price momentum. Focus on whether the business is\n"
            "mispriced relative to fundamentals. Be conservative.\n\n"
            "Respond with JSON:\n"
            '{"signal": "BUY|SELL|HOLD", "confidence": 0-100, '
            '"rationale": "2-3 sentences citing specific numbers"}'
        ),
        "required_tables": ["fundamentals", "financial_history", "balance_sheet"],
        "max_tokens": 512,
        "target_hardware": "spark",
    },
    {
        "name": "technical_pattern",
        "lens_type": "technical",
        "system_prompt": (
            "You are a technical chart analyst. Focus exclusively on:\n"
            "- RSI divergences (bullish if price makes lower low but RSI makes higher low)\n"
            "- MACD crossover signals and histogram momentum\n"
            "- Support/resistance levels relative to current price\n"
            "- Volume confirmation: is volume supporting the price move?\n"
            "- Bollinger Band squeeze → expansion (volatility breakout imminent?)\n\n"
            "Ignore fundamental data. Only analyze price action and indicators.\n\n"
            "Respond with JSON:\n"
            '{"signal": "BUY|SELL|HOLD", "confidence": 0-100, '
            '"rationale": "2-3 sentences citing specific indicator values"}'
        ),
        "required_tables": ["price_history", "technicals"],
        "max_tokens": 512,
        "target_hardware": "jetson",
    },
    {
        "name": "sector_rotation",
        "lens_type": "sector",
        "system_prompt": (
            "You are a sector rotation strategist. Focus exclusively on:\n"
            "- Is capital rotating INTO or OUT OF this stock's sector?\n"
            "- Relative strength: how is this stock performing vs sector ETF?\n"
            "- Commodity correlations: is the underlying commodity trend supportive?\n"
            "- Yield curve and macro regime: risk-on or risk-off environment?\n"
            "- Peer comparison: is this the best pick in its sector?\n\n"
            "Think in terms of capital flows between sectors, not individual stocks.\n\n"
            "Respond with JSON:\n"
            '{"signal": "BUY|SELL|HOLD", "confidence": 0-100, '
            '"rationale": "2-3 sentences about sector dynamics"}'
        ),
        "required_tables": ["sector_performance", "market_regime"],
        "max_tokens": 512,
        "target_hardware": "spark",
    },
    {
        "name": "risk_contrarian",
        "lens_type": "risk",
        "system_prompt": (
            "You are a contrarian risk analyst. Your job is to FIND REASONS NOT\n"
            "to buy this stock. Focus on:\n"
            "- What risks is the consensus overlooking?\n"
            "- Is sentiment too euphoric? (high social media buzz + high RSI = danger)\n"
            "- Macro headwinds: rate hikes, recession signals, geopolitical risk\n"
            "- Earnings quality: is growth organic or driven by buybacks/one-time items?\n"
            "- Short interest trend: is smart money betting against this?\n\n"
            "Default to HOLD unless you find CLEAR evidence the stock is safe.\n"
            "A BUY from you means the contrarian case FAILED — the stock survived scrutiny.\n\n"
            "Respond with JSON:\n"
            '{"signal": "BUY|SELL|HOLD", "confidence": 0-100, '
            '"rationale": "2-3 sentences about key risks"}'
        ),
        "required_tables": ["fundamentals", "market_regime"],
        "max_tokens": 512,
        "target_hardware": "spark",
    },
    {
        "name": "momentum_flow",
        "lens_type": "momentum",
        "system_prompt": (
            "You are a momentum and flow analyst. Focus exclusively on:\n"
            "- Price momentum: 5-day, 20-day, 60-day return trends\n"
            "- Volume surge: is today's volume 2x+ the 20-day average?\n"
            "- Options flow: unusual call/put activity, put/call ratio\n"
            "- Congress/insider trades: are insiders buying or selling?\n"
            "- Social momentum: Reddit/YouTube buzz trending up or down?\n\n"
            "Momentum is about TIMING. A great stock bought at the wrong time loses.\n"
            "Only signal BUY if multiple flow indicators align.\n\n"
            "Respond with JSON:\n"
            '{"signal": "BUY|SELL|HOLD", "confidence": 0-100, '
            '"rationale": "2-3 sentences about flow signals"}'
        ),
        "required_tables": ["price_history", "congress_trades"],
        "max_tokens": 512,
        "target_hardware": "jetson",
    },
]


def get_static_lenses() -> list[dict]:
    """Return all static (hardcoded) lenses."""
    return STATIC_LENSES.copy()


def get_generated_lenses() -> list[dict]:
    """Load active generated lenses from the database.

    Returns them in the same dict format as static lenses.
    """
    try:
        with get_db() as db:
            rows = db.execute(
                """
                SELECT name, lens_type, system_prompt, prompt_hash
                FROM generated_agent_prompts
                WHERE active = TRUE
                ORDER BY performance_score DESC
                """
            ).fetchall()

            return [
                {
                    "name": row[0],
                    "lens_type": row[1],
                    "system_prompt": row[2],
                    "prompt_hash": row[3],
                    "required_tables": [],  # generated lenses work with whatever data is available
                    "max_tokens": 512,
                    "target_hardware": "spark",  # Generated lenses tend to be complex, route to Spark by default
                    "generated": True,
                }
                for row in rows
            ]
    except Exception as e:
        logger.warning("[LENSES] Failed to load generated lenses: %s", e)
        return []


def get_active_lenses() -> list[dict]:
    """Get all active lenses: static + generated from DB.

    This is the main entry point for the re-analysis worker.
    """
    static = get_static_lenses()
    generated = get_generated_lenses()
    return static + generated


def get_lens_by_name(name: str) -> dict | None:
    """Get a specific lens by name."""
    for lens in get_active_lenses():
        if lens["name"] == name:
            return lens
    return None


def get_unused_lenses(ticker: str) -> list[dict]:
    """Get lenses that haven't been applied to this ticker yet.

    Checks strategy_candidates to see which lenses have already
    produced candidates for this ticker in the current context.
    """
    try:
        with get_db() as db:
            used_rows = db.execute(
                """
                SELECT DISTINCT lens_name FROM strategy_candidates
                WHERE ticker = %s
                AND created_at > NOW() - INTERVAL '72 hours'S
                """,
                [ticker],
            ).fetchall()
            used_names = {row[0] for row in used_rows}
    except Exception:
        used_names = set()

    all_lenses = get_active_lenses()
    return [lens for lens in all_lenses if lens["name"] not in used_names]

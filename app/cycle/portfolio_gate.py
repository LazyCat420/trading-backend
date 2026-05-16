"""
Portfolio Awareness Gate -- Pre-execution filter to prevent sector overexposure.

Runs BEFORE trade execution in trading_phase.py to check:
1. Sector concentration: reject if existing positions in same sector > threshold
2. Correlation check: warn if new ticker is highly correlated with existing positions
3. Position count cap: hard limit on total concurrent positions

Usage:
    from app.cycle.portfolio_gate import check_portfolio_gate
    gate_result = check_portfolio_gate(ticker, action, bot_id)
    if gate_result["blocked"]:
        # Skip this trade
"""

import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def _load_thresholds() -> tuple[float, int, float]:
    """Load portfolio gate thresholds from Trading Constitution.

    Falls back to hardcoded defaults if the Constitution table
    doesn't exist or the rules haven't been seeded yet.
    """
    defaults = (30.0, 8, 0.70)
    try:
        from app.pipeline.trading_constitution import (
            get_constitution_param,
        )

        sector_pct = get_constitution_param("sector", "max_sector_pct", 30.0)
        max_pos = get_constitution_param("position_limits", "max_positions", 8)
        return float(sector_pct), int(max_pos), defaults[2]
    except Exception:
        return defaults


# Lazy-loaded thresholds (refreshed per call to check_portfolio_gate)
MAX_SECTOR_CONCENTRATION_PCT = 30.0  # Fallback only
MAX_CONCURRENT_POSITIONS = 8  # Fallback only
HIGH_CORRELATION_THRESHOLD = 0.70


def _get_ticker_sector(ticker: str) -> str | None:
    """Look up sector from ticker_metadata table."""
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT sector FROM ticker_metadata WHERE ticker = %s",
                [ticker.upper()],
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _get_open_positions(bot_id: str) -> list[dict]:
    """Get all open positions with their sectors."""
    try:
        with get_db() as db:
            rows = db.execute(
                """
                SELECT p.ticker, p.qty, p.avg_entry_price,
                       COALESCE(tm.sector, 'Unknown') AS sector
                FROM positions p
                LEFT JOIN ticker_metadata tm ON p.ticker = tm.ticker
                WHERE p.bot_id = %s
            """,
                [bot_id],
            ).fetchall()
            return [
                {"ticker": r[0], "qty": r[1], "entry_price": r[2], "sector": r[3]}
                for r in rows
            ]
    except Exception as e:
        logger.warning("[PIPELINE] [GATE] Failed to get positions: %s", e)
        return []


def _get_correlation(ticker_a: str, ticker_b: str) -> float | None:
    """Get stored correlation between two tickers."""
    try:
        with get_db() as db:
            row = db.execute(
                """
                SELECT correlation FROM ticker_correlations
                WHERE (ticker_a = %s AND ticker_b = %s)
                   OR (ticker_a = %s AND ticker_b = %s)
                ORDER BY computed_at DESC LIMIT 1
            """,
                [ticker_a, ticker_b, ticker_b, ticker_a],
            ).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def check_portfolio_gate(
    ticker: str,
    action: str,
    bot_id: str,
    confidence: int = 50,
) -> dict:
    """Run portfolio awareness checks before executing a trade.

    Returns:
        {
            "blocked": bool,
            "reason": str | None,
            "warnings": list[str],
            "sector": str | None,
            "position_count": int,
            "sector_count": int,
            "sector_pct": float,
        }
    """
    ticker = ticker.upper()
    result = {
        "blocked": False,
        "reason": None,
        "warnings": [],
        "sector": None,
        "position_count": 0,
        "sector_count": 0,
        "sector_pct": 0.0,
    }

    # Only gate BUY decisions
    if action != "BUY":
        return result

    # Load adaptive thresholds from Constitution
    sector_cap, pos_cap, corr_thresh = _load_thresholds()

    positions = _get_open_positions(bot_id)
    result["position_count"] = len(positions)

    # Gate 1: Position count cap
    if len(positions) >= pos_cap:
        result["blocked"] = True
        result["reason"] = (
            f"Position limit reached "
            f"({len(positions)}/{pos_cap}). "
            f"Close a position before opening new ones."
        )
        logger.warning(
            "[GATE] BLOCKED %s: position limit %d/%d",
            ticker,
            len(positions),
            pos_cap,
        )
        return result

    # Gate 2: Sector concentration
    new_sector = _get_ticker_sector(ticker)
    result["sector"] = new_sector

    if new_sector and positions:
        same_sector = [p for p in positions if p["sector"] == new_sector]
        result["sector_count"] = len(same_sector)

        if positions:
            sector_pct = (len(same_sector) / len(positions)) * 100
            result["sector_pct"] = round(sector_pct, 1)

            if sector_pct >= sector_cap and len(same_sector) >= 2:
                tickers_in_sector = [p["ticker"] for p in same_sector]
                result["blocked"] = True
                result["reason"] = (
                    f"Sector overexposure: {new_sector} is "
                    f"{sector_pct:.0f}% of portfolio "
                    f"({len(same_sector)} positions: "
                    f"{', '.join(tickers_in_sector)}). "
                    f"Max allowed: {sector_cap}%."
                )
                logger.warning(
                    "[GATE] BLOCKED %s: sector %s at %.0f%% (%d positions)",
                    ticker,
                    new_sector,
                    sector_pct,
                    len(same_sector),
                )
                return result

    # Gate 3: Correlation check (warning only, not blocking)
    for pos in positions:
        corr = _get_correlation(ticker, pos["ticker"])
        if corr is not None and abs(corr) > corr_thresh:
            warning = (
                f"High correlation ({corr:.2f}) with existing position "
                f"{pos['ticker']} ({pos['sector']})"
            )
            result["warnings"].append(warning)
            logger.info("[PIPELINE] [GATE] WARNING for %s: %s", ticker, warning)

    # Gate 4: Already holding this exact ticker
    existing = [p for p in positions if p["ticker"] == ticker]
    if existing:
        result["warnings"].append(
            f"Already holding {ticker} ({existing[0]['qty']:.2f} shares @ "
            f"${existing[0]['entry_price']:.2f}). This will ADD to position."
        )

    if result["warnings"]:
        logger.info(
            "[GATE] %s passed with %d warnings",
            ticker,
            len(result["warnings"]),
        )

    return result

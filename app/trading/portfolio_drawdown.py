"""
Portfolio Drawdown — computes max drawdown from realized trade history.

Used by the strategy auditor to report a real drawdown figure instead
of "Unknown".
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_portfolio_drawdown(db, initial_cash: float = 100_000.0) -> Optional[float]:
    """Compute max drawdown from the closed-trade PnL series.

    Builds a cumulative equity curve from trade-level PnL ordered by
    close time, then computes the standard peak-to-trough drawdown.

    Returns:
        A negative float (e.g. -0.182 for -18.2%) or None if there
        are no closed trades.
    """
    rows = db.execute(
        """
        SELECT total_pnl
        FROM trades
        WHERE status = 'CLOSED'
        ORDER BY closed_at ASC
        """
    ).fetchall()

    if not rows:
        return None

    equity = initial_cash
    peak = equity
    max_dd = 0.0

    for (pnl,) in rows:
        equity += float(pnl)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return max_dd

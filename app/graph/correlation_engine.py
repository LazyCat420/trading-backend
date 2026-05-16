"""
Correlation engine — computes pairwise Pearson correlation on daily returns.

Reads price_history, computes returns, calculates correlations for 30d/90d
windows, classifies into tiers, and writes to ticker_correlations.

Usage:
    from app.graph.correlation_engine import compute_correlations
    pairs = compute_correlations(["NVDA", "AMD", "AAPL", "XOM"])
"""

import logging
from datetime import datetime, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)

_INV_THRESHOLD = -0.4  # inversely_correlated


def _classify_tier(r: float) -> str | None:
    """Classify correlation coefficient into a tier label."""
    if r >= 0.8:
        return "highly_correlated"
    if r >= 0.6:
        return "correlated"
    if r >= 0.4:
        return "weakly_correlated"
    if r <= _INV_THRESHOLD:
        return "inversely_correlated"
    return None  # Not significant enough to store


def compute_correlations(
    tickers: list[str],
    periods: list[str] | None = None,
) -> int:
    """Compute pairwise correlations for all ticker pairs.

    Args:
        tickers: list of ticker symbols
        periods: list of period strings, default ['30d', '90d']

    Returns:
        Number of correlation pairs stored.
    """
    if periods is None:
        periods = ["30d", "90d"]

    with get_db() as db:
        stored = 0

        period_days = {"30d": 30, "90d": 90, "180d": 180, "1y": 365}

        for period in periods:
            days = period_days.get(period, 30)
            logger.info(
                "correlation: computing %s correlations for %d tickers",
                period,
                len(tickers),
            )

            # Fetch daily returns for all tickers in one query
            returns_map = _fetch_returns(tickers, days, db)

            if len(returns_map) < 2:
                logger.info(
                    "correlation: need >=2 tickers with data, got %d", len(returns_map)
                )
                continue

            # Pairwise correlation
            ticker_list = sorted(returns_map.keys())
            now = datetime.now(timezone.utc)

            for i in range(len(ticker_list)):
                for j in range(i + 1, len(ticker_list)):
                    t_a = ticker_list[i]
                    t_b = ticker_list[j]

                    r, n_points = _pearson(returns_map[t_a], returns_map[t_b])
                    if r is None or n_points < 20:
                        continue

                    tier = _classify_tier(r)
                    if tier is None:
                        continue  # Not significant

                    try:
                        db.execute(
                            "INSERT INTO ticker_correlations "
                            "(ticker_a, ticker_b, correlation, tier, period, "
                            "data_points, computed_at) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                            "ON CONFLICT (ticker_a, ticker_b, period) DO UPDATE "
                            "SET correlation=EXCLUDED.correlation, "
                            "tier=EXCLUDED.tier, data_points=EXCLUDED.data_points, "
                            "computed_at=EXCLUDED.computed_at",
                            [t_a, t_b, round(r, 4), tier, period, n_points, now],
                        )
                        stored += 1
                    except Exception as e:
                        logger.warning(
                            "correlation: store failed %s-%s: %s", t_a, t_b, e
                        )

        logger.info("correlation: stored %d pairs", stored)
        return stored


def get_correlated_tickers(
    ticker: str,
    min_r: float = 0.4,
    period: str = "30d",
    limit: int = 15,
) -> list[dict]:
    """Get tickers correlated with the given ticker.

    Returns list sorted by |correlation| descending.
    """
    with get_db() as db:
        t = ticker.upper().strip()
        try:
            rows = db.execute(
                "SELECT ticker_a, ticker_b, correlation, tier, data_points "
                "FROM ticker_correlations "
                "WHERE (ticker_a = %s OR ticker_b = %s) "
                "AND period = %s AND ABS(correlation) >= %s "
                "ORDER BY ABS(correlation) DESC LIMIT %s",
                [t, t, period, min_r, limit],
            ).fetchall()

            results = []
            for r in rows:
                other = r[1] if r[0] == t else r[0]
                results.append(
                    {
                        "ticker": other,
                        "correlation": r[2],
                        "tier": r[3],
                        "data_points": r[4],
                    }
                )
            return results
        except Exception:
            return []


def _fetch_returns(tickers: list[str], days: int, db) -> dict[str, dict[str, float]]:
    """Fetch daily returns for tickers as {ticker: {date_str: return_pct}}.

    Computes returns from close prices in price_history.
    """
    returns_map: dict[str, dict[str, float]] = {}

    if not tickers:
        return returns_map

    clean_tickers = [t.upper().strip() for t in tickers]
    placeholders = ",".join(["%s"] * len(clean_tickers))

    # We need the last `days + 1` rows for each ticker.
    # Use window function to partition by ticker.
    query = f"""
        SELECT ticker, date, close
        FROM (
            SELECT ticker, date, close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
            FROM price_history
            WHERE ticker IN ({placeholders})
        )
        WHERE rn <= %s
        ORDER BY ticker, date ASC
    """
    params = clean_tickers + [days + 1]

    try:
        rows = db.execute(query, params).fetchall()

        # Group rows by ticker
        grouped = {}
        for row in rows:
            ticker, date, close = row
            if ticker not in grouped:
                grouped[ticker] = []
            grouped[ticker].append((date, close))

        for ticker, t_rows in grouped.items():
            if len(t_rows) < 21:  # Need at least 20 returns
                continue

            daily_returns = {}
            for k in range(1, len(t_rows)):
                prev_close = t_rows[k - 1][1]
                curr_close = t_rows[k][1]
                if prev_close and prev_close > 0:
                    ret = (curr_close - prev_close) / prev_close
                    date_str = str(t_rows[k][0])
                    daily_returns[date_str] = ret

            if len(daily_returns) >= 20:
                returns_map[ticker] = daily_returns

    except Exception as e:
        logger.warning("correlation: batch returns fetch failed: %s", e)

    return returns_map


def _pearson(
    returns_a: dict[str, float],
    returns_b: dict[str, float],
) -> tuple[float | None, int]:
    """Compute Pearson correlation between two return series.

    Only uses overlapping dates. Returns (r, n_points).
    """
    common_dates = set(returns_a.keys()) & set(returns_b.keys())
    n = len(common_dates)
    if n < 20:
        return None, n

    dates = sorted(common_dates)
    a_vals = [returns_a[d] for d in dates]
    b_vals = [returns_b[d] for d in dates]

    import numpy as np

    std_a = np.std(a_vals)
    std_b = np.std(b_vals)
    if std_a == 0 or std_b == 0:
        return None, n

    r = np.corrcoef(a_vals, b_vals)[0, 1]
    if np.isnan(r):
        return None, n

    return round(float(r), 6), n

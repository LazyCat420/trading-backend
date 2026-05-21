"""
Quant Processor -- Pure math, no LLM. Computes quantitative metrics
for individual stocks and cross-stock comparisons.

Functions:
    get_zscore(ticker)          - How many std devs from mean
    get_correlations(tickers)   - Pairwise correlation matrix
    get_sharpe(ticker)          - Risk-adjusted return (annualized)
    get_sortino(ticker)         - Downside-only risk-adjusted return
    get_risk_reward(ticker, entry, stop, target) - R:R ratio
    get_beta(ticker)            - Beta vs SPY
    get_drawdown(ticker)        - Max drawdown + current drawdown
    get_ticker_score(ticker)    - Composite score 0-100
    get_relative_strength(tickers) - Rank tickers by recent performance
"""

import numpy as np
import pandas as pd
from app.db.connection import get_db


def _cursor_to_df(cursor) -> pd.DataFrame:
    """Convert a PostgreSQL cursor result to a pandas DataFrame.

    Provides standard DB-API cursor logic to convert to DataFrame.
    """
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=cols)


def _get_returns(ticker: str, days: int = 60) -> pd.Series | None:
    """Get daily returns for a ticker."""
    with get_db() as db:
        df = _cursor_to_df(
            db.execute(
                """
            SELECT date, close FROM price_history
            WHERE ticker = %s ORDER BY date DESC LIMIT %s
        """,
                [ticker, days + 1],
            )
        )
        if df.empty or len(df) < 10:
            return None
        df = df.sort_values("date")
        df["return"] = df["close"].pct_change()
        return df.dropna(subset=["return"])


def get_zscore(ticker: str, window: int = 60) -> dict:
    """Z-score: how many std deviations current price is from its rolling mean.

    |Z| > 2 = unusual move (mean reversion candidate)
    |Z| > 3 = extreme move
    """
    with get_db() as db:
        df = _cursor_to_df(
            db.execute(
                """
            SELECT date, close FROM price_history
            WHERE ticker = %s ORDER BY date DESC LIMIT %s
        """,
                [ticker, window + 1],
            )
        )

        if df.empty or len(df) < 20:
            return {"ticker": ticker, "error": "insufficient data"}

        df = df.sort_values("date")
        current_price = df["close"].iloc[-1]
        mean_price = df["close"].mean()
        std_price = df["close"].std()

        if std_price == 0:
            return {"ticker": ticker, "zscore": 0.0, "signal": "NO_VARIANCE"}

        zscore = (current_price - mean_price) / std_price

        if zscore <= -2:
            signal = "OVERSOLD_EXTREME"
        elif zscore <= -1:
            signal = "OVERSOLD"
        elif zscore >= 2:
            signal = "OVERBOUGHT_EXTREME"
        elif zscore >= 1:
            signal = "OVERBOUGHT"
        else:
            signal = "NEUTRAL"

        return {
            "ticker": ticker,
            "zscore": round(zscore, 3),
            "current_price": round(current_price, 2),
            "mean_price": round(mean_price, 2),
            "std_dev": round(std_price, 2),
            "upper_2sd": round(mean_price + 2 * std_price, 2),
            "lower_2sd": round(mean_price - 2 * std_price, 2),
            "signal": signal,
        }


def get_correlations(tickers: list[str], days: int = 60) -> dict:
    """Compute pairwise correlation matrix between tickers.

    Returns correlation values + highly correlated pairs (>0.7).
    Useful for diversification: don't buy two stocks that are 90% correlated.
    """
    with get_db() as db:
        # Build a DataFrame with closes for each ticker
        dfs = {}
        for ticker in tickers:
            df = _cursor_to_df(
                db.execute(
                    """
                SELECT date, close FROM price_history
                WHERE ticker = %s ORDER BY date DESC LIMIT %s
            """,
                    [ticker, days],
                )
            )
            if not df.empty and len(df) > 10:
                df = df.sort_values("date").set_index("date")
                dfs[ticker] = df["close"]

        if len(dfs) < 2:
            return {"error": "need at least 2 tickers with data"}

        prices = pd.DataFrame(dfs)
        returns = prices.pct_change().dropna()
        corr_matrix = returns.corr()

        # Find highly correlated pairs
        high_corr = []
        checked = set()
        for t1 in corr_matrix.columns:
            for t2 in corr_matrix.columns:
                if t1 != t2 and (t2, t1) not in checked:
                    corr = corr_matrix.loc[t1, t2]
                    if abs(corr) > 0.7:
                        high_corr.append(
                            {
                                "pair": f"{t1}/{t2}",
                                "correlation": round(corr, 3),
                                "warning": "HIGH" if abs(corr) > 0.85 else "MODERATE",
                            }
                        )
                    checked.add((t1, t2))

        # Convert matrix to dict for JSON serialization
        matrix = {}
        for t1 in corr_matrix.columns:
            matrix[t1] = {
                t2: round(corr_matrix.loc[t1, t2], 3) for t2 in corr_matrix.columns
            }

        return {
            "tickers": list(corr_matrix.columns),
            "matrix": matrix,
            "high_correlations": high_corr,
            "diversification_score": round(
                1 - corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean(), 3
            )
            if len(corr_matrix) > 1
            else 1.0,
        }


def get_sharpe(ticker: str, days: int = 252, risk_free_rate: float = 0.05) -> dict:
    """Sharpe ratio: (annualized return - risk free) / annualized volatility.

    > 1.0 = good, > 2.0 = very good, > 3.0 = excellent
    """
    data = _get_returns(ticker, days)
    if data is None:
        return {"ticker": ticker, "error": "insufficient data"}

    daily_rf = risk_free_rate / 252
    excess_returns = data["return"] - daily_rf

    annualized_return = data["return"].mean() * 252
    annualized_vol = data["return"].std() * np.sqrt(252)

    if annualized_vol == 0:
        return {"ticker": ticker, "sharpe": 0.0}

    sharpe = (annualized_return - risk_free_rate) / annualized_vol

    if sharpe >= 2.0:
        rating = "EXCELLENT"
    elif sharpe >= 1.0:
        rating = "GOOD"
    elif sharpe >= 0:
        rating = "MEDIOCRE"
    else:
        rating = "POOR"

    return {
        "ticker": ticker,
        "sharpe": round(sharpe, 3),
        "annualized_return": round(annualized_return * 100, 2),
        "annualized_volatility": round(annualized_vol * 100, 2),
        "rating": rating,
    }


def get_sortino(ticker: str, days: int = 252, risk_free_rate: float = 0.05) -> dict:
    """Sortino ratio: like Sharpe but only penalizes downside volatility.

    Better for measuring risk because upside volatility is good.
    """
    data = _get_returns(ticker, days)
    if data is None:
        return {"ticker": ticker, "error": "insufficient data"}

    daily_rf = risk_free_rate / 252
    excess_returns = data["return"] - daily_rf
    downside = excess_returns[excess_returns < 0]

    annualized_return = data["return"].mean() * 252
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0

    if downside_vol == 0:
        return {"ticker": ticker, "sortino": float("inf"), "rating": "NO_DOWNSIDE"}

    sortino = (annualized_return - risk_free_rate) / downside_vol

    return {
        "ticker": ticker,
        "sortino": round(sortino, 3),
        "annualized_return": round(annualized_return * 100, 2),
        "downside_volatility": round(downside_vol * 100, 2),
        "rating": "EXCELLENT"
        if sortino >= 2.0
        else "GOOD"
        if sortino >= 1.0
        else "POOR",
    }


def get_risk_reward(
    ticker: str,
    entry_price: float | None = None,
    stop_loss_pct: float = 0.05,
    target_pct: float = 0.10,
) -> dict:
    """Risk/Reward ratio calculation.

    R:R >= 2.0 is ideal (risking $1 to make $2).
    Uses ATR for dynamic stop-loss if no explicit stop given.
    """
    with get_db() as db:
        # Get current price if entry not specified
        if entry_price is None:
            row = db.execute(
                """
                SELECT close FROM price_history
                WHERE ticker = %s ORDER BY date DESC LIMIT 1
            """,
                [ticker],
            ).fetchone()
            if not row:
                return {"ticker": ticker, "error": "no price data"}
            entry_price = row[0]

        # Get ATR for dynamic stop
        atr_row = db.execute(
            """
            SELECT atr_14, support, resistance FROM technicals
            WHERE ticker = %s ORDER BY date DESC LIMIT 1
        """,
            [ticker],
        ).fetchone()

        if atr_row and atr_row[0]:
            atr = atr_row[0]
            support = atr_row[1] or (entry_price * (1 - stop_loss_pct))
            resistance = atr_row[2] or (entry_price * (1 + target_pct))

            # ATR-based stop: 2x ATR below entry
            atr_stop = entry_price - (2 * atr)
            # Use whichever is tighter: support level or ATR stop
            stop_price = max(atr_stop, support)
            target_price = resistance
        else:
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + target_pct)

        risk = entry_price - stop_price
        reward = target_price - entry_price

        if risk <= 0:
            return {"ticker": ticker, "error": "negative risk (stop above entry)"}

        rr_ratio = reward / risk

        return {
            "ticker": ticker,
            "entry": round(entry_price, 2),
            "stop_loss": round(stop_price, 2),
            "target": round(target_price, 2),
            "risk_per_share": round(risk, 2),
            "reward_per_share": round(reward, 2),
            "risk_reward_ratio": round(rr_ratio, 2),
            "rating": "EXCELLENT"
            if rr_ratio >= 3
            else "GOOD"
            if rr_ratio >= 2
            else "ACCEPTABLE"
            if rr_ratio >= 1
            else "POOR",
        }


def get_beta(ticker: str, benchmark: str = "SPY", days: int = 252) -> dict:
    """Beta: how volatile the stock is relative to the market (SPY).

    Beta > 1 = more volatile than market
    Beta < 1 = less volatile
    Beta < 0 = inverse correlation
    """
    with get_db() as db:
        # Need both ticker and benchmark data
        stock_df = _cursor_to_df(
            db.execute(
                """
            SELECT date, close FROM price_history
            WHERE ticker = %s ORDER BY date DESC LIMIT %s
        """,
                [ticker, days + 1],
            )
        )

        bench_df = _cursor_to_df(
            db.execute(
                """
            SELECT date, close FROM price_history
            WHERE ticker = %s ORDER BY date DESC LIMIT %s
        """,
                [benchmark, days + 1],
            )
        )

        if stock_df.empty or bench_df.empty:
            # Fallback: use fundamentals table beta
            fund_row = db.execute(
                """
                SELECT beta FROM fundamentals
                WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1
            """,
                [ticker],
            ).fetchone()
            if fund_row and fund_row[0]:
                return {
                    "ticker": ticker,
                    "beta": round(fund_row[0], 3),
                    "source": "fundamentals",
                    "rating": "HIGH_VOL"
                    if fund_row[0] > 1.5
                    else "MODERATE"
                    if fund_row[0] > 1
                    else "LOW_VOL",
                }
            return {"ticker": ticker, "error": f"no data for {ticker} or {benchmark}"}

        # Merge on date
        stock_df = stock_df.sort_values("date").set_index("date")
        bench_df = bench_df.sort_values("date").set_index("date")

        merged = pd.DataFrame(
            {
                "stock": stock_df["close"],
                "bench": bench_df["close"],
            }
        ).dropna()

        if len(merged) < 20:
            return {"ticker": ticker, "error": "insufficient overlapping data"}

        stock_returns = merged["stock"].pct_change().dropna()
        bench_returns = merged["bench"].pct_change().dropna()

        covariance = np.cov(stock_returns, bench_returns)[0][1]
        benchmark_variance = np.var(bench_returns)

        if benchmark_variance == 0:
            return {"ticker": ticker, "error": "zero benchmark variance"}

        beta = covariance / benchmark_variance

        return {
            "ticker": ticker,
            "benchmark": benchmark,
            "beta": round(beta, 3),
            "source": "computed",
            "rating": "HIGH_VOL"
            if beta > 1.5
            else "MODERATE"
            if beta > 1
            else "LOW_VOL"
            if beta > 0
            else "INVERSE",
        }


def get_drawdown(ticker: str, days: int = 252) -> dict:
    """Max drawdown and current drawdown from peak.

    Max drawdown < -20% = significant risk.
    Current drawdown near max = potential bottom.
    """
    with get_db() as db:
        df = _cursor_to_df(
            db.execute(
                """
            SELECT date, close FROM price_history
            WHERE ticker = %s ORDER BY date DESC LIMIT %s
        """,
                [ticker, days],
            )
        )

        if df.empty or len(df) < 5:
            return {"ticker": ticker, "error": "insufficient data"}

        df = df.sort_values("date")
        prices = df["close"].values

        # Rolling max (peak)
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak

        max_dd = drawdown.min()
        current_dd = drawdown[-1]
        peak_price = peak[-1]
        current_price = prices[-1]

        # Find max drawdown period
        max_dd_idx = np.argmin(drawdown)
        max_dd_date = df["date"].iloc[max_dd_idx]

        return {
            "ticker": ticker,
            "max_drawdown_pct": round(max_dd * 100, 2),
            "current_drawdown_pct": round(current_dd * 100, 2),
            "peak_price": round(peak_price, 2),
            "current_price": round(current_price, 2),
            "max_drawdown_date": str(max_dd_date),
            "recovery_needed_pct": round((peak_price / current_price - 1) * 100, 2)
            if current_price > 0
            else 0,
            "rating": "CRITICAL"
            if current_dd < -0.30
            else "DEEP"
            if current_dd < -0.15
            else "MODERATE"
            if current_dd < -0.05
            else "SHALLOW",
        }


def get_ticker_score(ticker: str) -> dict:
    """Composite score 0-100 aggregating all quant + technical signals.

    Weights:
        Technical momentum (RSI, MACD, Stoch): 25%
        Valuation (P/E, P/B):                  20%
        Risk-adjusted return (Sharpe):          15%
        Volatility (ATR, Beta):                 15%
        Mean reversion (Z-score):               15%
        Drawdown severity:                      10%
    """
    with get_db() as db:
        scores = {}

        # -- Technical momentum (25%) --
        tech = db.execute(
            """
            SELECT rsi_14, macd_hist, stoch_k, adx_14
            FROM technicals WHERE ticker = %s ORDER BY date DESC LIMIT 1
        """,
            [ticker],
        ).fetchone()

        if tech:
            rsi, macd_h, stoch_k, adx = tech
            # RSI: oversold (30) = high score, overbought (70) = low score for buy signals
            rsi_score = max(0, min(100, (70 - (rsi or 50)) * 2 + 50))
            # MACD histogram positive = bullish
            macd_score = 70 if (macd_h or 0) > 0 else 30
            # Stochastic oversold = good
            stoch_score = max(0, min(100, (80 - (stoch_k or 50)) * 1.5 + 50))
            scores["technical"] = rsi_score * 0.4 + macd_score * 0.3 + stoch_score * 0.3
        else:
            scores["technical"] = 50

        # -- Valuation (20%) --
        fund = db.execute(
            """
            SELECT pe_ratio, forward_pe, price_to_book, revenue_growth
            FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1
        """,
            [ticker],
        ).fetchone()

        if fund:
            pe, fwd_pe, ptb, rev_growth = fund
            # Lower P/E = higher score (capped at reasonable ranges)
            pe_score = max(0, min(100, 100 - ((fwd_pe or pe or 25) - 10) * 3))
            # Revenue growth bonus
            growth_score = min(100, (rev_growth or 0) * 200 + 50)
            scores["valuation"] = pe_score * 0.6 + growth_score * 0.4
        else:
            scores["valuation"] = 50

        # -- Sharpe (15%) --
        sharpe_data = get_sharpe(ticker, days=60)
        sharpe_val = sharpe_data.get("sharpe", 0)
        scores["sharpe"] = max(0, min(100, sharpe_val * 30 + 50))

        # -- Volatility (15%) --
        beta_data = get_beta(ticker)
        beta_val = beta_data.get("beta", 1.0)
        # Lower beta = higher score (less risky)
        scores["volatility"] = max(0, min(100, 100 - abs(beta_val - 1) * 40))

        # -- Mean reversion (15%) --
        zscore_data = get_zscore(ticker)
        z = zscore_data.get("zscore", 0)
        # Negative Z-score (oversold) = higher score for buy signals
        scores["mean_reversion"] = max(0, min(100, -z * 25 + 50))

        # -- Drawdown (10%) --
        dd_data = get_drawdown(ticker, days=60)
        dd = dd_data.get("current_drawdown_pct", 0)
        # Moderate drawdown (-5 to -15%) = good entry, extreme = risky
        if -15 < dd < -3:
            scores["drawdown"] = 75  # Good entry zone
        elif dd >= -3:
            scores["drawdown"] = 50  # Near highs
        else:
            scores["drawdown"] = 30  # Deep drawdown, risky

        # Weighted composite
        weights = {
            "technical": 0.25,
            "valuation": 0.20,
            "sharpe": 0.15,
            "volatility": 0.15,
            "mean_reversion": 0.15,
            "drawdown": 0.10,
        }

        composite = sum(scores[k] * weights[k] for k in weights)

        return {
            "ticker": ticker,
            "composite_score": round(composite, 1),
            "components": {k: round(v, 1) for k, v in scores.items()},
            "rating": "STRONG_BUY"
            if composite >= 75
            else "BUY"
            if composite >= 60
            else "NEUTRAL"
            if composite >= 40
            else "AVOID",
        }


def get_relative_strength(tickers: list[str], days: int = 20) -> list[dict]:
    """Rank tickers by recent performance (relative strength).

    Returns tickers sorted by best recent return.
    """
    with get_db() as db:
        results = []

        for ticker in tickers:
            df = _cursor_to_df(
                db.execute(
                    """
                SELECT close FROM price_history
                WHERE ticker = %s ORDER BY date DESC LIMIT %s
            """,
                    [ticker, days + 1],
                )
            )

            if df.empty or len(df) < 2:
                continue

            current = df["close"].iloc[0]
            past = df["close"].iloc[-1]
            pct_change = (current - past) / past * 100

            results.append(
                {
                    "ticker": ticker,
                    "return_pct": round(pct_change, 2),
                    "current_price": round(current, 2),
                }
            )

        results.sort(key=lambda x: x["return_pct"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

"""
Risk Processor — compute volatility, beta, drawdown, macro context.

Pure Python + pandas. No LLM calls. No hallucinations.
"""

import pandas as pd
from app.db.connection import get_db


def _cursor_to_df(cursor) -> pd.DataFrame:
    """Convert a PostgreSQL cursor result to a pandas DataFrame."""
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=cols)


def get_signals(ticker: str) -> str:
    """
    Get pre-formatted risk signals for the LLM.
    Computes from price_history and macro_indicators tables.
    """
    with get_db() as db:
        lines = [f"=== RISK ASSESSMENT: {ticker} ==="]

        # ── Price-based risk metrics ──
        df = _cursor_to_df(
            db.execute(
                """
            SELECT date, close
            FROM price_history
            WHERE ticker = %s
            ORDER BY date ASC
        """,
                [ticker],
            )
        )

        if len(df) >= 5:
            df["returns"] = df["close"].pct_change()

            # Volatility (annualized std of daily returns)
            vol_30d = df["returns"].tail(30).std() * (252**0.5)
            if vol_30d and not pd.isna(vol_30d):
                label = (
                    "HIGH" if vol_30d > 0.4 else "LOW" if vol_30d < 0.15 else "MODERATE"
                )
                lines.append(
                    f"30-Day Volatility: {vol_30d * 100:.1f}% annualized ({label})"
                )

            # Max drawdown (last 30 days)
            recent = df["close"].tail(30)
            peak = recent.cummax()
            drawdown = (recent - peak) / peak
            max_dd = drawdown.min()
            if max_dd and not pd.isna(max_dd):
                label = (
                    "SEVERE"
                    if max_dd < -0.15
                    else "MINOR"
                    if max_dd > -0.05
                    else "MODERATE"
                )
                lines.append(f"Max Drawdown (30d): {max_dd * 100:.1f}% ({label})")

            # Price vs recent high
            current = df["close"].iloc[-1]
            high_30d = df["close"].tail(30).max()
            pct_from_high = (current - high_30d) / high_30d * 100
            lines.append(f"Price vs 30d High: {pct_from_high:+.1f}%")
        else:
            lines.append("Insufficient price data for risk metrics.")

        # ── Macro context ──
        lines.append("\nMacro Environment:")

        # Yield curve (10Y - 2Y spread)
        t10 = db.execute("""
            SELECT value FROM macro_indicators
            WHERE indicator = 'TREASURY_10Y'
            ORDER BY date DESC LIMIT 1
        """).fetchone()

        t2 = db.execute("""
            SELECT value FROM macro_indicators
            WHERE indicator = 'TREASURY_2Y'
            ORDER BY date DESC LIMIT 1
        """).fetchone()

        if t10 and t2:
            spread = t10[0] - t2[0]
            label = (
                "INVERTED (recession signal)"
                if spread < 0
                else "NORMAL"
                if spread > 0.5
                else "FLAT"
            )
            lines.append(f"  Yield Curve (10Y-2Y): {spread:+.2f}% ({label})")
            lines.append(f"  10Y Treasury: {t10[0]:.2f}%")
            lines.append(f"  2Y Treasury: {t2[0]:.2f}%")

        # Fed funds rate
        fed = db.execute("""
            SELECT value FROM macro_indicators
            WHERE indicator = 'FED_FUNDS'
            ORDER BY date DESC LIMIT 1
        """).fetchone()

        if fed:
            label = (
                "RESTRICTIVE"
                if fed[0] > 4
                else "ACCOMMODATIVE"
                if fed[0] < 2
                else "NEUTRAL"
            )
            lines.append(f"  Fed Funds Rate: {fed[0]:.2f}% ({label})")

        # CPI (inflation)
        cpi_recent = db.execute("""
            SELECT value FROM macro_indicators
            WHERE indicator = 'CPI'
            ORDER BY date DESC LIMIT 2
        """).fetchall()

        if len(cpi_recent) >= 2 and cpi_recent[1][0] and cpi_recent[1][0] != 0:
            cpi_change = (
                (cpi_recent[0][0] - cpi_recent[1][0]) / cpi_recent[1][0] * 100 * 12
            )
            label = (
                "HIGH INFLATION"
                if cpi_change > 4
                else "DEFLATION RISK"
                if cpi_change < 1
                else "MODERATE"
            )
            lines.append(f"  CPI (annualized MoM): {cpi_change:+.1f}% ({label})")

        return "\n".join(lines)

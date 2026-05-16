"""
Data Sanity Guardrails — Post-collection spot checks.

Run after data collection to catch value conversion bugs,
missing data, and obviously wrong values before they reach the LLM.

Usage:
    from app.processors.data_sanity import run_sanity_checks
    failures = run_sanity_checks()
    if failures:
        for f in failures:
            print(f"❌ {f}")
    else:
        print("✅ All sanity checks passed")
"""

import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def run_sanity_checks() -> list[str]:
    """Run all sanity checks. Returns list of failure messages. Empty = all good."""
    failures = []
    with get_db() as db:
        # ── 13F Holdings ──
        try:
            # No single position > $1T
            row = db.execute("SELECT MAX(value_usd) FROM sec_13f_holdings").fetchone()
            if row and row[0] and row[0] > 1e12:
                failures.append(
                    f"13F: Max position value ${row[0] / 1e9:.1f}B exceeds $1T ceiling"
                )

            # No position with negative value
            row = db.execute(
                "SELECT COUNT(*) FROM sec_13f_holdings WHERE value_usd < 0"
            ).fetchone()
            if row and row[0] > 0:
                failures.append(f"13F: {row[0]} holdings with negative value_usd")

            # No position with 0 shares but positive value
            row = db.execute(
                "SELECT COUNT(*) FROM sec_13f_holdings WHERE shares <= 0 AND value_usd > 0"
            ).fetchone()
            if row and row[0] > 0:
                failures.append(
                    f"13F: {row[0]} holdings with 0 shares but positive value"
                )

            # Berkshire AAPL sanity (if exists)
            row = db.execute("""
                SELECT value_usd FROM sec_13f_holdings
                WHERE filer_name LIKE '%%Berkshire%%' AND ticker = 'AAPL'
                ORDER BY value_usd DESC LIMIT 1
            """).fetchone()
            if row and row[0]:
                if row[0] < 30e9:
                    failures.append(
                        f"13F: Berkshire AAPL = ${row[0] / 1e9:.1f}B (expected > $30B)"
                    )
                elif row[0] > 300e9:
                    failures.append(
                        f"13F: Berkshire AAPL = ${row[0] / 1e9:.1f}B (expected < $300B)"
                    )
        except Exception as e:
            logger.warning(f"13F sanity check error: {e}")

        # ── Fundamentals ──
        try:
            # AAPL market cap > $1T
            row = db.execute("""
                SELECT market_cap FROM fundamentals WHERE ticker = 'AAPL'
                ORDER BY snapshot_date DESC LIMIT 1
            """).fetchone()
            if row and row[0] and row[0] < 1e12:
                failures.append(
                    f"Fundamentals: AAPL market cap ${row[0] / 1e9:.1f}B < $1T"
                )

            # No negative market caps
            row = db.execute(
                "SELECT COUNT(*) FROM fundamentals WHERE market_cap < 0"
            ).fetchone()
            if row and row[0] > 0:
                failures.append(
                    f"Fundamentals: {row[0]} tickers with negative market cap"
                )

            # P/E ratio sanity (should be 0-500 or NULL)
            row = db.execute("""
                SELECT ticker, pe_ratio FROM fundamentals
                WHERE pe_ratio IS NOT NULL AND (pe_ratio < -100 OR pe_ratio > 1000)
                LIMIT 5
            """).fetchall()
            if row:
                tickers = [f"{r[0]}={r[1]:.0f}" for r in row]
                failures.append(
                    f"Fundamentals: Extreme P/E ratios: {', '.join(tickers)}"
                )
        except Exception as e:
            logger.warning(f"Fundamentals sanity check error: {e}")

        # ── Price Data ──
        try:
            # No $0 or negative prices
            row = db.execute(
                "SELECT COUNT(*) FROM price_history WHERE close <= 0"
            ).fetchone()
            if row and row[0] > 0:
                failures.append(f"Prices: {row[0]} rows with close <= $0")

            # No absurd single-day moves (>80%)
            row = db.execute("""
                SELECT ticker, date, open, close,
                       ABS((close - open) / NULLIF(open, 0)) * 100 as pct_move
                FROM price_history
                WHERE ABS((close - open) / NULLIF(open, 0)) > 0.80
                LIMIT 5
            """).fetchall()
            if row:
                moves = [f"{r[0]} {r[1]} {r[4]:.0f}%" for r in row]
                failures.append(
                    f"Prices: Absurd single-day moves (>80%): {', '.join(moves)}"
                )
        except Exception as e:
            logger.warning(f"Price sanity check error: {e}")

        # ── Congress Trades ──
        try:
            # Check both chambers exist
            chambers = db.execute(
                "SELECT DISTINCT chamber FROM congress_trades WHERE chamber IS NOT NULL"
            ).fetchall()
            chamber_set = {r[0] for r in chambers}
            if (
                chamber_set
                and "Senate" not in chamber_set
                and "senate" not in {c.lower() for c in chamber_set}
            ):
                failures.append(
                    f"Congress: Only chambers found: {chamber_set} — missing Senate data"
                )
            if (
                chamber_set
                and "House" not in chamber_set
                and "house" not in {c.lower() for c in chamber_set}
            ):
                failures.append(
                    f"Congress: Only chambers found: {chamber_set} — missing House data"
                )
        except Exception as e:
            logger.warning(f"Congress sanity check error: {e}")

        # ── Technicals ──
        try:
            # RSI should be 0-100
            row = db.execute("""
                SELECT COUNT(*) FROM technicals
                WHERE rsi_14 IS NOT NULL AND (rsi_14 < 0 OR rsi_14 > 100)
            """).fetchone()
            if row and row[0] > 0:
                failures.append(
                    f"Technicals: {row[0]} rows with RSI outside 0-100 range"
                )
        except Exception as e:
            logger.warning(f"Technicals sanity check error: {e}")

        return failures


def print_sanity_report():
    """Run checks and print formatted report."""
    print("\n" + "=" * 60)
    print("[SANITY CHECK] Post-Collection Data Validation")
    print("=" * 60)

    failures = run_sanity_checks()

    if not failures:
        print("  [OK] All sanity checks passed")
    else:
        for f in failures:
            print(f"  [FAIL] {f}")
        print(
            f"\n  {len(failures)} issue(s) detected — review before running LLM analysis"
        )

    print("=" * 60 + "\n")
    return failures

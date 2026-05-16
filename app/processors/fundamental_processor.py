"""
Fundamental Processor — format company fundamentals into labeled signals.

Pure Python formatting. No LLM calls. No hallucinations.
"""

from app.db.connection import get_db
from app.utils.text_utils import fmt_usd


def get_signals(ticker: str) -> str:
    """
    Get pre-formatted fundamental signals for the LLM.
    Reads from fundamentals, financial_history, balance_sheet tables.
    """
    with get_db() as db:
        lines = [f"=== FUNDAMENTAL ANALYSIS: {ticker} ==="]

        # ── Valuation metrics ──
        fund = db.execute(
            """
            SELECT * FROM fundamentals
            WHERE ticker = %s
            ORDER BY snapshot_date DESC LIMIT 1
        """,
            [ticker],
        ).fetchone()

        if fund:
            cols = [
                d[0]
                for d in db.execute("SELECT * FROM fundamentals LIMIT 0").description
            ]
            f = dict(zip(cols, fund))

            lines.append(f"\nSnapshot: {f.get('snapshot_date')}")

            # PE
            pe = f.get("pe_ratio")
            if pe:
                label = "HIGH" if pe > 35 else "LOW" if pe < 15 else "MODERATE"
                lines.append(f"P/E Ratio: {pe:.1f} ({label})")

            fpe = f.get("forward_pe")
            if fpe:
                lines.append(f"Forward P/E: {fpe:.1f}")

            peg = f.get("peg_ratio")
            if peg:
                label = (
                    "UNDERVALUED" if peg < 1 else "OVERVALUED" if peg > 2 else "FAIR"
                )
                lines.append(f"PEG Ratio: {peg:.2f} ({label})")

            pb = f.get("price_to_book")
            if pb:
                lines.append(f"P/B: {pb:.2f}")

            ev = f.get("ev_to_ebitda")
            if ev:
                lines.append(f"EV/EBITDA: {ev:.1f}")

            # Profitability
            margin = f.get("profit_margin")
            if margin:
                pct = margin * 100
                label = "STRONG" if pct > 20 else "WEAK" if pct < 5 else "MODERATE"
                lines.append(f"Profit Margin: {pct:.1f}% ({label})")

            roe = f.get("roe")
            if roe:
                pct = roe * 100
                label = "EXCELLENT" if pct > 20 else "POOR" if pct < 10 else "AVERAGE"
                lines.append(f"ROE: {pct:.1f}% ({label})")

            # Growth
            rev_growth = f.get("revenue_growth")
            if rev_growth:
                pct = rev_growth * 100
                label = (
                    "HIGH GROWTH"
                    if pct > 20
                    else "DECLINING"
                    if pct < 0
                    else "MODERATE"
                )
                lines.append(f"Revenue Growth: {pct:+.1f}% YoY ({label})")

            # Health
            dte = f.get("debt_to_equity")
            if dte:
                label = "RISKY" if dte > 2 else "HEALTHY" if dte < 0.5 else "MODERATE"
                lines.append(f"Debt/Equity: {dte:.2f} ({label})")

            cr = f.get("current_ratio")
            if cr:
                label = "STRONG" if cr > 2 else "WEAK" if cr < 1 else "ADEQUATE"
                lines.append(f"Current Ratio: {cr:.2f} ({label})")

            beta = f.get("beta")
            if beta:
                label = (
                    "HIGH VOLATILITY"
                    if beta > 1.5
                    else "LOW VOL"
                    if beta < 0.8
                    else "MARKET-LIKE"
                )
                lines.append(f"Beta: {beta:.2f} ({label})")

            # Market cap
            mc = f.get("market_cap")
            if mc:
                lines.append(f"Market Cap: {fmt_usd(mc)}")

            # 52-week range
            h52 = f.get("week_52_high")
            l52 = f.get("week_52_low")
            if h52 and l52:
                lines.append(f"52-Week Range: ${l52:.2f} - ${h52:.2f}")

            short = f.get("short_float_pct")
            if short:
                label = "HIGH SHORT INTEREST" if short > 10 else "LOW"
                lines.append(f"Short Float: {short:.1f}% ({label})")
        else:
            lines.append("No fundamental data available.")

        # ── Recent financials ──
        fins = db.execute(
            """
            SELECT period_end, revenue, net_income, eps, free_cash_flow
            FROM financial_history
            WHERE ticker = %s
            ORDER BY period_end DESC LIMIT 4
        """,
            [ticker],
        ).fetchall()

        lines.append("\nRecent Quarterly Financials:")
        if fins:
            for row in fins:
                rev = fmt_usd(row[1]) if row[1] else "N/A"
                ni = fmt_usd(row[2]) if row[2] else "N/A"
                lines.append(
                    f"  {row[0]}: Rev={rev}, Net Income={ni}, EPS=${row[3]:.2f}"
                    if row[3]
                    else f"  {row[0]}: Rev={rev}, Net Income={ni}"
                )
        else:
            lines.append("  No quarterly financials available.")

        # ── Balance sheet ──
        bs = db.execute(
            """
            SELECT period_end, total_assets, total_debt, cash, total_equity
            FROM balance_sheet
            WHERE ticker = %s
            ORDER BY period_end DESC LIMIT 1
        """,
            [ticker],
        ).fetchone()

        lines.append("\nBalance Sheet:")
        if bs:
            lines.append(f"  Period End: {bs[0]}")
            if bs[1]:
                lines.append(f"  Total Assets: {fmt_usd(bs[1])}")
            if bs[2]:
                lines.append(f"  Total Debt: {fmt_usd(bs[2])}")
            if bs[3]:
                lines.append(f"  Cash: {fmt_usd(bs[3])}")
            if bs[4]:
                lines.append(f"  Equity: {fmt_usd(bs[4])}")
        else:
            lines.append("  No balance sheet data available.")

        return "\n".join(lines)

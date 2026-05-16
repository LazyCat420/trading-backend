"""
Institutional Processor — format 13F holdings + congress trades.

Pure Python formatting. No LLM calls. No hallucinations.
"""

from app.db.connection import get_db


def get_signals(ticker: str) -> str:
    """
    Get pre-formatted institutional signals for the LLM.
    Reads from sec_13f_holdings and congress_trades tables.
    """
    with get_db() as db:
        lines = [f"=== INSTITUTIONAL & CONGRESS SIGNALS: {ticker} ==="]

        # ── 13F Holdings (which big funds hold this ticker) ──
        holdings = db.execute(
            """
            SELECT f.filer_name, h.shares, h.value_usd, h.filing_quarter,
                   h.is_new_position, h.is_exit
            FROM sec_13f_holdings h
            LEFT JOIN sec_13f_filers f ON h.cik = f.cik
            WHERE UPPER(h.ticker) LIKE %s
            ORDER BY value_usd DESC
            LIMIT 10
        """,
            [f"%{ticker}%"],
        ).fetchall()

        if holdings:
            lines.append(f"\n13F Institutional Holdings ({len(holdings)} funds):")
            for h in holdings:
                value_str = (
                    f"${h[2] / 1e6:.1f}M"
                    if h[2] and h[2] > 1e6
                    else f"${h[2]:,.0f}"
                    if h[2]
                    else "N/A"
                )
                shares_str = f"{h[1]:,}" if h[1] else "N/A"
                flags = []
                if h[4]:
                    flags.append("NEW POSITION")
                if h[5]:
                    flags.append("EXITED")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                lines.append(
                    f"  {h[0]}: {shares_str} shares (${value_str}) Q:{h[3]}{flag_str}"
                )
        else:
            lines.append("\nNo 13F holdings found for this ticker.")
            lines.append(
                "  (Note: 13F uses issuer names, not tickers — partial match used)"
            )

        # ── Congress trades ──
        trades = db.execute(
            """
            SELECT politician, party, transaction_type, amount_range,
                   trade_date, disclosure_date, days_to_disclose
            FROM congress_trades
            WHERE UPPER(ticker) = %s
            ORDER BY trade_date DESC
            LIMIT 10
        """,
            [ticker.upper()],
        ).fetchall()

        if trades:
            lines.append(f"\nCongressional Trades ({len(trades)} recent):")
            for t in trades:
                party_tag = f"({t[1]})" if t[1] else ""
                delay = f" (disclosed {t[6]}d later)" if t[6] else ""
                lines.append(f"  {t[0]} {party_tag}: {t[2]} {t[3]} on {t[4]}{delay}")

            # Summary: net direction
            buys = sum(
                1
                for t in trades
                if "purchase" in (t[2] or "").lower() or "buy" in (t[2] or "").lower()
            )
            sells = sum(
                1
                for t in trades
                if "sale" in (t[2] or "").lower() or "sell" in (t[2] or "").lower()
            )
            if buys > sells:
                lines.append(
                    f"  → NET CONGRESS DIRECTION: BUYING ({buys} buys vs {sells} sells)"
                )
            elif sells > buys:
                lines.append(
                    f"  → NET CONGRESS DIRECTION: SELLING ({sells} sells vs {buys} buys)"
                )
            else:
                lines.append(
                    f"  → NET CONGRESS DIRECTION: NEUTRAL ({buys} buys, {sells} sells)"
                )
        else:
            lines.append("\nNo recent congressional trades for this ticker.")

        return "\n".join(lines)

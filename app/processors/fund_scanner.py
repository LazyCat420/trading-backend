"""
Fund Scanner — Discovery-mode analysis of institutional 13F holdings.

Scans ALL 13F holdings in the DB and produces:
  1. Fund portfolio snapshots — top holdings per fund
  2. Cross-fund consensus — tickers held by multiple top funds
  3. Quarterly changes — new positions, exits, size changes
  4. Watchlist comparison — overlap between fund holdings and our tickers
  5. Discovery — tickers funds hold that we're NOT watching

Data source: sec_13f_holdings table (populated by sec_collector.py)
"""

from app.db.connection import get_db


def get_fund_portfolios(top_holdings: int = 20) -> list[dict]:
    """Get the top holdings for each fund in the latest filing quarter."""
    with get_db() as db:
        # Get the latest quarter per fund
        funds = db.execute("""
            SELECT DISTINCT f.filer_name, h.cik, h.filing_quarter
            FROM sec_13f_holdings h
            JOIN sec_13f_filers f ON h.cik = f.cik
            WHERE h.filing_quarter = (
                SELECT MAX(filing_quarter) FROM sec_13f_holdings AS sub
                WHERE sub.cik = h.cik
            )
            ORDER BY f.filer_name
        """).fetchall()

        portfolios = []
        for filer_name, cik, quarter in funds:
            holdings = db.execute(
                """
                SELECT ticker, shares, value_usd, pct_change,
                       is_new_position, is_exit
                FROM sec_13f_holdings
                WHERE cik = %s AND filing_quarter = %s
                ORDER BY value_usd DESC
                LIMIT %s
            """,
                [cik, quarter, top_holdings],
            ).fetchall()

            total_value = (
                db.execute(
                    """
                SELECT SUM(value_usd) FROM sec_13f_holdings
                WHERE cik = %s AND filing_quarter = %s
            """,
                    [cik, quarter],
                ).fetchone()[0]
                or 0
            )

            holding_count = (
                db.execute(
                    """
                SELECT COUNT(*) FROM sec_13f_holdings
                WHERE cik = %s AND filing_quarter = %s
            """,
                    [cik, quarter],
                ).fetchone()[0]
                or 0
            )

            top = []
            for h in holdings:
                pct_of_portfolio = (h[2] / total_value * 100) if total_value > 0 else 0
                top.append(
                    {
                        "ticker": h[0],
                        "shares": h[1],
                        "value_usd": h[2],
                        "pct_change": h[3],
                        "is_new": bool(h[4]),
                        "is_exit": bool(h[5]),
                        "pct_of_portfolio": round(pct_of_portfolio, 2),
                    }
                )

            portfolios.append(
                {
                    "fund": filer_name,
                    "quarter": quarter,
                    "total_value": total_value,
                    "holding_count": holding_count,
                    "top_holdings": top,
                }
            )

        return portfolios


def find_crossfund_consensus(min_funds: int = 3) -> list[dict]:
    """Find tickers held by multiple top funds — consensus = conviction.

    If Berkshire, Citadel, AND Renaissance all hold the same stock,
    that's a strong institutional conviction signal.
    """
    with get_db() as db:
        # Use latest quarter per fund
        rows = db.execute(
            """
            WITH latest_quarters AS (
                SELECT cik, MAX(filing_quarter) as q
                FROM sec_13f_holdings
                GROUP BY cik
            )
            SELECT h.ticker,
                   COUNT(DISTINCT h.cik) as fund_count,
                   STRING_AGG(DISTINCT f.filer_name, ', ') as funds,
                   SUM(h.value_usd) as total_value,
                   SUM(h.shares) as total_shares
            FROM sec_13f_holdings h
            JOIN latest_quarters lq ON h.cik = lq.cik AND h.filing_quarter = lq.q
            JOIN sec_13f_filers f ON h.cik = f.cik
            WHERE h.ticker != 'nan' AND h.ticker != '' AND LENGTH(h.ticker) <= 5
            GROUP BY h.ticker
            HAVING COUNT(DISTINCT h.cik) >= %s
            ORDER BY COUNT(DISTINCT h.cik) DESC, total_value DESC
        """,
            [min_funds],
        ).fetchall()

        consensus = []
        for r in rows:
            consensus.append(
                {
                    "ticker": r[0],
                    "fund_count": r[1],
                    "funds": r[2].split(",") if r[2] else [],
                    "total_value": r[3],
                    "total_shares": r[4],
                }
            )

        return consensus


def detect_quarterly_changes() -> dict:
    """Detect new positions, exits, and significant size changes across funds.

    Compares the latest filing quarter against the previous one.
    """
    with get_db() as db:
        # Get the two most recent quarters across all funds
        quarters = db.execute("""
            SELECT DISTINCT filing_quarter FROM sec_13f_holdings
            ORDER BY filing_quarter DESC LIMIT 2
        """).fetchall()

        if len(quarters) < 2:
            return {
                "new_positions": [],
                "exits": [],
                "size_changes": [],
                "note": "Need at least 2 quarters of data",
            }

        latest_q = quarters[0][0]
        prev_q = quarters[1][0]

        # New positions — in latest but not previous
        new_positions = db.execute(
            """
            SELECT f.filer_name, h.ticker, h.shares, h.value_usd
            FROM sec_13f_holdings h
            JOIN sec_13f_filers f ON h.cik = f.cik
            WHERE h.filing_quarter = %s
              AND h.ticker NOT IN (
                  SELECT ticker FROM sec_13f_holdings
                  WHERE cik = h.cik AND filing_quarter = %s
              )
              AND h.ticker != 'nan' AND h.ticker != ''
            ORDER BY h.value_usd DESC
            LIMIT 50
        """,
            [latest_q, prev_q],
        ).fetchall()

        # Exits — in previous but not latest
        exits = db.execute(
            """
            SELECT f.filer_name, h.ticker, h.shares, h.value_usd
            FROM sec_13f_holdings h
            JOIN sec_13f_filers f ON h.cik = f.cik
            WHERE h.filing_quarter = %s
              AND h.ticker NOT IN (
                  SELECT ticker FROM sec_13f_holdings
                  WHERE cik = h.cik AND filing_quarter = %s
              )
              AND h.ticker != 'nan' AND h.ticker != ''
            ORDER BY h.value_usd DESC
            LIMIT 50
        """,
            [prev_q, latest_q],
        ).fetchall()

        return {
            "latest_quarter": latest_q,
            "previous_quarter": prev_q,
            "new_positions": [
                {"fund": r[0], "ticker": r[1], "shares": r[2], "value": r[3]}
                for r in new_positions
            ],
            "exits": [
                {"fund": r[0], "ticker": r[1], "shares": r[2], "value": r[3]}
                for r in exits
            ],
            "new_position_count": len(new_positions),
            "exit_count": len(exits),
        }


def compare_with_watchlist(watchlist_tickers: list[str]) -> dict:
    """Compare fund holdings against our watchlist.

    Returns:
      - overlap: tickers held by funds AND in our watchlist
      - discovery: tickers held by funds that we're NOT watching
      - not_held: watchlist tickers with no institutional presence
    """
    with get_db() as db:
        # All tickers currently held by funds (latest quarter)
        fund_tickers = db.execute("""
            WITH latest_quarters AS (
                SELECT cik, MAX(filing_quarter) as q
                FROM sec_13f_holdings GROUP BY cik
            )
            SELECT DISTINCT h.ticker
            FROM sec_13f_holdings h
            JOIN latest_quarters lq ON h.cik = lq.cik AND h.filing_quarter = lq.q
            WHERE h.ticker != 'nan' AND h.ticker != '' AND LENGTH(h.ticker) <= 5
        """).fetchall()
        fund_set = {r[0] for r in fund_tickers}

        watchlist_set = {t.upper() for t in watchlist_tickers}

        overlap = fund_set & watchlist_set
        discovery = fund_set - watchlist_set
        not_held = watchlist_set - fund_set

        # Details for overlap
        overlap_details = []
        for ticker in sorted(overlap):
            holders = db.execute(
                """
                WITH latest_quarters AS (
                    SELECT cik, MAX(filing_quarter) as q
                    FROM sec_13f_holdings GROUP BY cik
                )
                SELECT f.filer_name, h.shares, h.value_usd
                FROM sec_13f_holdings h
                JOIN latest_quarters lq ON h.cik = lq.cik AND h.filing_quarter = lq.q
                JOIN sec_13f_filers f ON h.cik = f.cik
                WHERE h.ticker = %s
                ORDER BY h.value_usd DESC
            """,
                [ticker],
            ).fetchall()
            overlap_details.append(
                {
                    "ticker": ticker,
                    "fund_count": len(holders),
                    "total_value": sum(h[2] or 0 for h in holders),
                    "holders": [
                        {"fund": h[0], "shares": h[1], "value": h[2]}
                        for h in holders[:5]
                    ],
                }
            )

        # Details for discovery — top tickers by value held
        discovery_details = []
        for ticker in sorted(discovery):
            holders = db.execute(
                """
                WITH latest_quarters AS (
                    SELECT cik, MAX(filing_quarter) as q
                    FROM sec_13f_holdings GROUP BY cik
                )
                SELECT f.filer_name, h.shares, h.value_usd
                FROM sec_13f_holdings h
                JOIN latest_quarters lq ON h.cik = lq.cik AND h.filing_quarter = lq.q
                JOIN sec_13f_filers f ON h.cik = f.cik
                WHERE h.ticker = %s
                ORDER BY h.value_usd DESC
            """,
                [ticker],
            ).fetchall()
            total_val = sum(h[2] or 0 for h in holders)
            if total_val > 0:
                discovery_details.append(
                    {
                        "ticker": ticker,
                        "fund_count": len(holders),
                        "total_value": total_val,
                        "top_holder": holders[0][0] if holders else "",
                    }
                )

        discovery_details.sort(key=lambda x: x["total_value"], reverse=True)

        return {
            "overlap": overlap_details,
            "overlap_count": len(overlap),
            "discovery": discovery_details[:30],
            "discovery_count": len(discovery),
            "not_held": sorted(not_held),
            "not_held_count": len(not_held),
            "fund_total_tickers": len(fund_set),
        }


def generate_report(watchlist_tickers: list[str] | None = None) -> str:
    """Generate a human-readable institutional holdings report."""
    lines = []
    lines.append("=" * 70)
    lines.append("INSTITUTIONAL FUND SCANNER REPORT")
    lines.append("=" * 70)

    # Fund portfolios
    portfolios = get_fund_portfolios(top_holdings=10)
    lines.append(f"\n📊 Fund Portfolios ({len(portfolios)} funds tracked):")
    for p in portfolios:
        total_fmt = f"${p['total_value']:,.0f}" if p["total_value"] else "$0"
        lines.append(
            f"\n   {p['fund']} ({p['quarter']}) — "
            f"{p['holding_count']} holdings, {total_fmt} total"
        )
        for h in p["top_holdings"][:5]:
            val_fmt = f"${h['value_usd']:,.0f}" if h["value_usd"] else "$0"
            new_flag = " 🆕" if h["is_new"] else ""
            lines.append(
                f"      {h['ticker']}: {h['shares']:,} shares, "
                f"{val_fmt} ({h['pct_of_portfolio']:.1f}%){new_flag}"
            )

    # Cross-fund consensus
    consensus = find_crossfund_consensus(min_funds=2)
    if consensus:
        lines.append(
            f"\n🎯 Cross-Fund Consensus ({len(consensus)} tickers held by 2+ funds):"
        )
        for c in consensus[:15]:
            val_fmt = f"${c['total_value']:,.0f}" if c["total_value"] else "$0"
            lines.append(
                f"   {c['ticker']}: {c['fund_count']} funds ({val_fmt}) — "
                f"{', '.join(c['funds'][:3])}"
            )

    # Quarterly changes
    changes = detect_quarterly_changes()
    if changes.get("new_positions"):
        lines.append(f"\n🆕 New Positions ({changes['new_position_count']}):")
        for np in changes["new_positions"][:10]:
            val_fmt = f"${np['value']:,.0f}" if np["value"] else "$0"
            lines.append(f"   {np['fund']} → {np['ticker']} ({val_fmt})")
    if changes.get("exits"):
        lines.append(f"\n🚪 Exits ({changes['exit_count']}):")
        for ex in changes["exits"][:10]:
            val_fmt = f"${ex['value']:,.0f}" if ex["value"] else "$0"
            lines.append(f"   {ex['fund']} ← {ex['ticker']} ({val_fmt})")

    # Watchlist comparison
    if watchlist_tickers:
        comp = compare_with_watchlist(watchlist_tickers)
        lines.append("\n🔍 Watchlist Comparison:")
        lines.append(f"   Funds hold {comp['fund_total_tickers']} unique tickers")
        lines.append(f"   Overlap with watchlist: {comp['overlap_count']}")
        lines.append(
            f"   Discovery (funds hold, not on watchlist): {comp['discovery_count']}"
        )
        lines.append(f"   Not held by any fund: {comp['not_held_count']}")

        if comp["overlap"]:
            lines.append("\n   📌 Overlap:")
            for o in comp["overlap"]:
                val_fmt = f"${o['total_value']:,.0f}" if o["total_value"] else "$0"
                lines.append(f"      {o['ticker']}: {o['fund_count']} funds, {val_fmt}")
                for h in o["holders"][:3]:
                    lines.append(f"         {h['fund']}: {h['shares']:,} shares")

        if comp["discovery"]:
            lines.append("\n   🆕 Discovery (funds hold, you're not watching):")
            for d in comp["discovery"][:15]:
                val_fmt = f"${d['total_value']:,.0f}" if d["total_value"] else "$0"
                lines.append(
                    f"      {d['ticker']}: {d['fund_count']} funds, "
                    f"{val_fmt} (top: {d['top_holder']})"
                )

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)

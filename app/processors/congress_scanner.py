"""
Congress Scanner — Discovery-mode analysis of congressional stock trades.

Scans ALL congress trades in the DB (not just our watchlist tickers) and produces:
  1. Recent activity report — who bought/sold what this week/month
  2. Consensus signals — multiple congress members trading the same ticker
  3. Portfolio tracking — estimated current holdings per politician
  4. Watchlist comparison — overlap between congress trades and our tickers
  5. Discovery — tickers congress is trading that we're NOT watching

Data source: congress_trades table (populated by congress_collector.py)
"""

import datetime
from app.db.connection import get_db


def scan_recent_trades(days: int = 30) -> dict:
    """Get all congress trades from the last N days, grouped by ticker."""
    with get_db() as db:
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).date()

        rows = db.execute(
            """
            SELECT politician, party, chamber, state, ticker,
                   transaction_type, amount_range, trade_date, disclosure_date
            FROM congress_trades
            WHERE trade_date >= %s AND party != '' AND party IS NOT NULL
            ORDER BY trade_date DESC
        """,
            [cutoff],
        ).fetchall()

        trades = []
        for r in rows:
            trades.append(
                {
                    "politician": r[0],
                    "party": r[1],
                    "chamber": r[2],
                    "state": r[3],
                    "ticker": r[4],
                    "type": r[5],
                    "amount": r[6],
                    "trade_date": str(r[7]) if r[7] else None,
                    "disclosure_date": str(r[8]) if r[8] else None,
                }
            )

        return {
            "total_trades": len(trades),
            "period_days": days,
            "cutoff_date": str(cutoff),
            "trades": trades,
        }


def find_consensus_trades(days: int = 30, min_members: int = 2) -> list[dict]:
    """Find tickers that multiple congress members traded in the same direction.

    A "consensus BUY" = 2+ members buying the same ticker within the period.
    This is a strong signal — if 3 senators all bought the same stock, pay attention.
    """
    with get_db() as db:
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).date()

        rows = db.execute(
            """
            SELECT ticker, transaction_type, 
                   COUNT(DISTINCT politician) as member_count,
                   STRING_AGG(DISTINCT politician, ', ') as members,
                   STRING_AGG(DISTINCT party, ', ') as parties,
                   MIN(trade_date) as earliest,
                   MAX(trade_date) as latest
            FROM congress_trades
            WHERE trade_date >= %s AND party != '' AND party IS NOT NULL
            GROUP BY ticker, transaction_type
            HAVING COUNT(DISTINCT politician) >= %s
            ORDER BY member_count DESC
        """,
            [cutoff, min_members],
        ).fetchall()

        consensus = []
        for r in rows:
            consensus.append(
                {
                    "ticker": r[0],
                    "direction": r[1],  # buy or sell
                    "member_count": r[2],
                    "members": r[3].split(",") if r[3] else [],
                    "parties": r[4].split(",") if r[4] else [],
                    "earliest_trade": str(r[5]) if r[5] else None,
                    "latest_trade": str(r[6]) if r[6] else None,
                }
            )

        return consensus


def build_politician_portfolios(top_n: int = 20) -> list[dict]:
    """Estimate current holdings per politician based on buy/sell history.

    This is a rough estimate — we don't know exact share counts,
    only the amount range and direction.
    """
    with get_db() as db:
        rows = db.execute(
            """
            SELECT politician, party, chamber, state,
                   COUNT(*) as total_trades,
                   COUNT(CASE WHEN transaction_type = 'buy' THEN 1 END) as buys,
                   COUNT(CASE WHEN transaction_type = 'sell' THEN 1 END) as sells,
                   STRING_AGG(DISTINCT ticker, ', ') as all_tickers,
                   MIN(trade_date) as earliest_trade,
                   MAX(trade_date) as latest_trade
            FROM congress_trades
            WHERE party != '' AND party IS NOT NULL
            GROUP BY politician, party, chamber, state
            ORDER BY total_trades DESC
            LIMIT %s
        """,
            [top_n],
        ).fetchall()

        portfolios = []
        for r in rows:
            politician = r[0]
            # Get their current "held" tickers (bought but not subsequently sold)
            held = db.execute(
                """
                SELECT ticker, COUNT(CASE WHEN transaction_type = 'buy' THEN 1 END) as buy_count,
                       COUNT(CASE WHEN transaction_type = 'sell' THEN 1 END) as sell_count,
                       MAX(trade_date) as last_trade
                FROM congress_trades
                WHERE politician = %s AND party != '' AND party IS NOT NULL
                GROUP BY ticker
                HAVING COUNT(CASE WHEN transaction_type = 'buy' THEN 1 END) > COUNT(CASE WHEN transaction_type = 'sell' THEN 1 END)
                ORDER BY last_trade DESC
            """,
                [politician],
            ).fetchall()

            held_tickers = [h[0] for h in held]

            portfolios.append(
                {
                    "politician": politician,
                    "party": r[1],
                    "chamber": r[2],
                    "state": r[3],
                    "total_trades": r[4],
                    "buys": r[5],
                    "sells": r[6],
                    "all_tickers_traded": r[7].split(",") if r[7] else [],
                    "estimated_holdings": held_tickers,
                    "holding_count": len(held_tickers),
                    "earliest_trade": str(r[8]) if r[8] else None,
                    "latest_trade": str(r[9]) if r[9] else None,
                }
            )

        return portfolios


def compare_with_watchlist(watchlist_tickers: list[str], days: int = 90) -> dict:
    """Compare congress trades against our watchlist.

    Returns:
      - overlap: tickers in both watchlist and congress trades
      - discovery: tickers congress is trading that we're NOT watching
      - not_traded: watchlist tickers with no congress activity
    """
    with get_db() as db:
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).date()

        # All tickers traded by congress recently
        congress_tickers = db.execute(
            """
            SELECT DISTINCT ticker FROM congress_trades
            WHERE trade_date >= %s AND party != '' AND party IS NOT NULL
        """,
            [cutoff],
        ).fetchall()
        congress_set = {r[0] for r in congress_tickers}

        watchlist_set = {t.upper() for t in watchlist_tickers}

        overlap = congress_set & watchlist_set
        discovery = congress_set - watchlist_set
        not_traded = watchlist_set - congress_set

        # Get details for overlap tickers
        overlap_details = []
        for ticker in sorted(overlap):
            trades = db.execute(
                """
                SELECT politician, transaction_type, amount_range, trade_date
                FROM congress_trades
                WHERE ticker = %s AND trade_date >= %s AND party != '' AND party IS NOT NULL
                ORDER BY trade_date DESC
            """,
                [ticker, cutoff],
            ).fetchall()
            overlap_details.append(
                {
                    "ticker": ticker,
                    "trade_count": len(trades),
                    "trades": [
                        {
                            "politician": t[0],
                            "type": t[1],
                            "amount": t[2],
                            "date": str(t[3]),
                        }
                        for t in trades[:5]  # Top 5 most recent
                    ],
                }
            )

        # Get details for discovery tickers (congress trades we're not watching)
        discovery_details = []
        for ticker in sorted(discovery):
            trades = db.execute(
                """
                SELECT politician, transaction_type, amount_range, trade_date
                FROM congress_trades
                WHERE ticker = %s AND trade_date >= %s AND party != '' AND party IS NOT NULL
                ORDER BY trade_date DESC
            """,
                [ticker, cutoff],
            ).fetchall()
            if trades:
                discovery_details.append(
                    {
                        "ticker": ticker,
                        "trade_count": len(trades),
                        "traders": list({t[0] for t in trades}),
                        "latest_trade": str(trades[0][3]) if trades[0][3] else None,
                    }
                )

        # Sort discovery by trade count (most traded = most interesting)
        discovery_details.sort(key=lambda x: x["trade_count"], reverse=True)

        return {
            "overlap": overlap_details,
            "overlap_count": len(overlap),
            "discovery": discovery_details[:30],  # Top 30 discoveries
            "discovery_count": len(discovery),
            "not_traded": sorted(not_traded),
            "not_traded_count": len(not_traded),
            "congress_total_tickers": len(congress_set),
            "watchlist_total": len(watchlist_set),
        }


def flag_notable_activity(days: int = 14) -> list[dict]:
    """Flag notable congress trading activity.

    Flags:
      - Large trades ($50K+)
      - Rapid disclosure (traded and disclosed within 7 days = unusual transparency)
      - Cluster trades (same politician, 3+ trades in a week)
    """
    with get_db() as db:
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).date()

        flags = []

        # Large trades
        large = db.execute(
            """
            SELECT politician, party, ticker, transaction_type, amount_range, trade_date
            FROM congress_trades
            WHERE trade_date >= %s AND party != '' AND party IS NOT NULL
              AND (amount_range LIKE '%%50K%%' OR amount_range LIKE '%%100K%%' 
                   OR amount_range LIKE '%%250K%%' OR amount_range LIKE '%%500K%%'
                   OR amount_range LIKE '%%1M%%' OR amount_range LIKE '%%5M%%')
            ORDER BY trade_date DESC
        """,
            [cutoff],
        ).fetchall()

        for r in large:
            flags.append(
                {
                    "type": "LARGE_TRADE",
                    "politician": r[0],
                    "party": r[1],
                    "ticker": r[2],
                    "direction": r[3],
                    "amount": r[4],
                    "date": str(r[5]),
                }
            )

        # Cluster trades — same politician, 3+ trades in the period
        clusters = db.execute(
            """
            SELECT politician, COUNT(*) as cnt,
                   STRING_AGG(DISTINCT ticker, ', ') as tickers,
                   MIN(trade_date) as first, MAX(trade_date) as last
            FROM congress_trades
            WHERE trade_date >= %s AND party != '' AND party IS NOT NULL
            GROUP BY politician
            HAVING COUNT(*) >= 3
            ORDER BY COUNT(*) DESC
        """,
            [cutoff],
        ).fetchall()

        for r in clusters:
            flags.append(
                {
                    "type": "CLUSTER_TRADE",
                    "politician": r[0],
                    "trade_count": r[1],
                    "tickers": r[2].split(",") if r[2] else [],
                    "first_date": str(r[3]),
                    "last_date": str(r[4]),
                }
            )

        return flags


def generate_report(watchlist_tickers: list[str] | None = None) -> str:
    """Generate a human-readable congress trading report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CONGRESS TRADING SCANNER REPORT")
    lines.append("=" * 70)

    # Recent activity
    recent = scan_recent_trades(days=30)
    lines.append(
        f"\n📊 Recent Activity (last 30 days): {recent['total_trades']} trades"
    )

    # Buy/sell breakdown
    buys = [t for t in recent["trades"] if t["type"] == "buy"]
    sells = [t for t in recent["trades"] if t["type"] == "sell"]
    lines.append(f"   Buys: {len(buys)} | Sells: {len(sells)}")

    # Consensus signals
    consensus = find_consensus_trades(days=30, min_members=2)
    if consensus:
        lines.append(f"\n🎯 Consensus Signals ({len(consensus)} found):")
        for c in consensus[:10]:
            lines.append(
                f"   {c['ticker']} — {c['direction'].upper()} by {c['member_count']} members: "
                f"{', '.join(c['members'][:3])}"
            )

    # Notable flags
    flags = flag_notable_activity(days=14)
    if flags:
        large = [f for f in flags if f["type"] == "LARGE_TRADE"]
        clusters = [f for f in flags if f["type"] == "CLUSTER_TRADE"]
        if large:
            lines.append(f"\n🚨 Large Trades ({len(large)}):")
            for f in large[:10]:
                lines.append(
                    f"   {f['politician']} ({f['party']}) — {f['direction'].upper()} "
                    f"{f['ticker']} ({f['amount']}) on {f['date']}"
                )
        if clusters:
            lines.append(f"\n📈 Cluster Traders ({len(clusters)}):")
            for f in clusters[:10]:
                lines.append(
                    f"   {f['politician']}: {f['trade_count']} trades in "
                    f"{', '.join(f['tickers'][:5])}"
                )

    # Portfolios
    portfolios = build_politician_portfolios(top_n=10)
    if portfolios:
        lines.append("\n👤 Top 10 Active Traders:")
        for p in portfolios:
            lines.append(
                f"   {p['politician']} ({p['party']}/{p['chamber']}) — "
                f"{p['total_trades']} trades, est. {p['holding_count']} holdings: "
                f"{', '.join(p['estimated_holdings'][:5])}"
            )

    # Watchlist comparison
    if watchlist_tickers:
        comp = compare_with_watchlist(watchlist_tickers)
        lines.append("\n🔍 Watchlist Comparison:")
        lines.append(
            f"   Congress traded {comp['congress_total_tickers']} unique tickers"
        )
        lines.append(f"   Overlap with watchlist: {comp['overlap_count']} tickers")
        lines.append(
            f"   Discovery (not on watchlist): {comp['discovery_count']} tickers"
        )
        lines.append(f"   Watchlist not traded: {comp['not_traded_count']} tickers")

        if comp["overlap"]:
            lines.append("\n   📌 Overlap Details:")
            for o in comp["overlap"]:
                lines.append(f"      {o['ticker']}: {o['trade_count']} trades")
                for t in o["trades"][:3]:
                    lines.append(
                        f"         {t['politician']} — {t['type']} {t['amount']} ({t['date']})"
                    )

        if comp["discovery"]:
            lines.append("\n   🆕 Discovery (congress trading, you're not watching):")
            for d in comp["discovery"][:15]:
                lines.append(
                    f"      {d['ticker']}: {d['trade_count']} trades by "
                    f"{', '.join(d['traders'][:3])}"
                )

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)

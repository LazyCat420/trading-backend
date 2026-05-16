"""
Graph queries — relationship map builder for LLM context enrichment.

Queries ticker_metadata, ticker_correlations, congress_trades, and sec_13f_holdings
to build a structured "RELATIONSHIP MAP" section for the context blob.

Usage:
    from app.graph.graph_queries import build_relationship_map
    map_text = build_relationship_map("NVDA")
"""

import logging

from app.db.connection import get_db

logger = logging.getLogger(__name__)


def build_relationship_map(ticker: str) -> str:
    """Build a text section with cross-ticker relationships for the LLM.

    Sections:
      - Sector / Industry
      - Correlated tickers (30d)
      - Inversely correlated (hedging)
      - Congress network
      - Fund overlap
    """
    with get_db() as db:
        t = ticker.upper().strip()
        lines = ["\n## RELATIONSHIP MAP"]

        # ── Sector & Industry ──
        try:
            meta = db.execute(
                "SELECT name, sector, industry, market_cap_tier, asset_class "
                "FROM ticker_metadata WHERE ticker = %s",
                [t],
            ).fetchone()
            if meta:
                lines.append(f"Sector: {meta[1]} > {meta[2]}")
                lines.append(f"Cap Tier: {meta[3]}, Asset Class: {meta[4]}")

                # Sector peers
                peers = db.execute(
                    "SELECT ticker, name, industry FROM ticker_metadata "
                    "WHERE sector = %s AND ticker != %s "
                    "ORDER BY market_cap DESC LIMIT 5",
                    [meta[1], t],
                ).fetchall()
                if peers:
                    peer_str = ", ".join(f"{p[0]} ({p[2]})" for p in peers)
                    lines.append(f"Sector Peers: {peer_str}")
        except Exception:
            pass

        # ── Correlations (30d) ──
        try:
            corr_rows = db.execute(
                "SELECT ticker_a, ticker_b, correlation, tier "
                "FROM ticker_correlations "
                "WHERE (ticker_a = %s OR ticker_b = %s) AND period = '30d' "
                "AND correlation > 0 "
                "ORDER BY correlation DESC LIMIT 10",
                [t, t],
            ).fetchall()

            highly = []
            moderate = []
            weak = []
            for r in corr_rows:
                other = r[1] if r[0] == t else r[0]
                entry = f"{other} (r={r[2]:.2f})"
                if r[3] == "highly_correlated":
                    highly.append(entry)
                elif r[3] == "correlated":
                    moderate.append(entry)
                elif r[3] == "weakly_correlated":
                    weak.append(entry)

            if highly:
                lines.append(f"Highly Correlated (30d): {', '.join(highly)}")
            if moderate:
                lines.append(f"Correlated (30d): {', '.join(moderate)}")
            if weak:
                lines.append(f"Weakly Correlated (30d): {', '.join(weak)}")
        except Exception:
            pass

        # ── Inverse correlations (hedging) ──
        try:
            inv_rows = db.execute(
                "SELECT ticker_a, ticker_b, correlation "
                "FROM ticker_correlations "
                "WHERE (ticker_a = %s OR ticker_b = %s) AND period = '30d' "
                "AND correlation < -0.4 "
                "ORDER BY correlation ASC LIMIT 5",
                [t, t],
            ).fetchall()
            if inv_rows:
                inv_str = ", ".join(
                    f"{(r[1] if r[0] == t else r[0])} (r={r[2]:.2f})" for r in inv_rows
                )
                lines.append(f"Inversely Correlated (hedging): {inv_str}")
        except Exception:
            pass

        # ── Congress network ──
        try:
            congress = db.execute(
                "SELECT DISTINCT politician, party, transaction_type "
                "FROM congress_trades WHERE ticker = %s "
                "ORDER BY trade_date DESC LIMIT 5",
                [t],
            ).fetchall()
            if congress:
                for c in congress:
                    # What else did this politician trade%s
                    others = db.execute(
                        "SELECT DISTINCT ticker FROM congress_trades "
                        "WHERE politician = %s AND ticker != %s "
                        "ORDER BY trade_date DESC LIMIT 5",
                        [c[0], t],
                    ).fetchall()
                    other_str = ", ".join(o[0] for o in others) if others else "none"
                    lines.append(
                        f"Congress: {c[0]} ({c[1]}) {c[2]} — also trades: {other_str}"
                    )
        except Exception:
            pass

        # ── Fund overlap ──
        try:
            funds = db.execute(
                "SELECT f.filer_name, h.shares, h.value_usd "
                "FROM sec_13f_holdings h "
                "LEFT JOIN sec_13f_filers f ON h.cik = f.cik "
                "WHERE h.ticker = %s "
                "ORDER BY h.value_usd DESC LIMIT 3",
                [t],
            ).fetchall()
            if funds:
                for f in funds:
                    # What else does this fund hold%s
                    others = db.execute(
                        "SELECT DISTINCT h.ticker FROM sec_13f_holdings h "
                        "LEFT JOIN sec_13f_filers f ON h.cik = f.cik "
                        "WHERE f.filer_name = %s AND h.ticker != %s "
                        "ORDER BY h.value_usd DESC LIMIT 5",
                        [f[0], t],
                    ).fetchall()
                    other_str = ", ".join(o[0] for o in others) if others else "none"
                    val = f"${f[2]:,.0f}" if f[2] else "%s"
                    lines.append(
                        f"Fund: {f[0]} holds {f[1]:,} shares ({val}) — also holds: {other_str}"
                    )
        except Exception:
            pass

        # ── Cognition V2: Ontology subgraph context ──
        try:
            from app.cognition.ontology.ontology_builder import BrainGraph

            activated = BrainGraph.get_activated_context(ticker, max_chars=1500)
            if activated:
                lines.append(activated)
        except Exception as e:
            logger.debug("brain graph context failed: %s", e)

        # Only return if we have actual data
        if len(lines) <= 1:
            return ""

        return "\n".join(lines) + "\n"

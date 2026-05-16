"""
Gap Analyzer — scans a context blob and identifies missing data sections.

Returns a JSON list of data gaps that the pipeline can fill with
targeted collectors before the final decision round.

Usage:
    from app.agents.gap_analyzer import analyze_gaps
    gaps = analyze_gaps(context_blob, ticker)
    # gaps = ["options_flow", "insider_transactions", "earnings_history"]
"""

import logging

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# Registry of detectable data types and what to look for in the context
_GAP_CHECKS = {
    "options_flow": {
        "context_markers": ["put/call", "options", "calls", "puts", "open interest"],
        "db_table": None,  # Not stored in DB yet
        "description": "Options flow data (put/call ratio, volume)",
        "priority": "high",
    },
    "insider_transactions": {
        "context_markers": ["insider", "officer", "director bought", "insider sell"],
        "db_table": None,
        "description": "Insider buy/sell activity",
        "priority": "high",
    },
    "earnings_history": {
        "context_markers": [
            "earnings beat",
            "earnings miss",
            "EPS estimate",
            "earnings surprise",
        ],
        "db_table": None,
        "description": "Historical earnings vs estimates",
        "priority": "high",
    },
    "sector_peers": {
        "context_markers": ["RELATIONSHIP MAP", "Sector Peers", "Sector:"],
        "db_table": "ticker_metadata",
        "description": "Sector/industry peer comparison",
        "priority": "medium",
    },
    "correlations": {
        "context_markers": ["Correlated", "r=0.", "RELATIONSHIP MAP"],
        "db_table": "ticker_correlations",
        "description": "Cross-ticker correlation data",
        "priority": "medium",
    },
    "fundamentals": {
        "context_markers": ["PE", "MarketCap", "ForwardPE", "Fundamentals"],
        "db_table": "fundamentals",
        "description": "Company fundamentals",
        "priority": "high",
    },
    "technicals": {
        "context_markers": ["RSI", "MACD", "SMA", "Technical Indicators"],
        "db_table": "technicals",
        "description": "Technical analysis indicators",
        "priority": "high",
    },
}

# Maps gap names → collector functions
_COLLECTOR_REGISTRY: dict[str, callable] = {}


def register_collector(gap_name: str, fn: callable):
    """Register a collector function for a specific gap type."""
    _COLLECTOR_REGISTRY[gap_name] = fn


def analyze_gaps(context_blob: str, ticker: str) -> list[dict]:
    """Scan context and return list of missing data sections.

    Returns list of dicts: [{"gap": "options_flow", "priority": "high", ...}]
    """
    context_lower = context_blob.lower() if context_blob else ""
    gaps = []

    for gap_name, check in _GAP_CHECKS.items():
        # Check if any marker appears in the context
        found = any(
            marker.lower() in context_lower for marker in check["context_markers"]
        )

        if not found:
            # Also check DB for data existence
            if check["db_table"]:
                try:
                    with get_db() as db:
                        count = db.execute(
                            f"SELECT COUNT(*) FROM {check['db_table']} WHERE ticker = %s",
                            [ticker.upper()],
                        ).fetchone()
                        if count and count[0] > 0:
                            continue  # Data exists in DB, just not in context
                except Exception:
                    pass

            gaps.append(
                {
                    "gap": gap_name,
                    "priority": check["priority"],
                    "description": check["description"],
                    "has_collector": gap_name in _COLLECTOR_REGISTRY,
                }
            )

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda g: priority_order.get(g["priority"], 9))

    if gaps:
        logger.info(
            "gap_analyzer: %s has %d gaps: %s",
            ticker,
            len(gaps),
            ", ".join(g["gap"] for g in gaps),
        )

    return gaps


async def fill_gaps(
    gaps: list[dict],
    ticker: str,
    max_fills: int = 3,
) -> dict[str, bool]:
    """Run collectors for identified gaps (max 1 enrichment round).

    Returns dict of {gap_name: success_bool}.
    """
    results = {}
    filled = 0

    for gap in gaps:
        if filled >= max_fills:
            break

        gap_name = gap["gap"]
        collector = _COLLECTOR_REGISTRY.get(gap_name)
        if not collector:
            logger.debug("gap_analyzer: no collector for %s", gap_name)
            results[gap_name] = False
            continue

        try:
            logger.info("gap_analyzer: filling %s for %s", gap_name, ticker)
            result = await collector(ticker)
            results[gap_name] = bool(result)
            filled += 1
        except Exception as e:
            logger.warning("gap_analyzer: %s collector failed: %s", gap_name, e)
            results[gap_name] = False

    return results


# Auto-register available collectors on import
def _auto_register():
    """Register collectors that exist in the codebase."""
    try:
        from app.collectors.options_collector import collect_options

        register_collector("options_flow", collect_options)
    except ImportError:
        pass

    try:
        from app.collectors.insider_collector import collect_insider

        register_collector("insider_transactions", collect_insider)
    except ImportError:
        pass

    try:
        from app.collectors.earnings_collector import collect_earnings

        register_collector("earnings_history", collect_earnings)
    except ImportError:
        pass


_auto_register()

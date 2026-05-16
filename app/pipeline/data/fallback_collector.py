"""
Fallback Collector -- agentic data collection for tickers where standard
collectors fail.

When standard collectors can't provide data (yfinance rejected, no price
history, unknown asset type), this module uses the Hermes browser agent
to scrape financial data from Yahoo Finance.

Architecture:
  - Runs AFTER per-ticker collection (Pass 4.5 in data_phase.py)
  - Only activates for tickers with critical data gaps
  - Uses existing hermes_web_research tool (already has guardrails)
  - Stores results in the fallback_data table for downstream agents
  - Non-fatal: failures here don't block the pipeline
"""

import json
import logging
from datetime import datetime, timezone
from typing import Callable

from app.config.config_tickers import ALT_ASSET_TICKERS, CRYPTO_TICKERS
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# Minimum data thresholds before triggering fallback
_MIN_PRICE_ROWS = 5
_MIN_NEWS_ROWS = 1


def _ensure_table() -> None:
    """Create fallback_data table if it doesn't exist."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS fallback_data (
                id              SERIAL PRIMARY KEY,
                ticker          TEXT NOT NULL,
                data_type       TEXT NOT NULL,
                data_json       JSONB NOT NULL,
                source          TEXT DEFAULT 'hermes_yahoo',
                collected_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                cycle_id        TEXT
            )
        """)
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_fallback_ticker ON fallback_data(ticker)"
        )


def detect_data_gaps(tickers: list[str]) -> list[dict]:
    """Check which tickers have critical data gaps after standard collection.

    Only flags tickers that passed all validation gates but still lack
    the minimum data needed for analysis.

    Args:
        tickers: List of ticker symbols to check.

    Returns:
        List of dicts with ticker, missing categories, and context.
        Example: [{"ticker": "XYZ", "gaps": ["price", "fundamentals"],
                   "context": "..."}]
    """
    if not tickers:
        return []

    with get_db() as db:
        gaps = []

        for ticker in tickers:
            ticker_gaps = []
            t = ticker.upper()

            # Skip crypto tickers for price_history check — they use asset_prices
            if t not in CRYPTO_TICKERS:
                try:
                    price_count = db.execute(
                        "SELECT COUNT(*) FROM price_history WHERE ticker = %s",
                        [t],
                    ).fetchone()[0]
                    if price_count < _MIN_PRICE_ROWS:
                        ticker_gaps.append("price")
                except Exception:
                    ticker_gaps.append("price")

            # Skip alt-assets for fundamentals — they don't have P/E, D/E, etc.
            if t not in ALT_ASSET_TICKERS:
                try:
                    fund_count = db.execute(
                        "SELECT COUNT(*) FROM fundamentals WHERE ticker = %s",
                        [t],
                    ).fetchone()[0]
                    if fund_count == 0:
                        ticker_gaps.append("fundamentals")
                except Exception:
                    ticker_gaps.append("fundamentals")

            # News check — applies to all asset types
            try:
                news_count = db.execute(
                    "SELECT COUNT(*) FROM news_articles WHERE ticker = %s",
                    [t],
                ).fetchone()[0]
                if news_count < _MIN_NEWS_ROWS:
                    ticker_gaps.append("news")
            except Exception:
                ticker_gaps.append("news")

            if ticker_gaps:
                # Fetch discovery context if available
                context = ""
                try:
                    row = db.execute(
                        "SELECT context FROM discovered_tickers WHERE ticker = %s",
                        [t],
                    ).fetchone()
                    if row and row[0]:
                        context = row[0]
                except Exception:
                    pass

                gaps.append(
                    {
                        "ticker": t,
                        "gaps": ticker_gaps,
                        "context": context,
                    }
                )

        return gaps


def _build_hermes_prompt(ticker: str, gaps: list[str], context: str) -> str:
    """Build a focused research prompt for Hermes based on data gaps.

    The prompt instructs Hermes to gather specific data types that are
    missing from standard collectors.
    """
    gap_instructions = []
    if "price" in gaps:
        gap_instructions.append(
            "- Current price and recent price action (1d, 5d, 1mo change)"
        )
        gap_instructions.append("- 52-week high and low")
    if "fundamentals" in gaps:
        gap_instructions.append("- Market capitalization")
        gap_instructions.append("- P/E ratio and Forward P/E (if available)")
        gap_instructions.append("- Revenue and earnings growth")
        gap_instructions.append("- Debt-to-equity ratio")
    if "news" in gaps:
        gap_instructions.append(
            "- Top 3 recent news headlines with dates and brief summaries"
        )

    context_note = ""
    if context:
        context_note = (
            f'\nDiscovery context: "{context}"\n'
            "Use this context to understand what kind of asset this is."
        )

    return (
        f"Research the financial instrument '{ticker}' and extract the "
        f"following data. Search finance.yahoo.com or other financial data "
        f"sites.\n"
        f"{context_note}\n"
        f"Data needed:\n" + "\n".join(gap_instructions) + "\n\n"
        "Return ONLY valid JSON with the extracted data. Example format:\n"
        "{\n"
        '  "ticker": "XYZ",\n'
        '  "price": {"current": 123.45, "change_1d_pct": -1.2, '
        '"high_52w": 150.0, "low_52w": 90.0},\n'
        '  "fundamentals": {"market_cap": 5000000000, "pe_ratio": 25.3, '
        '"revenue_growth": 0.15},\n'
        '  "news": [{"title": "...", "date": "2026-04-23", '
        '"summary": "..."}]\n'
        "}\n"
        "Only include sections for data you can actually find. "
        "Do NOT fabricate numbers."
    )


def _parse_hermes_response(raw: str) -> dict | None:
    """Extract structured data from Hermes response.

    Handles cases where Hermes wraps JSON in markdown or adds commentary.
    """
    if not raw:
        return None

    # Try to parse the raw response as JSON directly
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to extract JSON from Hermes response wrapper
    try:
        wrapper = json.loads(raw)
        content = wrapper.get("response", "")
        if content:
            # Try to find JSON in the content
            import re

            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
            if json_match:
                return json.loads(json_match.group(0))
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def _store_fallback_data(
    ticker: str,
    data: dict,
    gaps: list[str],
    cycle_id: str = "",
) -> int:
    """Store parsed Hermes data in the fallback_data table.

    Returns number of records stored.
    """
    _ensure_table()
    with get_db() as db:
        stored = 0
        now = datetime.now(timezone.utc)

        for gap_type in gaps:
            section_data = None
            if gap_type == "price" and "price" in data:
                section_data = data["price"]
            elif gap_type == "fundamentals" and "fundamentals" in data:
                section_data = data["fundamentals"]
            elif gap_type == "news" and "news" in data:
                section_data = data["news"]

            if section_data:
                try:
                    db.execute(
                        "INSERT INTO fallback_data "
                        "(ticker, data_type, data_json, source, collected_at, cycle_id) "
                        "VALUES (%s, %s, %s::jsonb, 'hermes_yahoo', %s, %s)",
                        [ticker, gap_type, json.dumps(section_data), now, cycle_id],
                    )
                    stored += 1
                except Exception as e:
                    logger.warning(
                        "[FALLBACK] Failed to store %s data for %s: %s",
                        gap_type,
                        ticker,
                        e,
                    )

        return stored


async def fill_gaps_via_hermes(
    gaps: list[dict],
    emit: Callable | None = None,
    cycle_id: str = "",
    max_tickers: int = 5,
) -> dict:
    """Use Hermes web research to fill data gaps for tickers.

    Args:
        gaps: Output from detect_data_gaps().
        emit: Pipeline event emitter.
        cycle_id: Current cycle ID for audit trail.
        max_tickers: Maximum tickers to research (cap to avoid long delays).

    Returns:
        Summary dict with counts of attempted, filled, and failed tickers.
    """
    from app.tools.web_tools import hermes_web_research

    summary = {"attempted": 0, "filled": 0, "failed": 0, "details": []}

    # Cap to avoid pipeline delays
    gaps_to_process = gaps[:max_tickers]

    for gap_info in gaps_to_process:
        ticker = gap_info["ticker"]
        gap_types = gap_info["gaps"]
        context = gap_info.get("context", "")

        summary["attempted"] += 1

        if emit:
            emit(
                "collecting",
                f"fallback_{ticker}",
                f"Hermes researching {ticker} (gaps: {', '.join(gap_types)})...",
                status="running",
            )

        try:
            prompt = _build_hermes_prompt(ticker, gap_types, context)
            raw_result = await hermes_web_research(
                query=prompt,
                ticker=ticker,
            )

            parsed = _parse_hermes_response(raw_result)
            if parsed:
                stored = _store_fallback_data(ticker, parsed, gap_types, cycle_id)
                if stored > 0:
                    summary["filled"] += 1
                    summary["details"].append(
                        {
                            "ticker": ticker,
                            "gaps_filled": gap_types,
                            "stored": stored,
                        }
                    )
                    logger.info(
                        "[FALLBACK] %s: filled %d gap(s) via Hermes: %s",
                        ticker,
                        stored,
                        ", ".join(gap_types),
                    )
                    if emit:
                        emit(
                            "collecting",
                            f"fallback_{ticker}",
                            f"Hermes filled {stored} data gap(s) for {ticker}",
                            status="ok",
                        )
                else:
                    summary["failed"] += 1
                    logger.warning(
                        "[FALLBACK] %s: Hermes returned data but nothing stored",
                        ticker,
                    )
            else:
                summary["failed"] += 1
                logger.warning("[FALLBACK] %s: could not parse Hermes response", ticker)
                if emit:
                    emit(
                        "collecting",
                        f"fallback_{ticker}",
                        f"Hermes research for {ticker} returned unparseable data",
                        status="warning",
                    )

        except Exception as e:
            summary["failed"] += 1
            logger.warning("[FALLBACK] %s: Hermes research failed: %s", ticker, e)
            if emit:
                emit(
                    "collecting",
                    f"fallback_{ticker}",
                    f"Hermes fallback failed for {ticker}: {e}",
                    status="error",
                )

    logger.info(
        "[FALLBACK] Summary: %d attempted, %d filled, %d failed",
        summary["attempted"],
        summary["filled"],
        summary["failed"],
    )

    return summary


def get_fallback_context(ticker: str) -> str | None:
    """Retrieve stored fallback data for a ticker, formatted for context injection.

    Called by context_builder when primary data sources are empty.

    Returns:
        Formatted string for LLM context, or None if no fallback data exists.
    """
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT data_type, data_json, collected_at FROM fallback_data "
                "WHERE ticker = %s ORDER BY collected_at DESC LIMIT 3",
                [ticker.upper()],
            ).fetchall()

            if not rows:
                return None

            sections = ["\n## Fallback Data (Hermes Web Research)"]
            sections.append(
                "Note: This data was collected via web research when standard "
                "collectors failed. Verify independently before making decisions.\n"
            )

            for row in rows:
                data_type = row[0]
                data_json = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                collected = row[2]

                if data_type == "price":
                    sections.append(f"### Price Data (collected {collected})")
                    for key, val in data_json.items():
                        sections.append(f"  {key}: {val}")
                elif data_type == "fundamentals":
                    sections.append(f"### Fundamentals (collected {collected})")
                    for key, val in data_json.items():
                        sections.append(f"  {key}: {val}")
                elif data_type == "news":
                    sections.append(f"### Recent News (collected {collected})")
                    if isinstance(data_json, list):
                        for item in data_json[:5]:
                            title = item.get("title", "%s")
                            date = item.get("date", "%s")
                            summary = item.get("summary", "")
                            sections.append(f"  [{date}] {title}")
                            if summary:
                                sections.append(f"    {summary[:200]}")

            return "\n".join(sections) + "\n"

    except Exception as e:
        logger.debug("[FALLBACK] get_fallback_context failed for %s: %s", ticker, e)
        return None

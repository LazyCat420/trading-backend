"""
Macro Strategy Scout — LLM-powered macro analysis that runs in parallel
with per-ticker data collection.

While the pipeline waits for scraper data to arrive for each ticker
(~3 min per ticker), the scout productively:
  1. Reads FRED macro indicators already in DB
  2. Scans general news / Reddit / YouTube for macro themes
  3. Asks the LLM to identify the current macro regime
  4. Generates targeted search queries for deeper research
  5. Suggests tickers to investigate (added to next-cycle watchlist)
  6. Produces a structured "Macro Strategy Memo" injected into
     every ticker's analysis context

Usage:
    from app.pipeline.analysis.macro_scout import run_macro_scout
    memo = await run_macro_scout(emit=my_callback)
"""

import logging
import time
from typing import Callable

from app.db.connection import get_db
from app.utils.pipeline_utils import noop as _noop
from app.utils.text_utils import sanitize_ascii

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Data Gathering (pure DB reads — no LLM calls)
# ─────────────────────────────────────────────────────────────────────


def _gather_fred_summary() -> str:
    """Pull latest FRED macro indicators from DB."""
    with get_db() as db:
        lines = []
        try:
            rows = db.execute("""
                SELECT indicator, value, date
                FROM macro_indicators
                ORDER BY date DESC
                LIMIT 30
            """).fetchall()
            if rows:
                lines.append("## FRED Macro Indicators (Recent)")
                seen: set[str] = set()  # deduplicate by indicator name
                for name, value, date in rows:
                    if name in seen:
                        continue
                    seen.add(name)
                    lines.append(f"  - {name}: {value} ({date})")
        except Exception as e:
            logger.warning("[PIPELINE] [macro_scout] FRED query failed: %s", e)
            lines.append("## FRED Macro Indicators: [unavailable]")
        return "\n".join(lines)


def _gather_commodity_summary() -> str:
    """Pull latest commodity/futures prices from DB."""
    with get_db() as db:
        lines = []
        try:
            # Get the most recent price for each commodity
            rows = db.execute("""
                SELECT symbol, close, date
                FROM asset_prices
                WHERE asset_class = 'commodity'
                AND date = (SELECT MAX(date) FROM asset_prices
                            WHERE symbol = asset_prices.symbol
                            AND asset_class = 'commodity')
                ORDER BY symbol
            """).fetchall()
            if rows:
                lines.append("## Commodity Prices (Latest)")
                for symbol, close, date in rows:
                    lines.append(f"  - {symbol}: ${close:.2f} ({date})")
        except Exception as e:
            logger.warning("[PIPELINE] [macro_scout] Commodity query failed: %s", e)
        return "\n".join(lines)


def _gather_general_news(limit: int = 20) -> str:
    """Pull recent general market news from DB."""
    with get_db() as db:
        lines = []
        try:
            rows = db.execute(
                """
                SELECT title, publisher, published_at,
                       COALESCE(llm_summary, summary) AS best_summary
                FROM news_articles
                WHERE published_at > CURRENT_TIMESTAMP - INTERVAL '3 days'
                ORDER BY published_at DESC
                LIMIT %s
            """,
                [limit],
            ).fetchall()
            if rows:
                lines.append(f"## Recent Market News ({len(rows)} articles)")
                for title, publisher, pub_at, summary in rows:
                    snippet = (summary or "")[:200]
                    lines.append(f"  - [{publisher}] {title} ({pub_at})")
                    if snippet:
                        lines.append(f"    {snippet}")
        except Exception as e:
            logger.warning("[PIPELINE] [macro_scout] News query failed: %s", e)
        return "\n".join(lines)


def _gather_reddit_macro(limit: int = 15) -> str:
    """Pull top Reddit posts that aren't ticker-specific (macro themes)."""
    with get_db() as db:
        lines = []
        try:
            rows = db.execute(
                """
                SELECT subreddit, title, body, score
                FROM reddit_posts
                WHERE created_utc > CURRENT_TIMESTAMP - INTERVAL '3 days'
                ORDER BY score DESC
                LIMIT %s
            """,
                [limit],
            ).fetchall()
            if rows:
                lines.append(f"## Top Reddit Posts ({len(rows)} posts)")
                for sub, title, body, score in rows:
                    snippet = (body or "")[:150]
                    lines.append(f"  - r/{sub} (score {score}): {title}")
                    if snippet:
                        lines.append(f"    {snippet}")
        except Exception as e:
            logger.warning("[PIPELINE] [macro_scout] Reddit query failed: %s", e)
        return "\n".join(lines)


def _gather_congress_signals() -> str:
    """Pull recent congress trading signals."""
    with get_db() as db:
        lines = []
        try:
            rows = db.execute("""
                SELECT politician, party, ticker, transaction_type,
                       amount_range, trade_date
                FROM congress_trades
                WHERE trade_date > CURRENT_TIMESTAMP - INTERVAL '30 days'
                ORDER BY trade_date DESC
                LIMIT 15
            """).fetchall()
            if rows:
                lines.append(f"## Recent Congress Trades ({len(rows)} trades)")
                for pol, party, tkr, txn, amt, dt in rows:
                    lines.append(f"  - {pol} ({party}): {txn} {tkr} [{amt}] ({dt})")
        except Exception as e:
            logger.warning("[PIPELINE] [macro_scout] Congress query failed: %s", e)
        return "\n".join(lines)


def _gather_portfolio_context() -> str:
    """Pull current portfolio state for strategic awareness."""
    lines = []
    try:
        from app.trading.portfolio import get_current_state

        state = get_current_state()
        lines.append("## Current Portfolio State")
        lines.append(f"  Cash: ${state['cash']:,.2f}")
        lines.append(f"  Total Value: ${state['total_value']:,.2f}")
        lines.append(f"  Open Positions: {state['position_count']}")
        for p in state.get("positions", []):
            entry = p["avg_entry_price"]
            curr = p["current_price"]
            pnl = ((curr - entry) / entry * 100) if entry else 0
            lines.append(
                f"  - {p['ticker']}: {p['qty']:.2f} shares "
                f"(entry ${entry:.2f}, now ${curr:.2f}, {pnl:+.1f}%)"
            )
    except Exception as e:
        logger.warning("[PIPELINE] [macro_scout] Portfolio query failed: %s", e)
    return "\n".join(lines)


def _gather_watchlist() -> str:
    """Pull current watchlist for context."""
    with get_db() as db:
        lines = []
        try:
            rows = db.execute(
                "SELECT ticker FROM watchlist WHERE status = 'active' "
                "ORDER BY added_at DESC"
            ).fetchall()
            if rows:
                tickers = [r[0] for r in rows]
                lines.append(f"## Active Watchlist ({len(tickers)} tickers)")
                lines.append(f"  {', '.join(tickers)}")
        except Exception as e:
            logger.warning("[PIPELINE] [macro_scout] Watchlist query failed: %s", e)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# LLM Analysis
# ─────────────────────────────────────────────────────────────────────

MACRO_SCOUT_PROMPT = """You are a macro strategist for an autonomous trading bot.

Below is a snapshot of REAL macro data, market news, social sentiment,
congress trades, commodity prices, and the bot's current portfolio.

Your job is to produce a structured **Macro Strategy Memo** that will be
prepended to every individual ticker analysis. The per-ticker analysts
CANNOT see macro data themselves — they rely on YOUR memo for big-picture
context.

## Your output MUST contain EXACTLY these sections:

### MACRO REGIME
One of: RISK_ON, RISK_OFF, TRANSITIONAL, UNCERTAIN
Brief explanation (2-3 sentences) citing specific data points.

### KEY THEMES
List 3-5 dominant macro themes right now. For each:
- Theme name
- Evidence (cite specific data from below)
- Impact on equities (bullish/bearish/sector-specific)

### SECTOR OUTLOOK
For each major sector (Tech, Healthcare, Energy, Financials, Consumer,
Industrials, Materials), give a 1-line outlook based on the macro data.

### SEARCH QUERIES
Generate 3-5 specific search queries the bot should use for deeper
research in the next cycle. These should be timely and actionable, e.g.:
- "semiconductor tariff impact Q2 2026"
- "OPEC production cut timeline"

### WATCHLIST SUGGESTIONS
Suggest 3-8 tickers that the bot should add to its watchlist for the
NEXT cycle, based on macro themes. For each:
- Ticker symbol
- Reason (tied to a specific macro theme above)
- Only suggest liquid US stocks/ETFs (no penny stocks, no OTC)

### RISK WARNINGS
List 2-3 active risks the per-ticker analysts should factor in.

Keep it concise and data-driven. Cite numbers from the data below.
Do NOT invent data — only reference what's provided.

---
DATA SNAPSHOT:
"""


async def _call_llm_for_memo(data_snapshot: str, emit: Callable) -> str:
    """Call vLLM to generate the macro strategy memo."""
    from app.services.vllm_client import llm, Priority

    system_prompt = (
        "You are a macro strategist. Produce structured analysis. "
        "Be concise and data-driven."
    )
    user_prompt = MACRO_SCOUT_PROMPT + data_snapshot

    text, tokens, ms = await llm.chat(
        system=system_prompt,
        user=user_prompt,
        temperature=0.3,
        max_tokens=2000,
        priority=Priority.LOW,  # don't block user chat or agents
        agent_name="macro_scout",
    )

    logger.info(
        "[PIPELINE] [macro_scout] LLM responded: %d chars, %d tokens, %dms",
        len(text),
        tokens,
        ms,
    )
    return text


def _extract_watchlist_suggestions(memo: str) -> list[str]:
    """Parse ticker suggestions from the memo text.

    Looks for lines like "- TICKER:" or "- **TICKER**:" in the
    WATCHLIST SUGGESTIONS section.
    """
    import re

    tickers: list[str] = []
    in_section = False

    for line in memo.split("\n"):
        upper = line.strip().upper()
        if "WATCHLIST SUGGESTIONS" in upper:
            in_section = True
            continue
        if in_section and line.strip().startswith("###"):
            break  # hit the next section
        if in_section and line.strip().startswith("-"):
            # Try to extract a ticker from patterns like "- NVDA:" or "- **NVDA**:"
            match = re.match(
                r"^\s*-\s*\**([A-Z]{1,5})\**\s*[:\-—]",
                line.strip(),
            )
            if match:
                tickers.append(match.group(1))

    return tickers


def _save_watchlist_suggestions(tickers: list[str]) -> int:
    """Add scout-suggested tickers to discovered_tickers for next cycle."""
    if not tickers:
        return 0
    with get_db() as db:
        added = 0
        for ticker in tickers:
            try:
                db.execute(
                    """
                    INSERT INTO discovered_tickers
                    (ticker, source, context, score, discovered_at)
                    VALUES (%s, 'macro_scout', 'Suggested by macro strategy scout', 0.6, CURRENT_TIMESTAMP)
                    ON CONFLICT (ticker) DO NOTHING
                """,
                    [ticker],
                )
                added += 1
            except Exception as e:
                logger.warning(
                    "[PIPELINE] [macro_scout] Failed to save suggestion %s: %s",
                    ticker,
                    e,
                )
        return added


# ─────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────


async def run_macro_scout(
    emit: Callable | None = None,
) -> str:
    """Run the macro strategy scout.

    Gathers all macro data from DB, sends to LLM, returns a structured
    Macro Strategy Memo. This memo is injected into every ticker's
    analysis context.

    Returns:
        Formatted memo string, or empty string on failure.
    """
    if emit is None:
        emit = _noop

    start = time.monotonic()
    logger.info("[PIPELINE] \n" + "=" * 60)
    logger.info("[PIPELINE] MACRO STRATEGY SCOUT: Starting...")
    logger.info("[PIPELINE] =" * 60)

    emit(
        "collecting",
        "macro_scout_start",
        "Macro Strategy Scout: Gathering macro data...",
        status="running",
    )

    # ── Gather all data from DB (no network calls, fast) ──
    t0 = time.monotonic()
    sections = []
    sections.append(_gather_fred_summary())
    sections.append(_gather_commodity_summary())
    sections.append(_gather_general_news(limit=20))
    sections.append(_gather_reddit_macro(limit=15))
    sections.append(_gather_congress_signals())
    sections.append(_gather_portfolio_context())
    sections.append(_gather_watchlist())

    data_snapshot = "\n\n".join(s for s in sections if s)
    gather_ms = int((time.monotonic() - t0) * 1000)

    if not data_snapshot.strip():
        emit(
            "collecting",
            "macro_scout_skip",
            "Macro Scout: No macro data in DB yet (first run?)",
            status="skipped",
            elapsed_ms=gather_ms,
        )
        return ""

    emit(
        "collecting",
        "macro_scout_data",
        f"Macro Scout: Gathered {len(data_snapshot):,} chars of macro data",
        status="ok",
        elapsed_ms=gather_ms,
    )
    logger.info(
        f"[PIPELINE]   [scout] Gathered {len(data_snapshot):,} chars ({gather_ms}ms)"
    )

    # ── Call LLM to build the memo ──
    t0 = time.monotonic()
    emit(
        "collecting",
        "macro_scout_llm",
        "Macro Scout: Generating strategy memo via LLM...",
        status="running",
    )

    try:
        raw_memo = await _call_llm_for_memo(data_snapshot, emit)
    except Exception as e:
        llm_ms = int((time.monotonic() - t0) * 1000)
        logger.error("[PIPELINE] [macro_scout] LLM call failed: %s", e)
        emit(
            "collecting",
            "macro_scout_error",
            f"Macro Scout: LLM failed — {e}",
            status="error",
            elapsed_ms=llm_ms,
        )
        return ""

    llm_ms = int((time.monotonic() - t0) * 1000)

    if not raw_memo or len(raw_memo.strip()) < 50:
        emit(
            "collecting",
            "macro_scout_empty",
            "Macro Scout: LLM returned empty/short response",
            status="error",
            elapsed_ms=llm_ms,
        )
        return ""

    # Sanitize for Windows/RLM compatibility
    memo = sanitize_ascii(raw_memo)

    emit(
        "collecting",
        "macro_scout_llm",
        f"Macro Scout: Memo generated ({len(memo):,} chars, {llm_ms}ms)",
        status="ok",
        data={"chars": len(memo)},
        elapsed_ms=llm_ms,
    )
    logger.info(f"[PIPELINE]   [scout] LLM memo: {len(memo):,} chars ({llm_ms}ms)")

    # ── Extract and save watchlist suggestions for next cycle ──
    suggestions = _extract_watchlist_suggestions(memo)
    if suggestions:
        saved = _save_watchlist_suggestions(suggestions)
        emit(
            "collecting",
            "macro_scout_suggestions",
            f"Macro Scout: Suggested {len(suggestions)} tickers "
            f"for next cycle: {', '.join(suggestions)}",
            status="ok",
            data={"tickers": suggestions, "saved": saved},
        )
        logger.info(f"[PIPELINE]   [scout] Suggested tickers: {', '.join(suggestions)}")

    total_ms = int((time.monotonic() - start) * 1000)

    # ── Format final memo with header ──
    formatted_memo = (
        "# MACRO STRATEGY MEMO (Auto-generated by Macro Scout)\n"
        "This memo provides big-picture macro context for your "
        "per-ticker analysis. Reference these themes and risks "
        "when making individual trading decisions.\n\n"
        f"{memo}\n\n"
        f"---\n"
        f"Scout runtime: {total_ms}ms | "
        f"Data: {len(data_snapshot):,} chars | "
        f"Memo: {len(memo):,} chars\n"
    )

    emit(
        "collecting",
        "macro_scout_done",
        f"Macro Strategy Scout complete: "
        f"{total_ms / 1000:.1f}s total, {len(memo):,} char memo, "
        f"{len(suggestions)} ticker suggestions",
        status="ok",
        data={
            "total_ms": total_ms,
            "memo_chars": len(memo),
            "suggestions": suggestions,
        },
        elapsed_ms=total_ms,
    )

    logger.info(f"[PIPELINE] \n{'=' * 60}")
    logger.info(
        f"[PIPELINE] MACRO SCOUT COMPLETE: {total_ms}ms ({total_ms / 1000:.1f}s)"
    )
    logger.info(f"[PIPELINE]   Memo: {len(memo):,} chars")
    logger.info(
        f"[PIPELINE]   Suggestions: {', '.join(suggestions) if suggestions else 'none'}"
    )
    logger.info(f"[PIPELINE] {'=' * 60}\n")

    return formatted_memo

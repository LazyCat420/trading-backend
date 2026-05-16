import logging
import csv
import json
from pathlib import Path

_EVOLVE_ROOT = Path(__file__).resolve().parent.parent.parent

import re
from datetime import datetime, timezone
from app.db.connection import get_db

from app.config.config_tickers import CRYPTO_TICKERS, COMMODITY_TICKERS
from app.utils.text_utils import (
    fmt_usd as _fmt_usd,
    truncate as _truncate,
    sanitize_ascii as _sanitize_text,
    is_scrape_artifact as _is_scrape_artifact,
    format_db_section as _section,
)
from app.utils.db_migrations import ensure_summary_columns

logger = logging.getLogger(__name__)

# _ensure_summary_columns, _section, _SCRAPE_ARTIFACT_PATTERNS, _is_scrape_artifact
# moved to shared utilities (app.utils.db_migrations and app.utils.text_utils)


def _analyze_freshness(latest_timestamp, max_age_days: int) -> str:
    """Analyze freshness and return a formatted metadata string."""
    if not latest_timestamp:
        return "[STALE: no date available]"

    try:
        if isinstance(latest_timestamp, str):
            # Parse YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
            if len(latest_timestamp) > 10:
                # Handle possible 'T' or timezone offsets roughly by taking first 19 chars
                latest_timestamp = latest_timestamp[:19].replace("T", " ")
                dt = datetime.strptime(latest_timestamp, "%Y-%m-%d %H:%M:%S")
            else:
                dt = datetime.strptime(latest_timestamp[:10], "%Y-%m-%d")
        elif isinstance(latest_timestamp, (int, float)):
            dt = datetime.fromtimestamp(latest_timestamp)
        else:
            # Assume datetime object
            dt = latest_timestamp

        # Use timezone-aware utc now for comparison
        now = datetime.now(timezone.utc)
        # Make dt timezone-aware if it isn't
        if dt.tzinfo is None:
            from datetime import timezone as _tz

            dt = dt.replace(tzinfo=_tz.utc)
        diff = now - dt

        age_hours = int(diff.total_seconds() / 3600)
        age_days = diff.days

        date_str = dt.strftime("%Y-%m-%d %H:%M UTC")

        if age_days > max_age_days:
            return f"[STALE: latest {age_days}d ago]"
        else:
            if age_hours < 24:
                return f"[latest: {date_str} | age: {age_hours}h]"
            else:
                return f"[latest: {date_str} | age: {age_days}d]"

    except Exception as e:
        return f"[STALE: invalid date format ({e})]"


def _build_portfolio_section(ticker: str) -> str:
    res = ""
    try:
        from app.trading.portfolio import get_current_state, get_recent_trades

        state = get_current_state()
        pos = next(
            (p for p in state.get("positions", []) if p["ticker"] == ticker), None
        )
        if pos:
            entry, curr = pos["avg_entry_price"], pos["current_price"]
            pnl = ((curr - entry) / entry * 100) if entry else 0
            pos_lines = [
                f"## 🚨 YOUR CURRENT POSITION: {ticker}",
                f"You currently own **{pos['qty']:.2f} shares** of {ticker}.",
                f"  - Avg Entry Price: ${entry:.2f}",
                f"  - Current Price: ${curr:.2f}",
                f"  - **Unrealized PnL: {pnl:+.1f}%**",
                "Keep your current position and stop-losses into account when providing advice!\n",
            ]
            res += "\n".join(pos_lines) + "\n"

        recent = [t for t in get_recent_trades(limit=50) if t["ticker"] == ticker]
        if recent:
            order_lines = [f"\n## Recent Trades ({ticker})"]
            for t in recent[:5]:
                pnl_str = (
                    f" (Realized PnL: ${t['realized_pnl']:+.2f})"
                    if t.get("realized_pnl")
                    else ""
                )
                order_lines.append(
                    f"  - {t['created_at'][:10]}: {t['side']} {t['qty']:.1f} shares @ ${t['price']:.2f}{pnl_str}"
                )
            res += "\n".join(order_lines) + "\n"
    except Exception as e:
        logger.debug("portfolio pos query failed: %s", e)
    return res


def _build_news_section(db, ticker: str, since: datetime | None = None) -> str:
    """Build news articles section. If `since` is set, only includes articles
    published after that timestamp (delta mode)."""
    if since:
        rows = db.execute(
            """
            SELECT title, publisher, published_at,
                   COALESCE(llm_summary, summary) AS best_summary
            FROM news_articles
            WHERE ticker = %s AND published_at > %s
              AND (quality_status IS NULL OR quality_status != 'discarded')
            ORDER BY published_at DESC
            LIMIT 15
        """,
            [ticker, since],
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT title, publisher, published_at,
                   COALESCE(llm_summary, summary) AS best_summary
            FROM news_articles
            WHERE ticker = %s
              AND (quality_status IS NULL OR quality_status != 'discarded')
            ORDER BY published_at DESC
            LIMIT 15
        """,
            [ticker],
        ).fetchall()

    news_formatted = []
    latest_ts = None
    for i, row in enumerate(rows):
        if i == 0:
            latest_ts = row[2]
        summary = row[3] or ""
        if _is_scrape_artifact(summary):
            summary = "[Summary unavailable]"
        else:
            summary = _truncate(summary, 300)
        news_formatted.append((row[0], row[1], row[2], summary))

    freshness = (
        (" " + _analyze_freshness(latest_ts, max_age_days=14)) if latest_ts else ""
    )
    delta_label = " [NEW SINCE LAST ANALYSIS]" if since else ""

    return _section(
        f"News Articles ({ticker}-specific){freshness}{delta_label}",
        news_formatted,
        ["Title", "Publisher", "Published", "Summary"],
        max_rows=15,
    )


def _build_general_news_section(db) -> str:
    rows = db.execute("""
        SELECT title, publisher, published_at,
               COALESCE(llm_summary, summary) AS best_summary
        FROM news_articles
        WHERE ticker IS NULL
          AND (quality_status IS NULL OR quality_status != 'discarded')
        ORDER BY published_at DESC
        LIMIT 10
    """).fetchall()
    general_news = []
    for row in rows:
        summary = row[3] or ""
        if _is_scrape_artifact(summary):
            summary = "[Summary unavailable]"
        else:
            summary = _truncate(summary, 300)
        general_news.append((row[0], row[1], row[2], summary))
    return _section(
        "General Market News",
        general_news,
        ["Title", "Publisher", "Published", "Summary"],
        max_rows=10,
    )


def _build_reddit_section(db, ticker: str, since: datetime | None = None) -> str:
    """Build Reddit posts section. If `since` is set, only includes posts
    created after that timestamp (delta mode)."""
    if since:
        rows = db.execute(
            """
            SELECT subreddit, title,
                   COALESCE(summary, body) AS content,
                   score, comment_count, created_utc
            FROM reddit_posts
            WHERE ticker = %s AND created_utc > %s
              AND (quality_status IS NULL OR quality_status != 'discarded')
            ORDER BY score DESC
            LIMIT 10
        """,
            [ticker, since],
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT subreddit, title,
                   COALESCE(summary, body) AS content,
                   score, comment_count, created_utc
            FROM reddit_posts
            WHERE ticker = %s
              AND (quality_status IS NULL OR quality_status != 'discarded')
            ORDER BY score DESC
            LIMIT 10
        """,
            [ticker],
        ).fetchall()

    reddit_formatted = []
    latest_ts = None
    if rows:
        latest = max(rows, key=lambda x: x[5] if x[5] is not None else 0)
        latest_ts = latest[5]

    for row in rows:
        reddit_formatted.append(
            (
                row[0],
                _truncate(row[1] or "", 150),
                _truncate(row[2] or "", 400),
                row[3],
                row[4],
                row[5],
            )
        )

    freshness = (
        (" " + _analyze_freshness(latest_ts, max_age_days=14)) if latest_ts else ""
    )
    delta_label = " [NEW SINCE LAST ANALYSIS]" if since else ""

    return _section(
        f"Reddit Sentiment ({ticker}){freshness}{delta_label}",
        reddit_formatted,
        ["Subreddit", "Title", "Content", "Score", "Comments", "Posted"],
        max_rows=10,
    )


def _build_congress_section(db, ticker: str, since: datetime | None = None) -> str:
    """Build Congress trades section. If `since` is set, only includes trades
    filed after that timestamp (delta mode)."""
    if since:
        rows = db.execute(
            """
            SELECT politician, party, ticker, transaction_type, amount_range, trade_date
            FROM congress_trades
            WHERE ticker = %s AND trade_date > %s
            ORDER BY trade_date DESC
            LIMIT 10
        """,
            [ticker, since],
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT politician, party, ticker, transaction_type, amount_range, trade_date
            FROM congress_trades
            WHERE ticker = %s
            ORDER BY trade_date DESC
            LIMIT 10
        """,
            [ticker],
        ).fetchall()

    freshness = ""
    if rows:
        freshness = " " + _analyze_freshness(rows[0][5], max_age_days=90)
    delta_label = " [NEW SINCE LAST ANALYSIS]" if since else ""

    return _section(
        f"Congress Trades ({ticker}){freshness}{delta_label}",
        rows,
        ["Politician", "Party", "Ticker", "Type", "Amount", "Date"],
    )


def _build_recent_congress_section(db) -> str:
    rows = db.execute("""
        SELECT politician, party, ticker, transaction_type, amount_range, trade_date
        FROM congress_trades
        ORDER BY trade_date DESC
        LIMIT 10
    """).fetchall()

    freshness = ""
    if rows:
        freshness = " " + _analyze_freshness(rows[0][5], max_age_days=30)

    return _section(
        f"Recent Congress Trades (All Tickers){freshness}",
        rows,
        ["Politician", "Party", "Ticker", "Type", "Amount", "Date"],
    )


def _build_institutional_holdings_section(db, ticker: str) -> str:
    rows = db.execute(
        """
        SELECT f.filer_name, h.shares, h.value_usd, h.is_new_position, h.is_exit, h.filing_quarter
        FROM sec_13f_holdings h
        LEFT JOIN sec_13f_filers f ON h.cik = f.cik
        WHERE h.ticker = %s
        ORDER BY value_usd DESC
        LIMIT 10
    """,
        [ticker],
    ).fetchall()
    if rows:
        lines = [f"\n## Institutional Holdings ({ticker})"]
        lines.append(f"({len(rows)} records, showing up to 20)")
        for r in rows:
            new_flag = " [NEW]" if r[3] else ""
            exit_flag = " [EXIT]" if r[4] else ""
            lines.append(
                f"  Filer: {r[0]} | Shares: {r[1]:,} | ValueUSD: {_fmt_usd(r[2])}{new_flag}{exit_flag} | Quarter: {r[5]}"
            )
        return "\n".join(lines) + "\n"
    return f"\n## Institutional Holdings ({ticker})\nNo data available.\n"


def _build_youtube_section(db, ticker: str) -> str:
    rows = db.execute(
        """
        SELECT title, channel,
               COALESCE(summary, raw_transcript) AS content,
               published_at, tickers_mentioned
        FROM youtube_transcripts
        WHERE ticker = %s
        ORDER BY published_at DESC
        LIMIT 3
    """,
        [ticker],
    ).fetchall()

    if len(rows) < 3:
        remaining = 3 - len(rows)
        extra = db.execute(
            """
            SELECT title, channel,
                   COALESCE(summary, raw_transcript) AS content,
                   published_at, tickers_mentioned
            FROM youtube_transcripts
            WHERE ticker IS NULL
            ORDER BY published_at DESC
            LIMIT %s
        """,
            [remaining],
        ).fetchall()
        rows = list(rows) + list(extra)

    yt_formatted = []
    latest_ts = None
    for i, row in enumerate(rows):
        if i == 0:
            latest_ts = row[3]
        content = row[2] or ""
        if len(content) > 600:
            words = content.split()
            if (
                words
                and sum(1 for w in words[:50] if len(w) <= 2) / min(len(words), 50)
                > 0.4
            ):
                content = re.sub(r"(?<=\w) (?=\w)", "", content)
        tickers = f" [Mentions: {row[4]}]" if row[4] else ""
        yt_formatted.append((row[0], row[1], _truncate(content, 600) + tickers, row[3]))

    freshness = (
        (" " + _analyze_freshness(latest_ts, max_age_days=14)) if latest_ts else ""
    )
    return _section(
        f"YouTube Analysis ({ticker}){freshness}",
        yt_formatted,
        ["Title", "Channel", "Summary", "Published"],
        max_rows=3,
    )


def _build_sec_10k_section(db, ticker: str) -> str:
    try:
        rows = db.execute(
            """
            SELECT extracted_tables
            FROM sec_10k_extractions
            WHERE ticker = %s
        """,
            [ticker],
        ).fetchone()
        if rows and rows[0]:
            return f"\n## SEC 10-K Financial Tables (Recent)\n{rows[0]}\n"
    except Exception:
        pass
    return ""


def _build_commodity_macro_section() -> str:
    try:
        from app.collectors.commodity_collector import format_commodity_context

        commodity_text = format_commodity_context()
        if commodity_text:
            return commodity_text
    except Exception as e:
        logger.warning(f"Commodity context failed: {e}")
        return "\n## Commodity Macro\n[MISSING] Data source unavailable.\n"
    return ""


def _build_congress_scanner_section() -> str:
    try:
        from app.processors.congress_scanner import find_consensus_trades

        consensus = find_consensus_trades(days=30, min_members=2)
        if consensus:
            intel_lines = ["\n## Congress Trading Signals"]
            for c in consensus[:5]:
                intel_lines.append(
                    f"  {c['ticker']}: {c['direction'].upper()} by {c['member_count']} members ({', '.join(c['members'][:3])})"
                )
            return "\n".join(intel_lines) + "\n"
    except Exception as e:
        logger.warning(f"Congress scanner failed: {e}")
    return ""


def _build_fund_scanner_section() -> str:
    try:
        from app.processors.fund_scanner import find_crossfund_consensus

        fund_consensus = find_crossfund_consensus(min_funds=3)
        if fund_consensus:
            fund_lines = ["\n## Institutional Consensus Signals"]
            for c in fund_consensus[:5]:
                val = f"${c['total_value']:,.0f}" if c.get("total_value") else ""
                fund_lines.append(
                    f"  {c['ticker']}: held by {c['fund_count']} funds ({val})"
                )
            return "\n".join(fund_lines) + "\n"
    except Exception as e:
        logger.warning(f"Fund scanner failed: {e}")
    return ""


def _build_crypto_price_section(db, ticker: str) -> str:
    if ticker.upper() in CRYPTO_TICKERS:
        rows = db.execute(
            """
            SELECT symbol, date, close, volume
            FROM asset_prices
            WHERE symbol = %s AND asset_class = 'crypto'
            ORDER BY date DESC
            LIMIT 30
        """,
            [ticker.upper()],
        ).fetchall()
        freshness = (
            (" " + _analyze_freshness(rows[0][1], max_age_days=5)) if rows else ""
        )
        return _section(
            f"Crypto Price History ({ticker}){freshness}",
            rows,
            ["Symbol", "Date", "Close", "Volume"],
            max_rows=30,
        )
    return ""


def _build_commodity_price_section(db, ticker: str) -> str:
    if ticker.upper() in COMMODITY_TICKERS:
        rows = db.execute(
            """
            SELECT symbol, date, open, high, low, close, volume
            FROM asset_prices
            WHERE symbol = %s AND asset_class = 'commodity'
            ORDER BY date DESC
            LIMIT 30
        """,
            [ticker.upper()],
        ).fetchall()
        freshness = (
            (" " + _analyze_freshness(rows[0][1], max_age_days=5)) if rows else ""
        )
        return _section(
            f"Commodity Price History ({ticker}){freshness}",
            rows,
            ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"],
            max_rows=30,
        )
    return ""


def _build_relationship_map_section(ticker: str) -> str:
    try:
        from app.graph.graph_queries import build_relationship_map

        rel_map = build_relationship_map(ticker)
        if rel_map:
            return rel_map
    except Exception as e:
        logger.debug("relationship map failed: %s", e)
    return ""


def _build_peer_section(ticker: str, watchlist: list[str]) -> str:
    try:
        from app.processors.peer_comparison_processor import build_comparison_context

        peer_section = build_comparison_context(ticker, watchlist or [])
        if peer_section and len(peer_section) > 50:
            logger.info(
                "[context] Peer comparison added (%d chars) for %s",
                len(peer_section),
                ticker,
            )
            return peer_section
    except Exception as e:
        logger.debug("peer comparison failed: %s", e)
    return ""


def _build_trader_notes_section(db, ticker: str) -> str:
    try:
        note_rows = db.execute(
            """
            SELECT feedback_type, content, sentiment, confidence,
                   created_at
            FROM user_feedback
            WHERE ticker = %s AND is_active = TRUE
              AND feedback_type NOT IN ('constraint')
            ORDER BY created_at DESC LIMIT 10
        """,
            [ticker],
        ).fetchall()
        if note_rows:
            note_lines = [f"\n## 🚨 TRADER GROUND TRUTH & INVESTIGATION SUGGESTIONS ({ticker})"]
            for nr in note_rows:
                ts = nr[4].strftime("%Y-%m-%d") if nr[4] else "?"
                sent = f" ({nr[2]})" if nr[2] else ""
                conf = f" confidence:{nr[3]}" if nr[3] else ""
                note_lines.append(
                    f'  [{ts}] {nr[0].upper()}{sent}{conf}: "{_truncate(nr[1], 300)}"'
                )
            return "\n".join(note_lines) + "\n"
    except Exception as e:
        logger.debug("user_feedback notes query failed: %s", e)
    return ""


def _build_trader_constraints_section(db, ticker: str) -> str:
    try:
        constraint_rows = db.execute(
            """
            SELECT constraint_type, constraint_val, content, ticker,
                   expires_at
            FROM user_feedback
            WHERE ticker = %s AND is_active = TRUE
              AND feedback_type = 'constraint'
            ORDER BY created_at DESC LIMIT 10
        """,
            [ticker],
        ).fetchall()
        if constraint_rows:
            cl = ["\n## TRADER CONSTRAINTS (MUST OBEY)"]
            cl.append(
                "The following constraints were set by the human "
                "trader. You MUST respect these regardless of "
                "your analysis."
            )
            for cr in constraint_rows:
                exp = f" (expires {cr[4].strftime('%Y-%m-%d')})" if cr[4] else ""
                reason = f' -- Trader says: "{cr[2]}"' if cr[2] else ""
                cl.append(
                    f"  - {cr[3]}: {(cr[0] or '?').upper()} = {cr[1]}{reason}{exp}"
                )
            return "\n".join(cl) + "\n"
    except Exception as e:
        logger.debug("user_feedback constraints query failed: %s", e)
    return ""


def _build_filtered_data_section(ticker: str) -> str:
    try:
        from app.services.data_flag_service import get_filtered_report

        report = get_filtered_report(ticker)
        if report and report.get("flagged_items"):
            fl = [f"\n## Filtered Data ({ticker})"]
            fl.append(
                f"{len(report['flagged_items'])} items flagged by trader (excluded from analysis):"
            )
            for item in report["flagged_items"][:5]:
                fl.append(
                    f"  - [{item.get('flag_type', '?')}] {item.get('source_table', '?')}/{item.get('source_id', '?')}: {item.get('reason', 'no reason')}"
                )
            return "\n".join(fl) + "\n"
    except Exception as e:
        logger.debug("filtered report failed: %s", e)
    return ""


def _build_rag_section(ticker: str) -> str:
    try:
        from app.db.vector_store import vector_store

        stats = vector_store.get_stats()
        if stats["total_embeddings"] > 0:
            from app.services.rag_ab_test import rag_retrieve, format_rag_context

            query = f"{ticker} stock market analysis trading"
            chunks, strategy = rag_retrieve(ticker, query)
            if chunks:
                rag_section = format_rag_context(chunks)
                logger.info(
                    f"[context] RAG added {len(chunks)} chunks via {strategy.value}"
                )
                return rag_section
        else:
            logger.debug("[context] No embeddings in DB, skipping RAG section")
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
    return ""


def _get_evolve_signal(ticker: str) -> str | None:
    """Load best_strategy.py signal + evolve_signal.json hypothesis.

    Returns a formatted context section string, or None if unavailable.
    """
    try:
        from app.constants import EVOLVE_SIGNAL_ENABLED

        if not EVOLVE_SIGNAL_ENABLED:
            return None
    except (ImportError, AttributeError):
        return None

    lines: list[str] = []

    # --- best_strategy.py signal ---
    best_strategy_path = _EVOLVE_ROOT / "best_strategy.py"
    if best_strategy_path.exists():
        try:
            import importlib.util
            import pandas as pd

            # Load OHLCV for this ticker from DB
            with get_db() as db:
                rows = db.execute(
                    "SELECT date, open, high, low, close, volume "
                    "FROM price_history WHERE ticker = %s ORDER BY date",
                    [ticker],
                ).fetchall()
                if rows:
                    ohlcv = pd.DataFrame(
                        rows, columns=["date", "open", "high", "low", "close", "volume"]
                    )
                    spec = importlib.util.spec_from_file_location(
                        "best_strategy", str(best_strategy_path)
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    signals = mod.generate_signals(ohlcv)
                    last_signal = int(signals.iloc[-1]) if len(signals) > 0 else 0
                    signal_label = {
                        1: "BUY (1)",
                        -1: "SELL (-1)",
                        0: "NEUTRAL (0)",
                    }.get(last_signal, f"UNKNOWN ({last_signal})")
                    lines.append(f"Current signal: {signal_label}")
        except Exception as e:
            logger.debug("[context] best_strategy signal failed for %s: %s", ticker, e)

    # --- results.tsv last KEEP Sharpe ---
    results_tsv = _EVOLVE_ROOT / "results.tsv"
    if results_tsv.exists():
        try:
            with open(results_tsv, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                last_keep = None
                for row in reader:
                    if row.get("status") == "KEEP":
                        last_keep = row
            if last_keep and last_keep.get("score"):
                lines.append(f"Sharpe from last KEEP: {last_keep['score']}")
        except Exception:
            pass

    # --- evolve_signal.json hypothesis ---
    signal_json = _EVOLVE_ROOT / "evolve_signal.json"
    if signal_json.exists():
        try:
            data = json.loads(signal_json.read_text(encoding="utf-8"))
            hyp = data.get("winning_hypothesis")
            if hyp:
                lines.append(f"Winning hypothesis: {hyp}")
        except Exception:
            pass

    if not lines:
        return None

    header = "\n## Evolve Signal (ASI-Evolve best_strategy.py)"
    return header + "\n" + "\n".join(f"  {l}" for l in lines) + "\n"


def _build_supplemental_analysis_section(db, ticker: str) -> str:
    try:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        row = db.execute(
            """
            SELECT result_json, confidence, created_at, cycle_id
            FROM analysis_results
            WHERE ticker = %s AND cycle_id = 'manual_run_analysis' AND created_at > %s
            ORDER BY created_at DESC LIMIT 1
            """,
            [ticker, cutoff]
        ).fetchone()
        
        if row:
            result_json_str, conf, created_at, cycle_id = row
            try:
                res_dict = json.loads(result_json_str) if result_json_str else {}
            except Exception:
                res_dict = {}
                
            action = res_dict.get("action", "UNKNOWN")
            rationale = res_dict.get("rationale", "")
            
            if rationale:
                # Truncate to 1000 chars to save tokens
                rationale_trunc = _truncate(rationale, 1000)
                date_str = created_at.strftime("%Y-%m-%d %H:%M UTC") if created_at else "?"
                
                lines = [
                    "\n## PAST DEEP ANALYSIS (SUPPLEMENTAL DATA)",
                    f"A prior deep analysis was run for {ticker} on {date_str}.",
                    f"  - Action: {action}",
                    f"  - Confidence: {conf}%",
                    f"  - Rationale (truncated): {rationale_trunc}",
                    ""
                ]
                return "\n".join(lines)
    except Exception as e:
        logger.debug("supplemental analysis query failed: %s", e)
    return ""


# _ensure_summary_columns, _section, _is_scrape_artifact moved to shared utilities
# (app.utils.db_migrations and app.utils.text_utils)


# _SCRAPE_ARTIFACT_PATTERNS and _is_scrape_artifact moved to app.utils.text_utils


async def build_context_blob(
    ticker: str,
    watchlist: list[str] | None = None,
    since: datetime | None = None,
) -> str:
    """Build a complete context blob from DB data for a given ticker.

    Args:
        ticker: Stock symbol to build context for.
        watchlist: Optional watchlist for peer comparison.
        since: If set, content sections (news, reddit, youtube, congress) only
               include data published AFTER this timestamp. Structural data
               (portfolio, fundamentals, price, 10-K) is always included.
               This enables progressive summarization — the thesis carries
               forward all prior understanding, so we only need new raw data.
    """
    from app.pipeline.analysis.grounded_context import GroundedContext

    if since:
        logger.info(
            "[CONTEXT] Delta mode for %s — only pulling data since %s",
            ticker,
            since.strftime("%Y-%m-%d %H:%M UTC"),
        )

    with get_db() as db:
        ensure_summary_columns()
        sections = []

        sections.append(f"# Market Data Report: {ticker}")
        if since:
            sections.append(
                f"**DELTA MODE**: Only showing NEW data since {since.strftime('%Y-%m-%d %H:%M UTC')}. "
                "Prior data is summarized in the CURRENT THESIS section above.\n"
            )
        else:
            sections.append(
                "Generated from database. All data is real, collected via automated pipeline.\n"
            )

        # B3: Inject warm-start brief
        from app.pipeline.orchestration.cycle_brief_builder import (
            build_warm_start_brief,
        )

        brief = build_warm_start_brief(ticker)
        if brief:
            sections.append(brief)

        # ── Structural data (always included — changes by nature) ──
        sections.append(_build_portfolio_section(ticker))

        try:
            ctx = await GroundedContext.build(ticker)
            sections.append(ctx.to_prompt())
        except Exception as e:
            logger.warning(f"Failed to build GroundedContext for {ticker}: {e}")

        # ── Content data (delta-filtered when thesis exists) ──
        sections.append(_build_news_section(db, ticker, since=since))
        sections.append(_build_general_news_section(db))
        sections.append(_build_reddit_section(db, ticker, since=since))
        sections.append(_build_congress_section(db, ticker, since=since))
        sections.append(_build_recent_congress_section(db))
        sections.append(_build_institutional_holdings_section(db, ticker))
        sections.append(_build_youtube_section(db, ticker))
        sections.append(_build_sec_10k_section(db, ticker))
        sections.append(_build_commodity_macro_section())
        sections.append(_build_congress_scanner_section())
        sections.append(_build_fund_scanner_section())
        sections.append(_build_crypto_price_section(db, ticker))
        sections.append(_build_commodity_price_section(db, ticker))
        sections.append(_build_relationship_map_section(ticker))
        sections.append(_build_peer_section(ticker, watchlist))
        sections.append(_build_trader_notes_section(db, ticker))
        sections.append(_build_trader_constraints_section(db, ticker))
        sections.append(_build_rag_section(ticker))
        sections.append(_build_supplemental_analysis_section(db, ticker))

        # Compile and remove empty lines that might have been appending empty sections
        blob = "\n".join(filter(None, sections))
        blob += f"\n---\nTotal context length: {len(blob):,} characters\n"
        return _sanitize_text(blob)


def build_general_context() -> str:
    with get_db() as db:
        sections = []

        sections.append("# General Market Intelligence Report")
        sections.append(
            "Generated from database. All data is real, collected via automated pipeline.\n"
        )

        try:
            rows = db.execute(
                "SELECT ticker, status, added_at FROM watchlist "
                "WHERE status = 'active' ORDER BY added_at DESC LIMIT 20"
            ).fetchall()
            if rows:
                wl_lines = ["\n## Active Watchlist"]
                wl_lines.append(f"({len(rows)} tickers)")
                for r in rows:
                    wl_lines.append(f"  {r[0]} (status: {r[1]})")
                sections.append("\n".join(wl_lines) + "\n")
        except Exception as e:
            logger.debug("watchlist query failed: %s", e)

        try:
            from app.trading.portfolio import get_current_state, get_recent_trades

            state = get_current_state()
            lines = [
                "\n## Portfolio State",
                f"  Cash: ${state['cash']:,.2f}",
                f"  Total Value: ${state['total_value']:,.2f}",
                f"  Total PnL: ${state['total_pnl']:,.2f}",
                f"  Open Positions: {state['position_count']}",
            ]
            if state["positions"]:
                lines.append("\n  **Current Open Positions:**")
                lines.append(
                    "  | Ticker | Quantity | Avg Entry | Current Price | Unrealized PnL % |"
                )
                lines.append(
                    "  |--------|----------|-----------|---------------|------------------|"
                )
                for p in state["positions"]:
                    entry, curr = p["avg_entry_price"], p["current_price"]
                    pnl = ((curr - entry) / entry * 100) if entry else 0
                    lines.append(
                        f"  | {p['ticker']} | {p['qty']:.2f} | ${entry:.2f} | ${curr:.2f} | {pnl:+.2f}% |"
                    )

            recent = get_recent_trades(limit=5)
            if recent:
                lines.append("\n  **Recent Executed Trades:**")
                for t in recent:
                    pnl_str = (
                        f" (PnL: ${t['realized_pnl']:+.2f})"
                        if t.get("realized_pnl")
                        else ""
                    )
                    lines.append(
                        f"  - {t['created_at'][:10]}: {t['side']} {t['qty']:.2f} shares of {t['ticker']} @ ${t['price']:.2f}{pnl_str}"
                    )

            sections.append("\n".join(lines) + "\n")
        except Exception as e:
            logger.debug("portfolio query failed: %s", e)

        try:
            rows = db.execute(
                "SELECT title, publisher, published_at, summary, ticker "
                "FROM news_articles "
                "ORDER BY published_at DESC LIMIT 20"
            ).fetchall()
            news_lines = ["\n## Recent Market News"]
            news_lines.append(
                "WARNING: The following articles are for general market awareness only. DO NOT assume the user holds these tickers unless they are explicitly listed in the 'Portfolio State' section above."
            )
            news_lines.append(f"({len(rows)} articles)")
            for r in rows:
                tkr = f" [{r[4]}]" if r[4] else ""
                summary = _truncate(r[3] or "", 150)
                if _is_scrape_artifact(summary):
                    summary = "[Summary unavailable]"
                news_lines.append(f"  {r[0]}{tkr} — {r[1]} ({r[2]})\n    {summary}")
            sections.append("\n".join(news_lines) + "\n")
        except Exception as e:
            logger.debug("news query failed: %s", e)

        try:
            rows = db.execute(
                "SELECT subreddit, title, body, score, comment_count, ticker, created_utc "
                "FROM reddit_posts ORDER BY score DESC LIMIT 15"
            ).fetchall()
            reddit_lines = ["\n## Top Reddit Posts (All Subreddits)"]
            reddit_lines.append(
                "WARNING: The following posts are for general market awareness only. DO NOT assume the user holds these tickers unless they are explicitly listed in the 'Portfolio State' section above."
            )
            reddit_lines.append(f"({len(rows)} posts)")
            for r in rows:
                tkr = f" [{r[5]}]" if r[5] else ""
                body = _truncate(r[2] or "", 200)
                reddit_lines.append(
                    f"  r/{r[0]}: {r[1]}{tkr} (score: {r[3]}, comments: {r[4]})\n    {body}"
                )
            sections.append("\n".join(reddit_lines) + "\n")
        except Exception as e:
            logger.debug("reddit query failed: %s", e)

        try:
            rows = db.execute(
                "SELECT title, channel, raw_transcript, published_at, ticker "
                "FROM youtube_transcripts ORDER BY published_at DESC LIMIT 5"
            ).fetchall()
            yt_lines = ["\n## Recent YouTube Analysis"]
            yt_lines.append(f"({len(rows)} transcripts)")
            for r in rows:
                tkr = f" [{r[4]}]" if r[4] else ""
                transcript = _truncate(r[2] or "", 400)
                yt_lines.append(f"  {r[0]}{tkr} — {r[1]} ({r[3]})\n    {transcript}")
            sections.append("\n".join(yt_lines) + "\n")
        except Exception as e:
            logger.debug("youtube query failed: %s", e)

        sections.append(_build_recent_congress_section(db))

        try:
            counts = {}
            for tbl in [
                "news_articles",
                "reddit_posts",
                "youtube_transcripts",
                "congress_trades",
                "sec_13f_holdings",
            ]:
                cnt = db.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
                counts[tbl] = cnt[0] if cnt else 0
            count_lines = ["\n## Database Summary"]
            for tbl, cnt in counts.items():
                count_lines.append(f"  {tbl}: {cnt:,} records")
            sections.append("\n".join(count_lines) + "\n")
        except Exception as e:
            logger.debug("count query failed: %s", e)

        blob = "\n".join(filter(None, sections))
        blob += f"\n---\nTotal context length: {len(blob):,} characters\n"
        return _sanitize_text(blob)

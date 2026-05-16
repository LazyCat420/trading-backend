"""
RLM Trading Tools -- Functions injected into the RLM REPL environment.

These are called BY THE MODEL inside ```repl``` blocks. Instead of parsing
18K chars of text, the model calls get_technicals("NVDA") and gets a clean dict.

All functions query the PostgreSQL database directly and return dicts/lists.
They're added to the REPL globals via RLM's custom_tools parameter.

CRITICAL FIX (2026-03-25): RLM's LocalREPL uses exec() to run code blocks.
exec() only captures print() output -- bare expressions like
`get_technicals("WFC")` silently discard the return value.
This caused tools to "fail" on the first call (empty stdout) and waste
2+ iterations before the LLM fell back to regex-parsing the raw context.
Solution: _auto_print_wrapper wraps each tool so return values are
auto-printed to stdout, making them visible in the REPL output.
"""

import functools
import json as _json

from app.db.connection import get_db


def _auto_print_wrapper(fn):
    """Wrap a tool function so its return value is auto-printed to stdout.

    The RLM LocalREPL uses exec() which only captures print() output.
    Bare expressions like `get_technicals("WFC")` return a dict but
    exec() silently discards it. This wrapper ensures the return value
    is always printed, making it visible in the REPL output fed back
    to the LLM.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        # Pretty-print dicts/lists as JSON for readability
        if isinstance(result, (dict, list)):
            print(_json.dumps(result, indent=2, default=str))
        else:
            print(result)
        return result

    return wrapper


def get_price(ticker: str, days: int = 30) -> list[dict]:
    """Get recent OHLCV price history for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. "NVDA")
        days: Number of trading days to return (default 30)

    Returns:
        List of dicts with date, open, high, low, close, volume
    """
    with get_db() as db:
        rows = db.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT %s
        """,
            [ticker.upper(), days],
        ).fetchall()
        return [
            {
                "date": str(r[0]),
                "open": round(r[1], 2),
                "high": round(r[2], 2),
                "low": round(r[3], 2),
                "close": round(r[4], 2),
                "volume": int(r[5]),
            }
            for r in rows
        ]


def get_technicals(ticker: str) -> dict:
    """Get latest technical indicators for a ticker.

    Returns dict with: rsi_14, macd, macd_signal, sma_20, sma_50, sma_200,
    bb_upper, bb_lower, atr_14, adx_14, stoch_k, stoch_d, support, resistance
    """
    with get_db() as db:
        row = db.execute(
            """
            SELECT date, rsi_14, macd, macd_signal, macd_hist,
                   sma_20, sma_50, sma_200, bb_upper, bb_lower,
                   atr_14, adx_14, stoch_k, stoch_d, support, resistance
            FROM technicals
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT 1
        """,
            [ticker.upper()],
        ).fetchone()
        if not row:
            return {"error": f"No technicals for {ticker}"}
        return {
            "date": str(row[0]),
            "rsi_14": round(row[1] or 0, 2),
            "macd": round(row[2] or 0, 4),
            "macd_signal": round(row[3] or 0, 4),
            "macd_hist": round(row[4] or 0, 4),
            "sma_20": round(row[5] or 0, 2),
            "sma_50": round(row[6] or 0, 2),
            "sma_200": round(row[7] or 0, 2),
            "bb_upper": round(row[8] or 0, 2),
            "bb_lower": round(row[9] or 0, 2),
            "atr_14": round(row[10] or 0, 2),
            "adx_14": round(row[11] or 0, 2),
            "stoch_k": round(row[12] or 0, 2),
            "stoch_d": round(row[13] or 0, 2),
            "support": round(row[14] or 0, 2),
            "resistance": round(row[15] or 0, 2),
        }


def get_fundamentals(ticker: str) -> dict:
    """Get latest fundamental data for a ticker.

    Returns dict with: market_cap, pe_ratio, forward_pe, peg_ratio,
    profit_margin, roe, revenue, revenue_growth, debt_to_equity, beta, etc.
    """
    with get_db() as db:
        row = db.execute(
            """
            SELECT snapshot_date, market_cap, pe_ratio, forward_pe, peg_ratio,
                   price_to_book, profit_margin, roe, revenue, revenue_growth,
                   debt_to_equity, beta, week_52_high, week_52_low, short_float_pct
            FROM fundamentals
            WHERE ticker = %s
            ORDER BY snapshot_date DESC
            LIMIT 1
        """,
            [ticker.upper()],
        ).fetchone()
        if not row:
            return {"error": f"No fundamentals for {ticker}"}
        return {
            "date": str(row[0]),
            "market_cap": row[1],
            "pe_ratio": round(row[2] or 0, 2),
            "forward_pe": round(row[3] or 0, 2),
            "peg_ratio": round(row[4] or 0, 2),
            "price_to_book": round(row[5] or 0, 2),
            "profit_margin": round(row[6] or 0, 4),
            "roe": round(row[7] or 0, 4),
            "revenue": row[8],
            "revenue_growth": round(row[9] or 0, 4),
            "debt_to_equity": round(row[10] or 0, 2),
            "beta": round(row[11] or 0, 2),
            "week_52_high": round(row[12] or 0, 2),
            "week_52_low": round(row[13] or 0, 2),
            "short_float_pct": round(row[14] or 0, 4),
        }


def get_sentiment(ticker: str) -> dict:
    """Get sentiment summary for a ticker from Reddit and News.

    Returns dict with: reddit_post_count, avg_reddit_score, news_count,
    top_reddit_titles, recent_headlines
    """
    with get_db() as db:
        # Reddit stats
        reddit = db.execute(
            """
            SELECT COUNT(*), AVG(score), AVG(upvote_ratio)
            FROM reddit_posts WHERE ticker = %s
        """,
            [ticker.upper()],
        ).fetchone()
        # Top reddit posts by score
        top_reddit = db.execute(
            """
            SELECT title, score, subreddit FROM reddit_posts
            WHERE ticker = %s ORDER BY score DESC LIMIT 5
        """,
            [ticker.upper()],
        ).fetchall()
        # News count and headlines
        news = db.execute(
            """
            SELECT COUNT(*) FROM news_articles WHERE ticker = %s
        """,
            [ticker.upper()],
        ).fetchone()
        headlines = db.execute(
            """
            SELECT title, publisher, published_at FROM news_articles
            WHERE ticker = %s ORDER BY published_at DESC LIMIT 5
        """,
            [ticker.upper()],
        ).fetchall()

        return {
            "reddit_posts": reddit[0] or 0,
            "avg_reddit_score": round(reddit[1] or 0, 1),
            "avg_upvote_ratio": round(reddit[2] or 0, 3),
            "news_articles": news[0] or 0,
            "top_reddit": [
                {"title": r[0][:100], "score": r[1], "sub": r[2]} for r in top_reddit
            ],
            "recent_headlines": [
                {"title": h[0][:100], "publisher": h[1], "date": str(h[2])}
                for h in headlines
            ],
        }


def get_congress(ticker: str) -> list[dict]:
    """Get recent congress trades for a ticker.

    Returns list of dicts with: politician, party, type, amount, date
    """
    with get_db() as db:
        rows = db.execute(
            """
            SELECT politician, party, chamber, transaction_type, amount_range, trade_date
            FROM congress_trades
            WHERE ticker = %s
            ORDER BY trade_date DESC
            LIMIT 10
        """,
            [ticker.upper()],
        ).fetchall()
        return [
            {
                "politician": r[0],
                "party": r[1],
                "chamber": r[2],
                "type": r[3],
                "amount": r[4],
                "date": str(r[5]),
            }
            for r in rows
        ]


def search_news(keyword: str, limit: int = 10) -> list[dict]:
    """Search news articles by keyword in title or summary.

    Args:
        keyword: Search term (case-insensitive)
        limit: Max results (default 10)

    Returns:
        List of dicts with: title, publisher, date, summary_preview
    """
    with get_db() as db:
        rows = db.execute(
            """
            SELECT title, publisher, published_at, summary
            FROM news_articles
            WHERE LOWER(title) LIKE %s OR LOWER(summary) LIKE %s
            ORDER BY published_at DESC
            LIMIT %s
        """,
            [f"%{keyword.lower()}%", f"%{keyword.lower()}%", limit],
        ).fetchall()
        return [
            {
                "title": r[0][:150],
                "publisher": r[1],
                "date": str(r[2]),
                "summary": (r[3] or "")[:200],
            }
            for r in rows
        ]


def get_institutional(ticker: str) -> list[dict]:
    """Get SEC 13F institutional holdings for a ticker.

    Returns list of dicts with: filer, shares, value_usd, pct_change, quarter
    """
    with get_db() as db:
        rows = db.execute(
            """
            SELECT f.filer_name, h.shares, h.value_usd, h.pct_change, h.is_new_position,
                   h.is_exit, h.filing_quarter
            FROM sec_13f_holdings h
            LEFT JOIN sec_13f_filers f ON h.cik = f.cik
            WHERE h.ticker = %s
            ORDER BY h.filing_quarter DESC
            LIMIT 10
        """,
            [ticker.upper()],
        ).fetchall()
        return [
            {
                "filer": r[0],
                "shares": r[1],
                "value_usd": r[2],
                "pct_change": r[3],
                "new_position": r[4],
                "exit": r[5],
                "quarter": r[6],
            }
            for r in rows
        ]


def get_latest_price(ticker: str) -> dict:
    """Get the single most recent price for a ticker.

    Returns dict with: date, close, volume, change_pct (vs prior day)
    """
    with get_db() as db:
        rows = db.execute(
            """
            SELECT date, close, volume FROM price_history
            WHERE ticker = %s ORDER BY date DESC LIMIT 2
        """,
            [ticker.upper()],
        ).fetchall()
        if not rows:
            return {"error": f"No price data for {ticker}"}
        today = rows[0]
        result = {
            "date": str(today[0]),
            "close": round(today[1], 2),
            "volume": int(today[2]),
        }
        if len(rows) > 1:
            prev = rows[1][1]
            result["change_pct"] = (
                round((today[1] - prev) / prev * 100, 2) if prev else 0
            )
        return result


def FINAL(decision: dict) -> str:
    """Submit the final trading decision.

    Args:
        decision: A dict with action, confidence, and rationale.
    """
    print(
        f"\n[SYSTEM: Final decision accepted: {_json.dumps(decision, default=str)}]\n"
    )
    return "[SYSTEM: Final decision accepted. You may stop processing now.]"


def add_reminder(
    ticker: str, condition: str, action: str, expires_in_days: int = 7
) -> str:
    """Add a prospective memory reminder for future trading cycles.

    Args:
        ticker: The stock ticker symbol.
        condition: Natural language or Python eval condition (e.g., 'price crosses $150').
        action: The intended action or note ('Recheck resistance levels', 'Buy 100 shares').
        expires_in_days: How many days until this reminder expires.
    """
    from app.services.memory.prospective_memory import prospective_memory_store

    try:
        reminder_id = prospective_memory_store.add_reminder(
            ticker=ticker.upper(),
            condition=condition,
            intended_action=action,
            expires_in_days=expires_in_days,
        )
        msg = f"[SYSTEM: Prospective reminder added successfully. ID: {reminder_id}]"
        print(msg)
        return msg
    except Exception as e:
        err = f"Failed to add reminder: {e}"
        print(err)
        return err


# ─── Build the custom_tools dict for RLM injection ───
TRADING_TOOLS = {
    "get_price": {
        "tool": get_price,
        "description": "get_price(ticker, days=30) -> list[dict] — OHLCV price history",
    },
    "get_technicals": {
        "tool": get_technicals,
        "description": "get_technicals(ticker) -> dict — RSI, MACD, SMA, BB, ATR, ADX, Stochastic, support/resistance",
    },
    "get_fundamentals": {
        "tool": get_fundamentals,
        "description": "get_fundamentals(ticker) -> dict — PE, revenue, margins, beta, 52w range",
    },
    "get_sentiment": {
        "tool": get_sentiment,
        "description": "get_sentiment(ticker) -> dict — Reddit posts, scores, news headlines",
    },
    "get_congress": {
        "tool": get_congress,
        "description": "get_congress(ticker) -> list[dict] — Recent congress trades for ticker",
    },
    "search_news": {
        "tool": search_news,
        "description": "search_news(keyword, limit=10) -> list[dict] — Search news by keyword",
    },
    "get_institutional": {
        "tool": get_institutional,
        "description": "get_institutional(ticker) -> list[dict] — SEC 13F institutional holdings",
    },
    "get_latest_price": {
        "tool": get_latest_price,
        "description": "get_latest_price(ticker) -> dict -- Latest price, volume, daily change %",
    },
    "FINAL": {
        "tool": FINAL,
        "description": "FINAL(decision_dict) -> Submit the completed JSON decision. Run this to finalize your answer.",
    },
    "add_reminder": {
        "tool": add_reminder,
        "description": "add_reminder(ticker, condition, action, expires_in_days=7) -> Log a reminder/alert to trigger in future cycles.",
    },
}

# ── Add quant tools to REPL ──
from app.processors.quant_processor import (
    get_zscore,
    get_sharpe,
    get_sortino,
    get_risk_reward,
    get_drawdown,
    get_ticker_score,
    get_beta,
)
from app.processors.market_regime import get_market_regime

TRADING_TOOLS.update(
    {
        "get_zscore": {
            "tool": get_zscore,
            "description": "get_zscore(ticker, window=60) -> dict -- Z-score (how many std devs from mean). |Z|>2 = extreme move",
        },
        "get_sharpe": {
            "tool": get_sharpe,
            "description": "get_sharpe(ticker, days=252) -> dict -- Sharpe ratio (risk-adjusted return). >1 good, >2 excellent",
        },
        "get_sortino": {
            "tool": get_sortino,
            "description": "get_sortino(ticker, days=252) -> dict -- Sortino ratio (downside-only risk). Better than Sharpe for buy signals",
        },
        "get_risk_reward": {
            "tool": get_risk_reward,
            "description": "get_risk_reward(ticker) -> dict -- Risk/Reward ratio with ATR-based stop loss. R:R >= 2 ideal",
        },
        "get_drawdown": {
            "tool": get_drawdown,
            "description": "get_drawdown(ticker, days=252) -> dict -- Max drawdown + current drawdown from peak",
        },
        "get_ticker_score": {
            "tool": get_ticker_score,
            "description": "get_ticker_score(ticker) -> dict -- Composite score 0-100 from all signals (technical+fundamental+quant)",
        },
        "get_beta": {
            "tool": get_beta,
            "description": "get_beta(ticker) -> dict -- Beta vs SPY. >1 = more volatile than market",
        },
        "get_market_regime": {
            "tool": get_market_regime,
            "description": "get_market_regime() -> dict -- Current market regime (BULL/BEAR/SIDEWAYS) from SPY trend + VIX",
        },
    }
)

# ── Add graph learning tool ──
from app.cognition.ontology.graph_learn_tool import graph_learn

TRADING_TOOLS["graph_learn"] = {
    "tool": graph_learn,
    "description": (
        "graph_learn(nodes=[], edges=[]) -> Record associations you discover. "
        "nodes: [{id, type, label}] (type: Claim|Signal|Hypothesis|Theme|Event|Risk). "
        "edges: [{source, target, relation, weight, reason}] "
        "(relation: SUPPORTS|CONTRADICTS|CAUSES|IMPACTS|EXPOSED_TO|CORRELATES_WITH). "
        "Example: graph_learn(edges=[{source:'NVDA', target:'AMD', relation:'CORRELATES_WITH', weight:0.8, reason:'AI capex'}])"
    ),
}

# ── Apply auto-print wrapper to ALL tools ──
# This ensures return values appear in stdout even when the LLM calls
# tools as bare expressions without print(). Fixes the "silent first call" bug.
for _name, _entry in TRADING_TOOLS.items():
    if callable(_entry.get("tool")):
        _entry["tool"] = _auto_print_wrapper(_entry["tool"])

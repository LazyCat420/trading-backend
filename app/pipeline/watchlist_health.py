"""
Watchlist Health Scoring Engine — tracks data quality + analysis quality
per ticker and computes a 0-100 health score.

Scoring formula (0-100):
  Data Richness   25%  — news + reddit + youtube article counts
  Data Freshness  15%  — penalty for zero_news_streaks
  Collection Rel. 10%  — penalty for collection failures
  Confidence      20%  — avg confidence from decision engine
  Action Signals  15%  — BUY/SELL show tradability; HOLD-only = low
  Tenure Bonus    15%  — grace period for new tickers

Tiers:
  strong   80-100   High data density, actionable signals
  healthy  60-79    Decent data, some signals
  weak     30-59    Low data, mostly HOLDs
  critical  0-29    Zero data, repeated failures → purge candidate
  new       n/a     Within grace period (< WATCHLIST_GRACE_CYCLES)
"""

import logging
from datetime import datetime, timezone

from app.config import settings
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def _ensure_row(ticker: str) -> None:
    """Ensure a ticker_health row exists (INSERT OR IGNORE)."""
    with get_db() as db:
        try:
            db.execute(
                "INSERT INTO ticker_health (ticker, first_seen_at, updated_at) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (ticker) DO NOTHING",
                (ticker, datetime.now(timezone.utc), datetime.now(timezone.utc)),
            )
        except Exception as e:
            logger.warning(
                "[PIPELINE] ticker_health: ensure_row %s failed: %s", ticker, e
            )


def update_signals_from_collection(ticker: str, counts: dict) -> None:
    """Update health signals after per-ticker data collection.

    Called inside data_phase.py after each ticker's collectors finish.

    Args:
        ticker: Ticker symbol
        counts: Dict with keys:
            news     (int) — articles collected this cycle
            reddit   (int) — reddit posts collected this cycle
            youtube  (int) — youtube transcripts collected this cycle
            yfinance_ok (bool) — did yfinance succeed%s
    """
    ticker = ticker.upper().strip()
    _ensure_row(ticker)
    with get_db() as db:
        now = datetime.now(timezone.utc)

        news = counts.get("news", 0) or 0
        reddit = counts.get("reddit", 0) or 0
        youtube = counts.get("youtube", 0) or 0
        yf_ok = counts.get("yfinance_ok", True)

        try:
            db.execute(
                """
                UPDATE ticker_health SET
                    total_cycles = total_cycles + 1,
                    news_article_count = news_article_count + %s,
                    reddit_post_count = reddit_post_count + %s,
                    youtube_count = youtube_count + %s,
                    zero_news_streak = CASE WHEN %s = 0 THEN zero_news_streak + 1 ELSE 0 END,
                    collection_failures = collection_failures + CASE WHEN %s THEN 0 ELSE 1 END,
                    updated_at = %s
                WHERE ticker = %s
            """,
                (news, reddit, youtube, news, yf_ok, now, ticker),
            )
        except Exception as e:
            logger.warning(
                "[PIPELINE] ticker_health: update_collection %s failed: %s", ticker, e
            )


def update_signals_from_analysis(ticker: str, result: dict) -> None:
    """Update health signals after decision engine analysis.

    Called inside decision_engine.py after each ticker's analysis finishes.

    Args:
        ticker: Ticker symbol
        result: Dict with keys:
            action     (str) — BUY | SELL | HOLD
            confidence (int) — 0-100
    """
    ticker = ticker.upper().strip()
    _ensure_row(ticker)
    with get_db() as db:
        now = datetime.now(timezone.utc)

        action = result.get("action", "HOLD").upper()
        confidence = result.get("confidence", 0) or 0

        try:
            # Read current values to compute rolling average
            row = db.execute(
                "SELECT total_analyses, avg_confidence FROM ticker_health WHERE ticker = %s",
                (ticker,),
            ).fetchone()

            if row:
                n = row[0] or 0
                old_avg = row[1] or 0.0
                new_avg = (
                    ((old_avg * n) + confidence) / (n + 1) if n >= 0 else confidence
                )
            else:
                new_avg = confidence

            is_hold = 1 if action == "HOLD" else 0
            is_buy = 1 if action == "BUY" else 0
            is_sell = 1 if action == "SELL" else 0

            db.execute(
                """
                UPDATE ticker_health SET
                    total_analyses = total_analyses + 1,
                    avg_confidence = %s,
                    hold_streak = CASE WHEN %s = 1 THEN hold_streak + 1 ELSE 0 END,
                    last_action = %s,
                    last_confidence = %s,
                    buy_count = buy_count + %s,
                    sell_count = sell_count + %s,
                    last_analyzed_at = %s,
                    updated_at = %s
                WHERE ticker = %s
            """,
                (
                    new_avg,
                    is_hold,
                    action,
                    confidence,
                    is_buy,
                    is_sell,
                    now,
                    now,
                    ticker,
                ),
            )
        except Exception as e:
            logger.warning(
                "[PIPELINE] ticker_health: update_analysis %s failed: %s", ticker, e
            )


def compute_health_score(ticker: str) -> dict:
    """Compute the 0-100 health score for a ticker.

    Returns:
        {"ticker": str, "score": int, "tier": str, "breakdown": dict}
    """
    ticker = ticker.upper().strip()
    _ensure_row(ticker)
    with get_db() as db:
        row = db.execute(
            """
            SELECT total_cycles, news_article_count, reddit_post_count,
                   youtube_count, zero_news_streak, collection_failures,
                   total_analyses, avg_confidence, hold_streak,
                   buy_count, sell_count
            FROM ticker_health WHERE ticker = %s
        """,
            (ticker,),
        ).fetchone()

    if not row:
        return {"ticker": ticker, "score": 50, "tier": "new", "breakdown": {}}

    (
        total_cycles,
        news_count,
        reddit_count,
        youtube_count,
        zero_streak,
        coll_failures,
        total_analyses,
        avg_conf,
        hold_streak,
        buy_count,
        sell_count,
    ) = row

    grace = settings.WATCHLIST_GRACE_CYCLES

    # Within grace period → return neutral score
    if (total_cycles or 0) < grace:
        return {
            "ticker": ticker,
            "score": 50,
            "tier": "new",
            "breakdown": {"reason": f"grace period ({total_cycles}/{grace} cycles)"},
        }

    # ── Data Richness (25 pts) ──
    # More data = better. Capped at 10 articles per source for full score.
    news_pts = min((news_count or 0), 10) / 10 * 25
    reddit_pts = min((reddit_count or 0), 8) / 8 * 25
    youtube_pts = min((youtube_count or 0), 5) / 5 * 25
    data_richness = news_pts * 0.5 + reddit_pts * 0.3 + youtube_pts * 0.2

    # ── Data Freshness (15 pts) ──
    # Penalty for consecutive zero-data cycles
    streak = zero_streak or 0
    if streak == 0:
        data_freshness = 15.0
    elif streak <= 2:
        data_freshness = 10.0
    elif streak <= 4:
        data_freshness = 5.0
    else:
        data_freshness = 0.0

    # ── Collection Reliability (10 pts) ──
    fails = coll_failures or 0
    cycles = max(total_cycles or 1, 1)
    fail_rate = fails / cycles
    if fail_rate <= 0.1:
        coll_reliability = 10.0
    elif fail_rate <= 0.3:
        coll_reliability = 6.0
    elif fail_rate <= 0.5:
        coll_reliability = 3.0
    else:
        coll_reliability = 0.0

    # ── Analysis Confidence (20 pts) ──
    conf = avg_conf or 0
    if conf >= 80:
        confidence_pts = 20.0
    elif conf >= 65:
        confidence_pts = 15.0
    elif conf >= 50:
        confidence_pts = 10.0
    elif conf >= 35:
        confidence_pts = 5.0
    else:
        confidence_pts = 0.0

    # ── Action Signals (15 pts) ──
    # BUY/SELL signals show the stock is actionable
    actions_total = (buy_count or 0) + (sell_count or 0)
    hold_s = hold_streak or 0
    if actions_total >= 3:
        action_pts = 15.0
    elif actions_total >= 1:
        action_pts = 10.0
    elif hold_s >= 5:
        action_pts = 0.0
    elif hold_s >= 3:
        action_pts = 5.0
    else:
        action_pts = 7.0  # neutral — not enough data

    # ── Tenure Bonus (15 pts) ──
    # Reward tickers that have been around long enough to accumulate data
    if cycles >= 10:
        tenure_pts = 15.0
    elif cycles >= 5:
        tenure_pts = 10.0
    else:
        tenure_pts = 5.0

    total_score = int(
        data_richness
        + data_freshness
        + coll_reliability
        + confidence_pts
        + action_pts
        + tenure_pts
    )
    total_score = max(0, min(100, total_score))

    # Determine tier
    if total_score >= 80:
        tier = "strong"
    elif total_score >= 60:
        tier = "healthy"
    elif total_score >= 30:
        tier = "weak"
    else:
        tier = "critical"

    breakdown = {
        "data_richness": round(data_richness, 1),
        "data_freshness": round(data_freshness, 1),
        "coll_reliability": round(coll_reliability, 1),
        "confidence": round(confidence_pts, 1),
        "action_signals": round(action_pts, 1),
        "tenure": round(tenure_pts, 1),
    }

    # Persist score
    now = datetime.now(timezone.utc)
    with get_db() as db:
        try:
            db.execute(
                """
                UPDATE ticker_health SET
                    health_score = %s, health_tier = %s,
                    last_scored_at = %s, updated_at = %s
                WHERE ticker = %s
            """,
                (total_score, tier, now, now, ticker),
            )

            # Denormalize to watchlist for fast API reads
            db.execute(
                "UPDATE watchlist SET health_score = %s WHERE ticker = %s",
                (total_score, ticker),
            )
        except Exception as e:
            logger.warning(
                "[PIPELINE] ticker_health: persist score %s failed: %s", ticker, e
            )

    return {
        "ticker": ticker,
        "score": total_score,
        "tier": tier,
        "breakdown": breakdown,
    }


def score_all_active() -> list[dict]:
    """Score all active watchlist tickers. Returns sorted list (worst first)."""
    with get_db() as db:
        try:
            rows = db.execute(
                "SELECT ticker FROM watchlist WHERE status = 'active'"
            ).fetchall()
        except Exception:
            return []

    results = []
    for (ticker,) in rows:
        r = compute_health_score(ticker)
        results.append(r)

    # Sort worst first
    results.sort(key=lambda x: x["score"])

    logger.info(
        "watchlist_health: scored %d tickers | worst: %s (%d) | best: %s (%d)",
        len(results),
        results[0]["ticker"] if results else "n/a",
        results[0]["score"] if results else 0,
        results[-1]["ticker"] if results else "n/a",
        results[-1]["score"] if results else 0,
    )

    return results


def get_purge_candidates(
    max_purge: int | None = None,
    min_score: int | None = None,
) -> list[dict]:
    """Get tickers eligible for purge.

    Filters:
      - Score below min_score
      - NOT holding open positions
      - NOT within grace period (tier != 'new')
      - NOT banned or already removed

    Returns list of dicts sorted worst-first, capped at max_purge.
    """
    if max_purge is None:
        max_purge = settings.WATCHLIST_MAX_PURGE
    if min_score is None:
        min_score = settings.WATCHLIST_PURGE_MIN_SCORE

    with get_db() as db:
        # Get tickers with open positions (immune) — filter by active bot
        try:
            from app.services.bot_manager import get_active_bot_id

            bid = get_active_bot_id()
        except Exception as e:
            logger.warning("[watchlist_health] Failed to get active bot ID: %s", e)
            from app.config import settings as _cfg

            bid = _cfg.BOT_ID
        try:
            pos_rows = db.execute(
                "SELECT DISTINCT ticker FROM positions WHERE qty > 0 AND bot_id = %s",
                (bid,),
            ).fetchall()
            positions = {r[0] for r in pos_rows}
        except Exception as e:
            logger.warning("[watchlist_health] Failed to fetch active position tickers: %s", e)
            positions = set()

    all_scores = score_all_active()

    candidates = []
    for item in all_scores:
        ticker = item["ticker"]
        score = item["score"]
        tier = item["tier"]

        # Skip grace period tickers
        if tier == "new":
            continue

        # Skip tickers with open positions
        if ticker in positions:
            continue

        # Only consider tickers below the threshold
        if score < min_score:
            candidates.append(item)

    # Already sorted worst-first, cap at max_purge
    return candidates[:max_purge]

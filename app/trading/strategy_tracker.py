"""
Strategy Tracker — P&L tracking per system prompt.

Links strategy candidates to trade outcomes to compute
win rates per analytical lens/prompt. Benches underperformers,
gives confidence bonuses to proven winners.

Extends the existing outcome_tracker.py with prompt-level tracking.

Usage:
    from app.trading.strategy_tracker import (
        record_strategy, evaluate_pnl, compute_rankings,
        get_confidence_bonus, bench_underperformers,
    )

    # After a BUY/SELL decision using a specific prompt:
    record_strategy(candidate_id, outcome_id, prompt_hash, ticker, "BUY", 150.0)

    # When a trade closes:
    evaluate_pnl(ticker, exit_price=165.0)

    # Periodic ranking update:
    rankings = compute_rankings()
    bench_underperformers()
"""

import logging
import uuid
from datetime import datetime, timezone

from app.config import settings
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def record_strategy(
    strategy_candidate_id: str | None,
    decision_outcome_id: str | None,
    agent_prompt_hash: str,
    ticker: str,
    signal: str,
    entry_price: float | None = None,
) -> str | None:
    """Record a strategy performance entry for P&L tracking.

    Called after a trading decision is made, linking the specific
    prompt (via hash) to the decision outcome.

    Args:
        strategy_candidate_id: FK to strategy_candidates table
        decision_outcome_id: FK to decision_outcomes table
        agent_prompt_hash: SHA256 hash of the system prompt used
        ticker: Stock ticker
        signal: BUY | SELL | HOLD
        entry_price: Entry price at time of decision

    Returns:
        Strategy performance ID if recorded, None if skipped (HOLD)
    """
    if signal not in ("BUY", "SELL"):
        return None  # Only track actionable signals

    with get_db() as db:
        perf_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        try:
            db.execute(
                """
                INSERT INTO strategy_performance
                (id, strategy_candidate_id, decision_outcome_id,
                 agent_prompt_hash, ticker, signal, entry_price, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    perf_id,
                    strategy_candidate_id,
                    decision_outcome_id,
                    agent_prompt_hash,
                    ticker,
                    signal,
                    entry_price,
                    now,
                ],
            )

            logger.info(
                "[STRATEGY] Recorded %s %s @ $%.2f (prompt=%s, id=%s)",
                signal,
                ticker,
                entry_price or 0,
                agent_prompt_hash[:8],
                perf_id[:8],
            )
            return perf_id

        except Exception as e:
            logger.warning("[STRATEGY] Failed to record %s %s: %s", signal, ticker, e)
            return None


def evaluate_pnl(ticker: str, exit_price: float) -> list[dict]:
    """Resolve open strategy performance entries for a closed trade.

    Finds all unresolved BUY entries for this ticker, calculates
    return_pct, and marks as WIN/LOSS.

    Args:
        ticker: Stock ticker
        exit_price: Price at which the position was closed

    Returns:
        List of resolved strategy performance records
    """
    with get_db() as db:
        resolved = []

        try:
            rows = db.execute(
                """
                SELECT id, entry_price, signal, agent_prompt_hash
                FROM strategy_performance
                WHERE ticker = %s AND signal = 'BUY'
                  AND resolved_at IS NULL AND active = TRUE
                """,
                [ticker],
            ).fetchall()

            if not rows:
                return []

            now = datetime.now(timezone.utc)

            for row in rows:
                perf_id = row[0]
                entry_price = row[1]
                prompt_hash = row[3]

                if entry_price and entry_price > 0:
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    win = return_pct > 0.5  # >0.5% return = win
                else:
                    return_pct = 0.0
                    win = False

                # Calculate hold days
                created_row = db.execute(
                    "SELECT created_at FROM strategy_performance WHERE id = %s",
                    [perf_id],
                ).fetchone()
                hold_days = 0
                if created_row and created_row[0]:
                    try:
                        created_at = created_row[0]
                        if isinstance(created_at, str):
                            created_at = datetime.fromisoformat(created_at)
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        hold_days = (now - created_at).days
                    except Exception:
                        pass

                db.execute(
                    """
                    UPDATE strategy_performance
                    SET exit_price = %s, return_pct = %s, win = %s,
                        hold_days = %s, resolved_at = %s
                    WHERE id = %s
                    """,
                    [exit_price, round(return_pct, 2), win, hold_days, now, perf_id],
                )

                resolved.append(
                    {
                        "id": perf_id,
                        "ticker": ticker,
                        "prompt_hash": prompt_hash,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": round(return_pct, 2),
                        "win": win,
                        "hold_days": hold_days,
                    }
                )

                logger.info(
                    "[STRATEGY] Resolved %s: %s (%.1f%%, prompt=%s)",
                    ticker,
                    "WIN" if win else "LOSS",
                    return_pct,
                    prompt_hash[:8],
                )

            # Update generated_agent_prompts stats
            _update_prompt_stats(db)

        except Exception as e:
            logger.warning("[STRATEGY] Failed to evaluate P&L for %s: %s", ticker, e)

        return resolved


def _update_prompt_stats(db=None) -> None:
    """Recalculate win_rate and total_trades for all generated prompts."""
    if db is None:
        with get_db() as new_db:
            _update_prompt_stats(new_db)
        return

    try:
        db.execute(
            """
            UPDATE generated_agent_prompts
            SET
                total_trades = COALESCE(sub.total, 0),
                win_rate = COALESCE(sub.win_rate, 0.0),
                performance_score = COALESCE(sub.avg_return, 0.0)
            FROM (
                SELECT
                    agent_prompt_hash,
                    COUNT(*) as total,
                    AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(return_pct) as avg_return
                FROM strategy_performance
                WHERE resolved_at IS NOT NULL
                GROUP BY agent_prompt_hash
            ) AS sub
            WHERE generated_agent_prompts.prompt_hash = sub.agent_prompt_hash
            """
        )
    except Exception as e:
        logger.debug("[STRATEGY] Stats update failed (non-fatal): %s", e)


def compute_rankings(limit: int = 50) -> list[dict]:
    """Compute strategy performance rankings by prompt hash.

    Returns a leaderboard sorted by win rate (min trades required).
    """
    with get_db() as db:
        try:
            rows = db.execute(
                """
                SELECT
                    agent_prompt_hash,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN win THEN 1 ELSE 0 END) as wins,
                    AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(return_pct) as avg_return,
                    AVG(hold_days) as avg_hold_days,
                    MIN(created_at) as first_trade,
                    MAX(resolved_at) as last_trade
                FROM strategy_performance
                WHERE resolved_at IS NOT NULL
                GROUP BY agent_prompt_hash
                HAVING COUNT(*) >= 3
                ORDER BY win_rate DESC, avg_return DESC
                LIMIT %s
                """,
                [limit],
            ).fetchall()

            rankings = []
            for row in rows:
                # Look up the prompt name if it's a generated prompt
                name_row = db.execute(
                    "SELECT name, lens_type FROM generated_agent_prompts WHERE prompt_hash = %s",
                    [row[0]],
                ).fetchone()

                rankings.append(
                    {
                        "prompt_hash": row[0],
                        "name": name_row[0] if name_row else "static_lens",
                        "lens_type": name_row[1] if name_row else "unknown",
                        "total_trades": row[1],
                        "wins": row[2],
                        "win_rate": round(row[3], 3),
                        "avg_return_pct": round(row[4], 2),
                        "avg_hold_days": round(row[5], 1) if row[5] else 0,
                        "first_trade": str(row[6]),
                        "last_trade": str(row[7]),
                    }
                )

            return rankings

        except Exception as e:
            logger.warning("[STRATEGY] Rankings computation failed: %s", e)
            return []


def get_confidence_bonus(prompt_hash: str) -> int:
    """Get a confidence bonus for historically winning prompts.

    Returns:
        +5 if win_rate > WIN_RATE_BONUS_THRESHOLD and enough trades
        0 otherwise
    """
    with get_db() as db:
        try:
            row = db.execute(
                """
                SELECT
                    COUNT(*) as total,
                    AVG(CASE WHEN win THEN 1.0 ELSE 0.0 END) as win_rate
                FROM strategy_performance
                WHERE agent_prompt_hash = %s AND resolved_at IS NOT NULL
                """,
                [prompt_hash],
            ).fetchone()

            if not row or row[0] < settings.MIN_TRADES_BEFORE_BENCH:
                return 0

            win_rate = row[1] or 0.0
            if win_rate >= settings.WIN_RATE_BONUS_THRESHOLD:
                logger.debug(
                    "[STRATEGY] Confidence bonus for prompt %s (%.0f%% win rate)",
                    prompt_hash[:8],
                    win_rate * 100,
                )
                return 5

        except Exception:
            pass

        return 0


def bench_underperformers() -> list[str]:
    """Deactivate generated prompts with poor win rates.

    Only benches prompts that have enough trades to be statistically
    meaningful (MIN_TRADES_BEFORE_BENCH).

    Returns list of benched prompt hashes.
    """
    with get_db() as db:
        benched = []

        try:
            rows = db.execute(
                """
                SELECT
                    gp.prompt_hash,
                    gp.name,
                    COUNT(sp.id) as total_trades,
                    AVG(CASE WHEN sp.win THEN 1.0 ELSE 0.0 END) as win_rate
                FROM generated_agent_prompts gp
                JOIN strategy_performance sp ON gp.prompt_hash = sp.agent_prompt_hash
                WHERE gp.active = TRUE AND sp.resolved_at IS NOT NULL
                GROUP BY gp.prompt_hash, gp.name
                HAVING COUNT(sp.id) >= %s
                   AND AVG(CASE WHEN sp.win THEN 1.0 ELSE 0.0 END) < %s
                """,
                [settings.MIN_TRADES_BEFORE_BENCH, settings.WIN_RATE_BENCH_THRESHOLD],
            ).fetchall()

            now = datetime.now(timezone.utc)
            for row in rows:
                prompt_hash, name, total, win_rate = row[0], row[1], row[2], row[3]

                db.execute(
                    """
                    UPDATE generated_agent_prompts
                    SET active = FALSE, benched_at = %s
                    WHERE prompt_hash = %s
                    """,
                    [now, prompt_hash],
                )
                benched.append(prompt_hash)

                logger.info(
                    "[STRATEGY] Benched '%s' (hash=%s): %.0f%% win rate over %d trades",
                    name,
                    prompt_hash[:8],
                    (win_rate or 0) * 100,
                    total,
                )

        except Exception as e:
            logger.warning("[STRATEGY] Bench check failed: %s", e)

        return benched


def get_ticker_strategy_timeline(ticker: str, limit: int = 20) -> list[dict]:
    """Get the full Data → Candidate → Performance timeline for a ticker.

    Used by the API to show the complete lifecycle of a strategy.
    """
    with get_db() as db:
        try:
            rows = db.execute(
                """
                SELECT
                    sc.lens_name,
                    sc.signal as candidate_signal,
                    sc.confidence_score,
                    sc.created_at as analyzed_at,
                    sp.signal as trade_signal,
                    sp.entry_price,
                    sp.exit_price,
                    sp.return_pct,
                    sp.win,
                    sp.hold_days,
                    sp.resolved_at
                FROM strategy_candidates sc
                LEFT JOIN strategy_performance sp
                  ON sc.id = sp.strategy_candidate_id
                WHERE sc.ticker = %s
                ORDER BY sc.created_at DESC
                LIMIT %s
                """,
                [ticker, limit],
            ).fetchall()

            return [
                {
                    "lens": row[0],
                    "candidate_signal": row[1],
                    "confidence": row[2],
                    "analyzed_at": str(row[3]),
                    "trade_signal": row[4],
                    "entry_price": row[5],
                    "exit_price": row[6],
                    "return_pct": row[7],
                    "win": row[8],
                    "hold_days": row[9],
                    "resolved_at": str(row[10]) if row[10] else None,
                }
                for row in rows
            ]

        except Exception as e:
            logger.warning("[STRATEGY] Timeline query failed for %s: %s", ticker, e)
            return []

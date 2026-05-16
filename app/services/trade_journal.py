"""
Trade Journal -- Unified read API for trade history and decision outcomes.

Provides queryable access to the bot's own trading history:
  - Past decisions (action, confidence, rationale)
  - Trade outcomes (WIN/LOSS/FLAT, PnL)
  - Lessons learned

Powers the get_past_decisions() RLM tool and the Decision Memory
Feedback Loop described in the KI.

Usage:
    from app.services.trade_journal import get_decision_history, get_trade_stats
"""

import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def get_decision_history(
    ticker: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Get past analysis decisions with their outcomes (if resolved).

    Joins analysis_results with decision_outcomes to show what the bot
    decided and whether it was right.
    """
    try:
        with get_db() as db:
            if ticker:
                rows = db.execute(
                    """
                    SELECT
                        ar.ticker,
                        ar.agent_name,
                        ar.confidence,
                        ar.result_json,
                        ar.created_at,
                        dout.action AS outcome_action,
                        dout.entry_price,
                        dout.exit_price,
                        dout.pnl_pct,
                        dout.outcome,
                        dout.lesson_stored
                    FROM analysis_results ar
                    LEFT JOIN decision_outcomes dout
                        ON ar.ticker = dout.ticker
                        AND ar.cycle_id = dout.cycle_id
                    WHERE ar.ticker = %s
                    ORDER BY ar.created_at DESC
                    LIMIT %s
                """,
                    [ticker.upper(), limit],
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT
                        ar.ticker,
                        ar.agent_name,
                        ar.confidence,
                        ar.result_json,
                        ar.created_at,
                        dout.action AS outcome_action,
                        dout.entry_price,
                        dout.exit_price,
                        dout.pnl_pct,
                        dout.outcome,
                        dout.lesson_stored
                    FROM analysis_results ar
                    LEFT JOIN decision_outcomes dout
                        ON ar.ticker = dout.ticker
                        AND ar.cycle_id = dout.cycle_id
                    ORDER BY ar.created_at DESC
                    LIMIT %s
                """,
                    [limit],
                ).fetchall()

        results = []
        for r in rows:
            import json

            try:
                result_data = json.loads(r[3]) if r[3] else {}
            except Exception:
                result_data = {}

            entry = {
                "ticker": r[0],
                "config": r[1],
                "confidence": r[2],
                "action": result_data.get("action", "?"),
                "rationale": result_data.get("rationale", "")[:200],
                "decided_at": str(r[4]) if r[4] else None,
            }

            # Attach outcome if resolved
            if r[9]:  # outcome exists
                entry["outcome"] = r[9]
                entry["pnl_pct"] = r[8]
                entry["entry_price"] = r[6]
                entry["exit_price"] = r[7]
                entry["lesson"] = r[10]

            results.append(entry)

        return results

    except Exception as e:
        logger.warning("[JOURNAL] Decision history query failed: %s", e)
        return []


def get_trade_stats(bot_id: str = "lazy-trader-v4") -> dict:
    """Get aggregate trading statistics from the trade journal."""
    try:
        with get_db() as db:
            # Win/loss stats from decision_outcomes
            stats_row = db.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN outcome = 'FLAT' THEN 1 ELSE 0 END) AS flats,
                    AVG(pnl_pct) AS avg_pnl,
                    AVG(CASE WHEN outcome = 'WIN' THEN pnl_pct END) AS avg_win,
                    AVG(CASE WHEN outcome = 'LOSS' THEN pnl_pct END) AS avg_loss
                FROM decision_outcomes
                WHERE resolved_at IS NOT NULL
            """).fetchone()

        if not stats_row or stats_row[0] == 0:
            return {"total_resolved": 0, "message": "No resolved trades yet"}

        total = stats_row[0]
        wins = stats_row[1] or 0
        win_rate = (wins / total * 100) if total > 0 else 0

        return {
            "total_resolved": total,
            "wins": wins,
            "losses": stats_row[2] or 0,
            "flats": stats_row[3] or 0,
            "win_rate_pct": round(win_rate, 1),
            "avg_pnl_pct": round(stats_row[4] or 0, 2),
            "avg_win_pct": round(stats_row[5] or 0, 2),
            "avg_loss_pct": round(stats_row[6] or 0, 2),
        }

    except Exception as e:
        logger.warning("[JOURNAL] Stats query failed: %s", e)
        return {"error": str(e)}


def get_portfolio_exposure(bot_id: str = "lazy-trader-v4") -> dict:
    """Get current portfolio sector exposure breakdown.

    Powers the get_portfolio_exposure() RLM tool.
    """
    try:
        with get_db() as db:
            rows = db.execute(
                """
                SELECT
                    COALESCE(tm.sector, 'Unknown') AS sector,
                    COUNT(*) AS position_count,
                    SUM(p.qty * p.avg_entry_price) AS total_value
                FROM positions p
                LEFT JOIN ticker_metadata tm ON p.ticker = tm.ticker
                WHERE p.bot_id = %s
                GROUP BY COALESCE(tm.sector, 'Unknown')
                ORDER BY total_value DESC
            """,
                [bot_id],
            ).fetchall()

        total_value = sum(r[2] or 0 for r in rows)
        sectors = []
        for r in rows:
            val = r[2] or 0
            pct = (val / total_value * 100) if total_value > 0 else 0
            sectors.append(
                {
                    "sector": r[0],
                    "positions": r[1],
                    "value_usd": round(val, 2),
                    "pct_of_portfolio": round(pct, 1),
                }
            )

        return {
            "total_positions": sum(r[1] for r in rows),
            "total_invested": round(total_value, 2),
            "sectors": sectors,
        }

    except Exception as e:
        logger.warning("[JOURNAL] Exposure query failed: %s", e)
        return {"error": str(e)}

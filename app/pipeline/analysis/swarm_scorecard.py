"""
Swarm Scorecard — Tracks individual model predictions for grading.

Each model's prediction is logged after every Swarm run. A separate
grading pass (cron or manual) later compares predictions to actual
market outcomes and computes accuracy scores per model.
"""

import logging
import uuid
from typing import Dict, Any

logger = logging.getLogger(__name__)

AGENT_WEIGHTS = {
    "quant_26B": 0.35,
    "macro_35B": 0.25,
    "cio_120B": 0.40
}


async def log_predictions(ticker: str, cycle_id: str, predictions: Dict[str, Any]):
    """
    Save each model's individual prediction to the swarm_scorecards table.

    Args:
        ticker: The stock symbol (e.g., "NVDA")
        cycle_id: Unique ID for this Swarm run
        predictions: Dict of {label: prediction_dict} from Phase 2
    """
    try:
        from app.db.connection import get_db

        with get_db() as db:
            for label, pred in predictions.items():
                try:
                    row_id = str(uuid.uuid4())
                    db.execute(
                        """INSERT INTO swarm_scorecards (
                            id, ticker, cycle_id, model_label, model_id,
                            predicted_action, predicted_confidence,
                            predicted_price_target, predicted_stop_loss,
                            key_signals, rationale
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            row_id,
                            ticker,
                            cycle_id,
                            label,
                            pred.get("model_id", "unknown"),
                            pred.get("action", "HOLD"),
                            pred.get("confidence", 0),
                            pred.get("price_target_5d", 0.0),
                            pred.get("stop_loss", 0.0),
                            str(pred.get("key_signals", [])),
                            pred.get("rationale", ""),
                        ),
                    )
                    logger.info(
                        f"[Scorecard] Logged {label} ({pred.get('model_id', '?')}): "
                        f"{pred.get('action', '?')} @ {pred.get('confidence', '?')}%"
                    )
                except Exception as e:
                    logger.error(f"[Scorecard] Failed to log {label}: {e}")
    except Exception as e:
        logger.warning(f"[Scorecard] DB not available, skipping: {e}")
        return


async def grade_predictions(
    ticker: str, actual_action: str, actual_price_change_pct: float
):
    """
    Grade past predictions for a ticker against actual market outcomes.

    Called by a cron job or manual trigger after the trade window closes.

    Args:
        ticker: Stock symbol
        actual_action: What the correct action would have been ("BUY"/"SELL"/"HOLD")
        actual_price_change_pct: Actual 5-day price change percentage
    """
    try:
        from app.db.connection import get_db

        with get_db() as db:
            ungraded = db.execute(
                """SELECT id, model_label, model_id, predicted_action, predicted_confidence,
                          predicted_price_target
                   FROM swarm_scorecards
                   WHERE ticker = %s AND graded_at IS NULL
                   ORDER BY created_at DESC""",
                (ticker,),
            ).fetchall()

            for row in ungraded:
                row_id, label, model_id, pred_action, pred_conf, pred_target = row

                # Score: was the directional call correct?
                action_correct = pred_action == actual_action
                # Score: confidence calibration (high confidence on correct = good)
                if action_correct:
                    accuracy = min(100, pred_conf)
                else:
                    accuracy = max(0, 100 - pred_conf)

                db.execute(
                    """UPDATE swarm_scorecards
                       SET actual_action = %s,
                           actual_price_change_pct = %s,
                           accuracy_score = %s,
                           action_correct = %s,
                           graded_at = CURRENT_TIMESTAMP
                       WHERE id = %s""",
                    (
                        actual_action,
                        actual_price_change_pct,
                        accuracy,
                        action_correct,
                        row_id,
                    ),
                )
                logger.info(
                    f"[Scorecard] Graded {label} ({model_id}): "
                    f"predicted={pred_action}, actual={actual_action}, "
                    f"correct={action_correct}, score={accuracy}"
                )
    except Exception as e:
        logger.warning(f"[Scorecard] DB not available for grading / failed: {e}")
        return


def get_leaderboard() -> list[dict]:
    """
    Return the model leaderboard — ranked by average accuracy score.

    Returns a list of dicts sorted by avg_accuracy descending:
    [{"model_id": "...", "total_predictions": N, "correct": N,
      "avg_accuracy": 85.2, "avg_confidence": 72.1}]
    """
    try:
        from app.db.connection import get_db

        with get_db() as db:
            rows = db.execute(
                """SELECT model_id, model_label,
                          COUNT(*) as total_predictions,
                          SUM(CASE WHEN action_correct THEN 1 ELSE 0 END) as correct,
                          AVG(accuracy_score) as avg_accuracy,
                          AVG(predicted_confidence) as avg_confidence
                   FROM swarm_scorecards
                   WHERE graded_at IS NOT NULL
                   GROUP BY model_id, model_label
                   ORDER BY avg_accuracy DESC"""
            ).fetchall()

            return [
                {
                    "model_id": row[0],
                    "model_label": row[1],
                    "total_predictions": row[2],
                    "correct": row[3],
                    "avg_accuracy": round(row[4], 1) if row[4] else 0,
                    "avg_confidence": round(row[5], 1) if row[5] else 0,
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"[Scorecard] Leaderboard query failed: {e}")
        return []

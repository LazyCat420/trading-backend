"""
Lens Scorecard & Reaper
Tracks the performance of dynamic system prompts (lenses) and automatically
retires (reaps) those that consistently underperform.
"""

import logging
import uuid

from app.db.connection import get_db

logger = logging.getLogger(__name__)


class LensScorecard:
    @staticmethod
    def log_lens_usage(
        lens_name: str,
        lens_type: str,
        system_prompt: str,
        cycle_id: str,
        ticker: str,
        action: str,
        confidence: int,
    ):
        """Log that a specific lens was used for a prediction."""
        try:
            with get_db() as db:
                row_id = str(uuid.uuid4())
                db.execute(
                    """INSERT INTO lens_scorecard (
                        id, lens_name, lens_type, system_prompt, cycle_id, ticker, 
                        predicted_action, predicted_confidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        row_id,
                        lens_name,
                        lens_type,
                        system_prompt,
                        cycle_id,
                        ticker,
                        action,
                        confidence,
                    ],
                )
        except Exception as e:
            logger.error(f"[LensScorecard] Failed to log lens usage: {e}")

    @staticmethod
    def grade_lens(ticker: str, actual_action: str, actual_price_change_pct: float):
        """Grade past uses of lenses against actual market outcomes."""
        try:
            with get_db() as db:
                ungraded = db.execute(
                    """SELECT id, lens_name, predicted_action, predicted_confidence
                       FROM lens_scorecard
                       WHERE ticker = %s AND graded_at IS NULL""",
                    [ticker],
                ).fetchall()

                for row in ungraded:
                    row_id, lens_name, pred_action, pred_conf = row

                    action_correct = pred_action == actual_action
                    if action_correct:
                        accuracy = min(100, pred_conf)
                    else:
                        accuracy = max(0, 100 - pred_conf)

                    db.execute(
                        """UPDATE lens_scorecard
                           SET actual_action = %s, actual_price_change_pct = %s,
                               accuracy_score = %s, action_correct = %s, graded_at = CURRENT_TIMESTAMP
                           WHERE id = %s""",
                        [
                            actual_action,
                            actual_price_change_pct,
                            accuracy,
                            action_correct,
                            row_id,
                        ],
                    )
                    logger.info(
                        f"[LensScorecard] Graded {lens_name}: correct={action_correct}, score={accuracy}"
                    )
        except Exception as e:
            logger.error(f"[LensScorecard] Grading failed: {e}")


class LensReaper:
    """Automatically retires underperforming lenses."""

    @staticmethod
    def reap_underperformers(min_uses: int = 10, threshold_score: float = 40.0):
        """Find lenses with >= min_uses and avg score < threshold_score, and mark them inactive."""
        try:
            with get_db() as db:
                # Find lenses that have enough data
                stats = db.execute(
                    """SELECT lens_name, COUNT(*) as uses, AVG(accuracy_score) as avg_score
                       FROM lens_scorecard
                       WHERE graded_at IS NOT NULL
                       GROUP BY lens_name
                       HAVING COUNT(*) >= %s""",
                    [min_uses],
                ).fetchall()

                for row in stats:
                    lens_name, uses, avg_score = row
                    if avg_score < threshold_score:
                        logger.warning(
                            f"[LensReaper] Lens '{lens_name}' underperforming (score {avg_score:.1f} over {uses} uses). Reaping."
                        )
                        # Mark as inactive in the prompt versions or system prompt store.
                        # Assuming there's a prompt_versions table holding these generated lenses
                        db.execute(
                            "UPDATE prompt_versions SET is_active = FALSE WHERE name = %s",
                            [lens_name],
                        )

        except Exception as e:
            logger.error(f"[LensReaper] Reaping failed: {e}")

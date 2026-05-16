import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def _get_score(confidence: int, is_correct: bool = None) -> float:
    score = float(confidence)
    if is_correct is True:
        score += 100.0
    elif is_correct is False:
        score -= 100.0
    return score


def save_best_result(
    ticker: str, result_data: dict, confidence: int, is_correct: bool = None
) -> bool:
    try:
        new_score = _get_score(confidence, is_correct)

        with get_db() as db:
            # Check existing
            row = db.execute(
                "SELECT score FROM best_per_ticker WHERE ticker = %s", (ticker,)
            ).fetchone()
            if row:
                old_score = row[0] or 0.0
                if new_score <= old_score:
                    logger.debug(
                        "Improvement gate blocked overwrite for %s (new %s <= old %s)",
                        ticker,
                        new_score,
                        old_score,
                    )
                    return False

            # Upsert
            db.execute(
                """INSERT INTO best_per_ticker (ticker, action, confidence, rationale, is_correct, score)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (ticker) DO UPDATE SET
                       action = EXCLUDED.action,
                       confidence = EXCLUDED.confidence,
                       rationale = EXCLUDED.rationale,
                       is_correct = EXCLUDED.is_correct,
                       score = EXCLUDED.score,
                       updated_at = CURRENT_TIMESTAMP
                """,
                (
                    ticker,
                    result_data.get("action"),
                    confidence,
                    result_data.get("rationale", ""),
                    is_correct,
                    new_score,
                ),
            )
        logger.info("Saved best result for %s (score %s)", ticker, new_score)
        return True
    except Exception as e:
        logger.error("Failed to save best result for %s: %s", ticker, e)
        return False


def load_best_result(ticker: str) -> dict | None:
    try:
        with get_db() as db:
            row = db.execute(
                "SELECT action, confidence, rationale, is_correct, score FROM best_per_ticker WHERE ticker = %s",
                (ticker,),
            ).fetchone()
            if row:
                return {
                    "action": row[0],
                    "confidence": row[1],
                    "rationale": row[2],
                    "is_correct": row[3],
                    "score": row[4],
                }
    except Exception as e:
        logger.warning("Failed to load best result for %s: %s", ticker, e)
    return None

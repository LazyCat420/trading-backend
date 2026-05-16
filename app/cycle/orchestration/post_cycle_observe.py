"""
Post-Cycle Observation -- Capture raw episodic observations from decisions and outcomes.
Developer 2 assignment.

Creates structured episodic observations after:
  - high-confidence decisions
  - low-confidence/confused decisions
  - escalations/disagreements
  - failed theses
  - outcome resolution

Writes observations to DB via Developer 1's storage layer contracts (episodic_observations).
"""

import logging
import uuid
from datetime import datetime, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)


async def create_decision_observation(
    ticker: str,
    result: dict,
    escalated: bool,
    cycle_id: str,
) -> None:
    """Extract and store a raw episodic observation if the decision was noteworthy."""
    try:
        conf = result.get("confidence", 50)
        action = result.get("action", "HOLD")

        # Determine if thesis failed during debate
        d_result = result.get("d_result") or {}
        debate_meta = d_result.get("debate", {})
        thesis_failed = debate_meta.get("thesis_won") is False

        # Only record notable decisions:
        # High confidence >= 85
        # Low confidence <= 35
        # Escalated or Disagreements
        # Failed theses
        is_notable = conf >= 85 or conf <= 35 or escalated or thesis_failed

        if not is_notable:
            return

        rationale = result.get("rationale", "")
        rationale_excerpt = rationale[:500] + ("..." if len(rationale) > 500 else "")

        config_used = result.get("config_used", "C")

        observation_text = (
            f"Decision: {action} with {conf}% confidence. "
            f"Escalated: {escalated}. "
            f"Config: {config_used}. "
            f"Failed Thesis: {thesis_failed}."
        )

        _insert_episodic_observation(
            cycle_id=cycle_id,
            ticker=ticker,
            source_type="decision",
            observation_text=observation_text,
            rationale_excerpt=rationale_excerpt,
            confidence_at_creation=float(conf),
        )
        logger.info("[OBSERVE] %s: Stored decision observation", ticker)

    except Exception as e:
        logger.warning(
            "[PIPELINE] [OBSERVE] Failed decision observation for %s (non-fatal): %s",
            ticker,
            e,
        )


def create_outcome_observation(
    ticker: str,
    outcome: str,
    pnl_pct: float,
    cycle_id: str = "manual",
) -> None:
    """Record an outcome observation when a trade resolves."""
    try:
        observation_text = f"Trade resolved with outcome: {outcome} (PnL: {pnl_pct}%)"

        _insert_episodic_observation(
            cycle_id=cycle_id,
            ticker=ticker,
            source_type="outcome",
            observation_text=observation_text,
            rationale_excerpt=None,
            confidence_at_creation=None,
            outcome_label=outcome,
            outcome_score=float(pnl_pct),
        )
        logger.info("[OBSERVE] %s: Stored outcome observation", ticker)
    except Exception as e:
        logger.warning(
            "[PIPELINE] [OBSERVE] Failed outcome observation for %s (non-fatal): %s",
            ticker,
            e,
        )


def _insert_episodic_observation(
    cycle_id: str,
    ticker: str,
    source_type: str,
    observation_text: str,
    rationale_excerpt: str | None = None,
    confidence_at_creation: float | None = None,
    outcome_label: str | None = None,
    outcome_score: float | None = None,
) -> None:
    with get_db() as db:
        # Try to grab the sector if available, otherwise None
        sector = None
        try:
            row = db.execute(
                "SELECT sector FROM ticker_metadata WHERE ticker = %s", [ticker]
            ).fetchone()
            if row:
                sector = row[0]
        except Exception:
            pass

        obs_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Write directly to the episodic_observations contract table.
        db.execute(
            """
            INSERT INTO episodic_observations (
                id, created_at, cycle_id, ticker, sector, source_type,
                observation_text, rationale_excerpt, confidence_at_creation,
                outcome_label, outcome_score, promoted_to_memory
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                obs_id,
                now,
                cycle_id or "manual",
                ticker,
                sector,
                source_type,
                observation_text,
                rationale_excerpt,
                confidence_at_creation,
                outcome_label,
                outcome_score,
                False,
            ],
        )

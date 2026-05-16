from app.cognition.base import BaseCognitionModule
from app.log_manager import log_manager
import datetime


class ReflectiveMemoryLogger(BaseCognitionModule):
    """
    Dev 3 Stage 5 Component: Memory Logger.
    Finalizes the cycle by saving lessons learned to the reflective memory store and writing to specialized logs.
    """

    def __init__(self):
        super().__init__("ReflectiveMemoryLogger")

    async def _execute(self, thesis: dict, context: dict) -> dict:
        """
        Logs the final thesis and reflection to the cycle log.
        """
        cycle_id = context.get("cycle_id", "no_id")
        ticker = context.get("ticker", "no_ticker")

        # Build reflection payload
        reflection = {
            "entity_id": ticker,
            "action": thesis.get("action"),
            "confidence": thesis.get("confidence"),
            "core_claims": thesis.get("core_claims", []),
            "weaknesses": thesis.get("weaknesses", []),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # Use LogManager to save to V2 jsonl
        log_manager.log_v2_cycle(
            cycle_id=cycle_id, step_name="reflective_memory", payload=reflection
        )

        return {"status": "logged_to_rlm", "reflection": reflection}

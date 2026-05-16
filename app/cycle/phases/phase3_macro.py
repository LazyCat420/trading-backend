import logging
import asyncio
from typing import Callable

from app.config import settings

logger = logging.getLogger(__name__)


async def run_phase3_macro(emit: Callable) -> str:
    """
    Phase 3: Macro Scout
    Runs the macro strategy scout independently to generate a memo.
    Guarded by a strict timeout to prevent pipeline stalls.
    """
    if not settings.MACRO_SCOUT_ENABLED:
        logger.info("[CYCLE] Macro Scout disabled via settings.")
        return ""

    try:
        from app.pipeline.analysis.macro_scout import run_macro_scout

        emit(
            "analyzing",
            "macro_start",
            "Generating Macro Strategy Memo...",
            status="running",
        )

        # Guard the macro scout with a strict 5-minute timeout
        # to ensure it never hangs the pipeline.
        memo = await asyncio.wait_for(run_macro_scout(emit=emit), timeout=300.0)

        if memo:
            emit(
                "collecting",
                "macro_memo_ready",
                f"Macro memo ready ({len(memo)} chars)",
                status="ok",
            )
            return memo
        return ""

    except asyncio.TimeoutError:
        logger.error("Macro scout TIMEOUT after 300s.")
        emit(
            "collecting",
            "macro_scout_error",
            "Macro Scout timed out. Proceeding without memo.",
            status="error",
        )
        return ""
    except Exception as e:
        logger.error("Macro scout failed: %s", e)
        emit(
            "collecting",
            "macro_scout_error",
            f"Macro Scout failed: {e}",
            status="error",
        )
        return ""

import asyncio
import logging
from typing import TypeVar, Awaitable

T = TypeVar("T")
log = logging.getLogger(__name__)

async def run_with_timeout(
    coro: Awaitable[T],
    timeout: float,
    label: str,
    fallback: T = None,
) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        log.warning("[%s] Timed out after %.0fs", label, timeout)
        return fallback
    except Exception as e:
        log.warning("[%s] Failed: %s", label, e)
        return fallback

async def safe_agent(coro: Awaitable, name: str, cycle_id: str = "") -> None:
    try:
        await coro
    except asyncio.CancelledError:
        log.info("[%s] Cancelled", name)
    except Exception as e:
        log.error("[%s] Failed: %s", name, e)
        try:
            from app.cycle.orchestration.state_manager import PipelineStateDB
            PipelineStateDB.safe_log_execution_error(
                cycle_id=cycle_id,
                phase="post_trade",
                ticker="system",
                error_type=f"{name}_failure",
                error=e,
            )
        except Exception:
            pass

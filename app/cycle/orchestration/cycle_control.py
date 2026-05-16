"""
Global cycle control for Pause, Resume, and Stop functionality.

Pause:  Blocks execution at checkpoints (cooperative).
        In-flight API calls are allowed to finish; new work is blocked.
Stop:   Sets a cooperative cancellation flag. The next checkpoint
        raises asyncio.CancelledError, which propagates up to _run_cycle().
Reset:  Clears all flags for a fresh cycle start.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


class CycleControl:
    def __init__(self):
        self._pause_event = None
        self.is_paused = False
        self.is_stopped = False

    @property
    def pause_event(self) -> asyncio.Event:
        if self._pause_event is None:
            # Create the event in the context of the currently running loop
            try:
                self._pause_event = asyncio.Event()
                if not self.is_paused:
                    self._pause_event.set()
            except RuntimeError:
                # If called outside a loop, we might not be able to create it.
                # But it shouldn't happen during async execution.
                pass
        return self._pause_event

    def pause(self):
        """Pause execution at the next checkpoint."""
        self.is_paused = True
        if self.pause_event:
            self.pause_event.clear()
        logger.info("[PIPELINE] [CYCLE_CONTROL] Pause requested")

    def resume(self):
        """Resume execution from a paused state."""
        self.is_paused = False
        if self.pause_event:
            self.pause_event.set()
        logger.info("[PIPELINE] [CYCLE_CONTROL] Resume requested")

    def stop(self):
        """Request cooperative cancellation.

        Also unblocks pause so the wait_if_paused() call can
        detect the stop flag and raise CancelledError.
        """
        self.is_stopped = True
        # Unblock pause so the coroutine wakes up and sees is_stopped
        if self.pause_event:
            self.pause_event.set()
        logger.info("[PIPELINE] [CYCLE_CONTROL] Stop requested")

    def reset(self):
        """Clear all flags for a fresh cycle start.

        Re-creates the Event to ensure it's bound to the current event loop
        (prevents stale Event from import-time loop after uvicorn reload).
        """
        self.is_stopped = False
        self.is_paused = False
        self._pause_event = None  # Lazy init on next use to guarantee current loop
        logger.info("[PIPELINE] [CYCLE_CONTROL] Reset (fresh cycle)")

    async def wait_if_paused(self):
        """Await this before any major operation.

        - If stopped: raises CancelledError immediately.
        - If paused:  blocks until resume() or stop() is called.
        - If active:  returns instantly (no overhead).
        """
        # Check stop BEFORE blocking on pause
        if self.is_stopped:
            raise asyncio.CancelledError("Cycle stopped by user")

        if not self.pause_event.is_set():
            logger.info(
                "[PIPELINE] [CYCLE_CONTROL] Execution frozen, waiting for resume or stop..."
            )
            await self.pause_event.wait()
            logger.info("[PIPELINE] [CYCLE_CONTROL] Unblocked — checking stop flag...")

        # Check stop AFTER waking from pause (stop() calls resume())
        if self.is_stopped:
            raise asyncio.CancelledError("Cycle stopped by user")


# Singleton
cycle_control = CycleControl()

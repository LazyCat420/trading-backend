"""Pipeline orchestration service.

Refactored to act as a lightweight facade extending modular mixins.
"""

from app.pipeline.orchestration.state_manager import PipelineStateMixin
from app.pipeline.orchestration.lifecycle_controller import LifecycleControllerMixin
from app.pipeline.orchestration.orchestrator_core import OrchestratorCoreMixin
from app.pipeline.orchestration.orchestrator_v2 import OrchestratorV2Mixin
from app.pipeline.core import PipelineContext  # noqa: F401 — re-exported for backward compat


class PipelineService(
    PipelineStateMixin,
    LifecycleControllerMixin,
    OrchestratorCoreMixin,
    OrchestratorV2Mixin,
):
    """
    Lightweight facade for pipeline orchestration.
    All operational logic is implemented in the mixin classes.
    """

    @classmethod
    async def _run_cycle(cls, ctx: PipelineContext):
        """Route the cycle through V1, V2 scaffold, or benchmark mode."""
        import logging

        logger = logging.getLogger(__name__)

        mode = cls._state.get("execution_mode", "production")
        v2_stage = cls._state.get("v2_stage", 0)

        # ── Benchmark Mode ──
        if "benchmark" in mode:
            logger.info("[CYCLE] Dispatching to Benchmark Mode (%s)", mode)
            try:
                from app.pipeline.benchmark_runner import run_benchmark_cycle
            except ImportError:
                logger.error(
                    "[CYCLE] benchmark_runner module not found — benchmark mode is not yet implemented"
                )
                raise NotImplementedError(
                    "Benchmark mode is not yet implemented (app.pipeline.benchmark_runner missing)"
                )
            return await run_benchmark_cycle(
                ctx.tickers, cls._state.get("benchmark_group", "baseline"), ctx.cycle_id
            )

        # ── V2 Scaffold ──
        if mode == "v2_scaffold":
            logger.info("[CYCLE] Dispatching to V2 Scaffold (Stage %d)", v2_stage)
            return await cls.run_v2_cycle(ctx)

        # ── A/B Testing ──
        if mode == "ab_test":
            logger.info("[CYCLE] Dispatching to A/B Test Runner")
            return await cls.run_ab_cycle(ctx)

        # ── Canonical Pipeline ──
        logger.info("[CYCLE] Dispatching to Canonical Production Pipeline")
        return await cls._execute_cycle(ctx)


# Global singleton instance export to preserve compatibility
pipeline_service = PipelineService()

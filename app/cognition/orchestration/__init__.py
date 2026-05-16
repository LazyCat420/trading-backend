"""
V2 cognition orchestration metadata and stage-0 runtime shims.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.config import settings
from app.config.config_cognition import cognition_settings

V1_ENTRYPOINTS = [
    "app/services/pipeline_service.py",
    "app/pipeline/decision_engine.py",
    "app/pipeline/context_builder.py",
    "app/pipeline/post_cycle_hooks.py",
    "app/services/rlm_wrapper.py",
]

V2_ENTRYPOINTS = [
    "app/cognition/orchestration/__init__.py",
    "app/cognition/memory/reader.py",
    "app/cognition/memory/writer.py",
    "app/cognition/evidence/packet_builder.py",
    "app/cognition/debate/thesis_agent.py",
]

MODULE_STATUS = [
    {
        "path": "app/pipeline/decision_engine.py",
        "lane": "v1",
        "status": "active_production",
        "owner": "Dev 2 — V1 Stabilization",
    },
    {
        "path": "app/pipeline/context_builder.py",
        "lane": "v1",
        "status": "active_production",
        "owner": "Dev 2 — V1 Stabilization",
    },
    {
        "path": "app/pipeline/context_builder.py",
        "lane": "v2",
        "status": "experimental_unwired",
        "owner": "Dev 1 — Pipeline Architecture",
    },
    {
        "path": "app/services/trading_memory.py",
        "lane": "v1",
        "status": "active_production",
        "owner": "Dev 2 — V1 Stabilization",
    },
    {
        "path": "app/services/memory/retriever.py",
        "lane": "v2",
        "status": "experimental_unwired",
        "owner": "Dev 3 — V2 Integration",
    },
    {
        "path": "app/services/memory/briefing.py",
        "lane": "v2",
        "status": "experimental_unwired",
        "owner": "Dev 3 — V2 Integration",
    },
    {
        "path": "app/pipeline/post_cycle_observe.py",
        "lane": "v2",
        "status": "experimental_unwired",
        "owner": "Dev 3 — V2 Integration",
    },
    {
        "path": "app/pipeline/validate_memories.py",
        "lane": "v2",
        "status": "experimental_unwired",
        "owner": "Dev 3 — V2 Integration",
    },
    {
        "path": "app/pipeline/benchmark.py",
        "lane": "shared",
        "status": "shared_dependency",
        "owner": "Dev 4 — Benchmark Evaluation",
    },
    {
        "path": "app/services/pipeline_service.py",
        "lane": "shared",
        "status": "shared_dependency",
        "owner": "Dev 1 — Pipeline Architecture",
    },
]

BENCHMARK_FIELDS = [
    "cycle_id",
    "requested_version",
    "effective_version",
    "benchmark_group",
    "execution_mode",
    "v2_stage",
    "total_ms",
    "total_tokens",
    "status",
]

STAGE_LABELS = {
    0: "stage0_scaffold",
    1: "stage1_context_only",
    2: "stage2_evidence_retrieval",
    3: "stage3_debate_refinement",
    4: "stage4_memory_writeback",
    5: "stage5_full_cycle_benchmark",
}


def _normalize_version(version: str | None) -> str:
    raw_version = version
    if raw_version is None:
        raw_version = settings.PIPELINE_VERSION
    if raw_version is None:
        raw_version = "v2"
    normalized = raw_version.strip().lower()
    if normalized not in {"v1", "v2", "ab"}:
        return "v2"
    return normalized


def _derived_stage_from_flags() -> int:
    if cognition_settings.ENABLE_REFLECTIVE_MEMORY:
        return 4
    if cognition_settings.ENABLE_DEBATE_REFINEMENT:
        return 3
    if (
        cognition_settings.ENABLE_EVIDENCE_FUSION
        or cognition_settings.ENABLE_VERIFICATION_GATE
    ):
        return 2
    if cognition_settings.ENABLE_ONTOLOGY_GRAPH:
        return 1
    return 0


def get_v2_stage() -> int:
    configured = int(getattr(cognition_settings, "COGNITION_V2_STAGE", 0))
    return max(0, min(5, configured or _derived_stage_from_flags()))


def resolve_cycle_runtime(
    requested_version: str | None = None,
    benchmark_group: str | None = None,
) -> dict[str, Any]:
    requested = _normalize_version(requested_version)
    stage = get_v2_stage()
    cognition_enabled = bool(cognition_settings.ENABLE_COGNITION_V2)
    dedicated_v2_runner = cognition_enabled  # V2 runner is live

    if requested == "v2":
        if cognition_enabled:
            effective = "v2"
            execution_mode = (
                "v2_stage0_delegates_to_v1"
                if not dedicated_v2_runner
                else "v2_dedicated"
            )
        else:
            effective = "v1"
            execution_mode = "v2_disabled_fallback_to_v1"
    elif requested == "ab":
        effective = "v1"
        execution_mode = (
            "ab_shadow_v2_against_v1" if cognition_enabled else "ab_shadow_v2_disabled"
        )
    else:
        effective = "v2"
        execution_mode = "production"

    return {
        "requested_version": requested,
        "effective_version": effective,
        "benchmark_group": benchmark_group or settings.PIPELINE_BENCHMARK_GROUP,
        "execution_mode": execution_mode,
        "v2_stage": stage,
        "v2_stage_label": STAGE_LABELS.get(stage, STAGE_LABELS[0]),
        "cognition_enabled": cognition_enabled,
        "dedicated_v2_runner": dedicated_v2_runner,
        "shadow_version": "v2" if requested == "ab" else None,
    }


def get_architecture_snapshot() -> dict[str, Any]:
    route = resolve_cycle_runtime()
    return {
        "default_runtime": route,
        "runtime_entrypoints": {
            "v1": V1_ENTRYPOINTS,
            "v2": V2_ENTRYPOINTS,
        },
        "lanes": {
            "v1": {
                "description": "Production baseline kept bugfix-only.",
                "owner": "Dev 2 — V1 Stabilization",
            },
            "v2": {
                "description": "Cognition rollout behind explicit version routing.",
                "owner": "Dev 3 — V2 Integration",
            },
            "shared": {
                "description": (
                    "Collectors, DB, state, logging, and benchmark infrastructure."
                ),
                "owner": "Dev 1/4 — Architecture + Benchmark",
            },
        },
        "module_status": MODULE_STATUS,
        "benchmark_contract": BENCHMARK_FIELDS,
    }


async def run_v2_cycle(
    *,
    route: dict[str, Any],
    emit: Callable[..., Any],
    run_v1: Callable[[], Awaitable[Any]],
    # V2-specific args passed through from pipeline_service
    tickers: list[str] | None = None,
    cycle_id: str = "",
    bot_id: str = "",
    macro_memo: str = "",
    collect: bool = True,
    analyze: bool = True,
    trade: bool = True,
) -> Any:
    """Execute the V2 cognition pipeline with automatic V1 fallback.

    If the V2 runner throws any exception, we log the error and
    seamlessly fall back to the V1 runner so production never breaks.
    """
    import logging as _logging

    _logger = _logging.getLogger(__name__)

    if not route.get("dedicated_v2_runner"):
        # Cognition disabled — delegate to V1
        emit(
            "starting",
            "v2_route",
            f"V2 {route['v2_stage_label']} — cognition disabled, falling back to V1.",
            status="ok",
            data=route,
        )
        return await run_v1()

    emit(
        "starting",
        "v2_route",
        f"V2 {route['v2_stage_label']} — dedicated cognition pipeline active.",
        status="ok",
        data=route,
    )

    try:
        from app.cognition.orchestration.runner import execute_v2_tickers

        if tickers and analyze:
            v2_results = await execute_v2_tickers(
                tickers,
                cycle_id=cycle_id,
                bot_id=bot_id,
                emit=emit,
                macro_memo=macro_memo,
            )
            return v2_results
        else:
            # No tickers or analyze=False — fall through
            # to V1 for collection/trading only
            return await run_v1()
    except Exception as e:
        _logger.error("[V2] Pipeline FAILED — falling back to V1: %s", e, exc_info=True)
        emit(
            "analyzing",
            "v2_fallback",
            f"⚠️ V2 pipeline failed ({e}), falling back to V1",
            status="error",
        )
        return await run_v1()


async def run_ab_cycle(
    *,
    route: dict[str, Any],
    emit: Callable[..., Any],
    run_v1: Callable[[], Awaitable[Any]],
) -> Any:
    emit(
        "starting",
        "ab_route",
        "A/B benchmark mode — running V1 baseline"
        " and tagging the cycle for V2 comparison.",
        status="ok",
        data=route,
    )
    return await run_v1()

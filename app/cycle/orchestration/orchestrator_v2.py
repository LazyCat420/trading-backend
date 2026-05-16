"""
CANONICAL ENGINE: orchestrator_v2.py
This is the canonical Orchestrator moving forward. It implements the fully agentic V2 pipeline.
(V1 logic in orchestrator_v1.py is formally deprecated).
"""

import logging

logger = logging.getLogger(__name__)


class OrchestratorV2Mixin:
    @classmethod
    async def run_v2_cycle(cls, ticker: str, context: dict) -> dict:
        """
        Manually trigger the V2 sequential pipeline for a single ticker.
        (For Dev integration/smoke testing)
        """
        from app.cognition.ontology.ontology_builder import OntologyBuilder
        from app.cognition.evidence.packet_builder import build_evidence_packet
        from app.cognition.debate.thesis_agent import generate_thesis
        from app.cognition.memory.rlm_logger import ReflectiveMemoryLogger
        import uuid

        cycle_id = f"v2-smoke-{uuid.uuid4().hex[:6]}"
        ctx = {"cycle_id": cycle_id, "ticker": ticker, **context}

        # 1. Ontology
        ontology = await OntologyBuilder().execute(ticker, ctx)
        ctx.update({"ontology": ontology})

        # 2. Evidence
        packet = await build_evidence_packet(ticker, ctx)

        # Dynamic Retrieval Loop
        if packet.missing_fields:
            import logging

            logging.getLogger(__name__).info(
                f"Missing fields detected: {packet.missing_fields}. Triggering dynamic retrieval..."
            )
            from app.pipeline.analysis.dynamic_tool_router import resolve_missing_data

            fetched = await resolve_missing_data(ticker, packet.missing_fields)
            if fetched:
                packet = await build_evidence_packet(ticker, ctx)

        ctx.update({"packet": packet})

        # 3. Debate Thesis
        thesis, _tokens = await generate_thesis(
            ticker, packet, bias="neutral", cycle_id=cycle_id
        )
        # Convert ArgumentDraft to dict for logger
        thesis_dict = {
            "action": thesis.action,
            "confidence": thesis.confidence,
            "core_claims": thesis.core_claims,
            "weaknesses": thesis.weaknesses,
            "rationale": thesis.rationale,
        }

        # 4. Reflective Memory
        reflection = await ReflectiveMemoryLogger().execute(thesis_dict, ctx)

        return {
            "status": "success",
            "cycle_id": cycle_id,
            "ticker": ticker,
            "reflection": reflection,
        }

    @classmethod
    async def run_ab_cycle(cls, ticker: str, context: dict) -> dict:
        """
        Manually trigger an A/B benchmark run comparing V1 logic against V2 logic.
        """
        from app.pipeline.analysis.decision_engine import analyze_ticker
        from app.log_manager import log_manager
        import uuid

        cycle_id = f"ab-smoke-{uuid.uuid4().hex[:6]}"
        ctx = {"cycle_id": cycle_id, "ticker": ticker, **context}

        # Run V1
        v1_result = await analyze_ticker(
            ticker, cycle_id=cycle_id, bot_id="test", emit=cls.emit
        )

        # Run V2
        try:
            v2_result = await cls.run_v2_cycle(ticker, ctx)
        except Exception as e:
            v2_result = {"error": str(e)}

        ab_chosen = "v2"  # Normally an LLM judge decides; default to V2 for smoke test

        log_manager.log_ab_result(
            cycle_id=cycle_id,
            ab_chosen=ab_chosen,
            v1_result=v1_result,
            v2_result=v2_result,
            context=ctx,
        )

        return {
            "status": "success",
            "cycle_id": cycle_id,
            "ab_chosen": ab_chosen,
            "v1_result": v1_result,
            "v2_result": v2_result,
        }

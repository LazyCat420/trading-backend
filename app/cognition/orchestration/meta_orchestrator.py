import logging
import asyncio
from typing import Dict

from app.cognition.contracts.evidence import EvidencePacket
from app.cognition.verification.sufficiency_gate import SufficiencyResult
from app.cognition.debate.specialized_agents import (
    analyze_sentiment,
    analyze_macro_risk,
    analyze_fundamentals,
    analyze_deep_research,
)

logger = logging.getLogger(__name__)


class MetaOrchestrator:
    """
    Deterministically routes to specialized agents depending on the EvidencePacket
    and Data Sufficiency.
    """

    @staticmethod
    async def orchestrate(
        entity_id: str,
        packet: EvidencePacket,
        sufficiency: SufficiencyResult,
        cycle_id: str,
        bot_id: str,
        is_highly_redundant: bool = False,
    ) -> tuple[Dict[str, str], int]:
        """
        Rule-based router. Dispatches specialized sub-agents based on evidence health.
        Returns a dict of agent_name -> insight, and total tokens used.
        """
        tasks = []
        labels = []

        # Rule 1: If macro indicator is somewhat intact, check macro risk
        if "regime" not in packet.missing_fields:
            tasks.append(analyze_macro_risk(entity_id, packet, cycle_id, bot_id))
            labels.append("macro_risk")

        # Rule 1.5: Deep Dive for high redundancy
        if is_highly_redundant:
            logger.info(f"[{entity_id}] MetaOrchestrator: High redundancy detected. Spawning DeepResearchAgent.")
            tasks.append(analyze_deep_research(entity_id, packet, cycle_id, bot_id))
            labels.append("deep_research")

        # Rule 2: If sentiment features are intact, run sentiment agent
        if (
            "sentiment" not in packet.missing_fields
            and "news" not in packet.missing_fields
        ):
            tasks.append(analyze_sentiment(entity_id, packet, cycle_id, bot_id))
            labels.append("sentiment")

        # Rule 3: Run Fundamental agent if basic financials exist
        if (
            "fundamentals" not in packet.missing_fields
            and "pe_ratio" not in packet.missing_fields
        ):
            tasks.append(analyze_fundamentals(entity_id, packet, cycle_id, bot_id))
            labels.append("fundamentals")

        if not tasks:
            logger.info(
                f"[{entity_id}] MetaOrchestrator: Evidence too sparse for specialized agents."
            )
            return {}, 0

        # Execute selected agents in parallel with per-agent timeouts.
        # Each agent gets 20s — prevents one hung LLM call from blocking
        # the entire orchestration (which has a 60s outer timeout).
        PER_AGENT_TIMEOUT = 20.0
        logger.info(f"[{entity_id}] MetaOrchestrator: Dispatching {labels} (timeout={PER_AGENT_TIMEOUT}s each)")
        results = {}
        total_tokens = 0
        try:
            wrapped_tasks = [
                asyncio.wait_for(task, timeout=PER_AGENT_TIMEOUT)
                for task in tasks
            ]
            outputs = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
            for label, out in zip(labels, outputs):
                if isinstance(out, asyncio.TimeoutError):
                    logger.warning(
                        f"[{entity_id}] MetaOrchestrator: {label} TIMED OUT after {PER_AGENT_TIMEOUT}s"
                    )
                    results[label] = f"Error: Agent timed out after {PER_AGENT_TIMEOUT}s"
                elif isinstance(out, Exception):
                    logger.error(
                        f"[{entity_id}] MetaOrchestrator task {label} failed: {out}"
                    )
                    results[label] = f"Error: {out}"
                else:
                    results[label] = out[0]
                    total_tokens += out[1]
        except Exception as e:
            logger.error(f"[{entity_id}] MetaOrchestrator execution crashed: {e}")

        # ── Auto-post specialist findings to TaskBoard for agent collaboration ──
        # This enables debate agents and the thesis generator to read what
        # each specialist discovered, creating actual inter-agent collaboration.
        try:
            from app.agents.task_board import task_board

            for label, insight in results.items():
                if isinstance(insight, str) and not insight.startswith("Error:"):
                    # Truncate long insights to keep the TaskBoard manageable
                    snippet = insight[:500] if len(insight) > 500 else insight
                    await task_board.post_finding(
                        source_agent=f"{label}_agent",
                        content=snippet,
                        ticker=entity_id,
                        cycle_id=cycle_id,
                        category="fact",
                        confidence=75,
                    )
            if results:
                logger.info(
                    "[%s] MetaOrchestrator: Posted %d specialist findings to TaskBoard",
                    entity_id, len([v for v in results.values() if not str(v).startswith("Error:")]),
                )
        except Exception as tb_err:
            logger.warning(
                "[%s] MetaOrchestrator: TaskBoard post failed (non-fatal): %s",
                entity_id, tb_err,
            )

        return results, total_tokens


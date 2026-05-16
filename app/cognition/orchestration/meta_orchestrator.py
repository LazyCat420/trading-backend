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

        # Execute selected agents in parallel
        logger.info(f"[{entity_id}] MetaOrchestrator: Dispatching {labels}")
        results = {}
        total_tokens = 0
        try:
            outputs = await asyncio.gather(*tasks, return_exceptions=True)
            for label, out in zip(labels, outputs):
                if isinstance(out, Exception):
                    logger.error(
                        f"[{entity_id}] MetaOrchestrator task {label} failed: {out}"
                    )
                    results[label] = f"Error: {out}"
                else:
                    results[label] = out[0]
                    total_tokens += out[1]
        except Exception as e:
            logger.error(f"[{entity_id}] MetaOrchestrator execution crashed: {e}")

        return results, total_tokens

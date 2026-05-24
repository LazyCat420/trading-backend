"""
Data Sufficiency Gate.
Ensures we have enough data before allowing reasoning to start.
"""

import logging
from typing import List
from ..contracts.evidence import EvidencePacket
from ..contracts.verification import SufficiencyResult, SufficiencyStatus

logger = logging.getLogger(__name__)


def check_data_sufficiency(entity_id: str, packet: EvidencePacket) -> SufficiencyResult:
    """
    Checks if an EvidencePacket meets the minimum requirements for analysis.
    """
    blockers: List[str] = []
    warnings: List[str] = []

    # 1. Missing Critical Data
    if "price" in packet.missing_fields:
        blockers.append("Missing critical price history data.")

    if "pe_ratio" in packet.missing_fields and entity_id.upper() not in [
        "BTC",
        "ETH",
        "SOL",
        "BNB",
        "XRP",
        "DOGE",
        "ADA",
        "AVAX",
    ]:
        warnings.append("Missing fundamental P/E Ratio data.")

    # 2. Low Diversity
    sq = packet.source_quality_summary
    if sq:
        if sq.source_diversity < 2:
            warnings.append(
                f"Low source diversity ({sq.source_diversity}). Only seeing limited perspectives."
            )

        # 3. Teaser Risk — WARNING only, never blocks analysis.
        #    The scorer now excludes structured docs, but even if teaser risk
        #    is high among unstructured sources, it shouldn't prevent a decision.
        if sq.teaser_artifact_risk > 0.8:
            warnings.append(
                f"High teaser/promo risk ({sq.teaser_artifact_risk * 100:.0f}%) in unstructured sources."
            )

        # 4. Stale Data & Cached Fallbacks
        if packet.freshness_summary and (packet.freshness_summary.is_stale or packet.freshness_summary.newest_data_age_hours > 24.0):
            warnings.append(
                f"Stale data/cached fallbacks in use. Newest data is {packet.freshness_summary.newest_data_age_hours:.1f} hours old."
            )

    if not packet.claims:
        blockers.append("No factual claims could be extracted.")

    status: SufficiencyStatus = "sufficient"
    if blockers:
        status = "critical_gap"
    elif warnings:
        status = "insufficient"  # Needs refinement

    logger.info(
        "[SUFFICIENCY] %s → %s | claims=%d, facts=%d, missing=%s, blockers=%s, warnings=%d",
        entity_id,
        status,
        len(packet.claims),
        len(packet.structured_facts),
        packet.missing_fields or "none",
        blockers or "none",
        len(warnings),
    )

    return SufficiencyResult(status=status, blockers=blockers, warnings=warnings)

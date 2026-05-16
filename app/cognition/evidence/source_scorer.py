"""
Source Scoring Logic.
Computes diversity, freshness, and structural confidence.
"""

from typing import List
from datetime import datetime, timezone
from ..contracts.evidence import SourceQuality
from ..contracts.claims import Claim
from ..contracts.retrieval import ContradictionRef
from .normalizer import NormalizedDocument


def score_sources(
    documents: List[NormalizedDocument],
    claims: List[Claim],
    contradictions: List[ContradictionRef],
    missing_fields: List[str],
) -> SourceQuality:
    """
    Computes packet-level scoring (Dev 2 single source of truth for scores).
    """
    now = datetime.now(timezone.utc)

    # 1. Source Diversity
    # Count distinct source types (news, reddit, youtube, structured)
    source_types = set(d.source_type for d in documents)
    source_diversity = len(source_types)

    # 2. Freshness Score & 6. Stale Data Severity
    total_freshness = 0.0
    max_age_hours = 0.0

    for doc in documents:
        if doc.timestamp is None:
            doc_ts = now
        elif isinstance(doc.timestamp, str):
            try:
                doc_ts = datetime.fromisoformat(doc.timestamp.replace("Z", "+00:00"))
            except ValueError:
                doc_ts = now
        else:
            doc_ts = (
                doc.timestamp.replace(tzinfo=timezone.utc)
                if getattr(doc.timestamp, "tzinfo", None) is None
                else doc.timestamp
            )
        age_delta = now - doc_ts
        age_hours = age_delta.total_seconds() / 3600.0

        if age_hours > max_age_hours:
            max_age_hours = age_hours

        # Exponential decay (24h half-life)
        # e^(-ln(2) * age / 24)
        doc_freshness = pow(0.5, age_hours / 24.0)
        total_freshness += doc_freshness

    avg_freshness = total_freshness / len(documents) if documents else 0.0

    # 3. Contradiction count
    contradiction_count = len(contradictions)

    # 4. Numeric Completeness
    expected_fields = {"price", "market_cap", "pe_ratio", "revenue"}
    present_fields = expected_fields - set(missing_fields)
    numeric_completeness = (
        len(present_fields) / len(expected_fields) if expected_fields else 1.0
    )

    # 5. Entity Confidence
    # Average of claim confidences, penalized by contradictions
    avg_claim_conf = sum(c.confidence for c in claims) / len(claims) if claims else 0.0
    entity_confidence = max(0.0, avg_claim_conf - (0.1 * contradiction_count))
    if max_age_hours > 72.0:
        entity_confidence *= 0.8  # Penalize for stale data

    # 7. Teaser Risk (Promo content probability)
    # Only evaluate UNSTRUCTURED documents (news, reddit, youtube).
    # Structured facts (prices, ratios) are naturally short numeric strings
    # and must NOT be counted as teaser/promo content.
    unstructured_docs = [d for d in documents if d.source_type != "structured"]
    teaser_docs = sum(
        1
        for d in unstructured_docs
        if "SUBSCRIBE" in d.content.upper() or len(d.content.strip()) < 50
    )
    teaser_artifact_risk = (
        teaser_docs / len(unstructured_docs) if unstructured_docs else 0.0
    )

    return SourceQuality(
        source_diversity=source_diversity,
        freshness_score=avg_freshness,
        contradiction_count=contradiction_count,
        numeric_completeness=numeric_completeness,
        entity_confidence=entity_confidence,
        stale_data_severity=max_age_hours,
        teaser_artifact_risk=teaser_artifact_risk,
    )

"""
Cross-source claim clustering.
Groups claims by predicate similarity and aggregates confidence.
"""

from typing import List, Dict
from dataclasses import dataclass
from ..contracts.claims import Claim


@dataclass
class ClaimCluster:
    topic_hash: str
    claims: List[Claim]
    consensus_value: str
    aggregated_confidence: float
    is_contradicted: bool = False


def cluster_claims(claims: List[Claim]) -> List[ClaimCluster]:
    """
    Group claims that assert things about the same predicate/topic.
    """
    groups: Dict[str, List[Claim]] = {}

    for claim in claims:
        # Simple deterministic hash: entity + predicate
        # In a real system, you might normalize predicates (e.g., 'price_target' vs 'target_price')
        topic_hash = f"{claim.subject_entity_id}::{claim.predicate}"
        if topic_hash not in groups:
            groups[topic_hash] = []
        groups[topic_hash].append(claim)

    clusters = []
    for topic_hash, grouped_claims in groups.items():
        # Heuristic for consensus: most common value
        values = [c.object_value for c in grouped_claims]
        consensus_value = max(set(values), key=values.count) if values else ""

        # Base confidence calculation
        max_conf = max((c.confidence for c in grouped_claims), default=0.0)

        # Boost confidence if corroborated by multiple sources
        unique_sources = len(set(sid for c in grouped_claims for sid in c.source_ids))
        boost = 0.05 * (unique_sources - 1)
        agg_conf = min(1.0, max_conf + boost)

        clusters.append(
            ClaimCluster(
                topic_hash=topic_hash,
                claims=grouped_claims,
                consensus_value=consensus_value,
                aggregated_confidence=agg_conf,
            )
        )

    return clusters

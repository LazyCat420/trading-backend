"""
Contradiction Detection.
Analyzes claim clusters for internal disagreement.
"""

from typing import List
from ..contracts.retrieval import ContradictionRef
from .clustering import ClaimCluster


def detect_contradictions(clusters: List[ClaimCluster]) -> List[ContradictionRef]:
    """
    Detects contradictions within formed clusters.
    """
    contradictions = []

    for cluster in clusters:
        values = [c.object_value.upper() for c in cluster.claims]
        if not values:
            continue

        # 1. Directional contradictions (e.g. sentiment: BULLISH vs BEARISH)
        if cluster.claims[0].predicate == "sentiment":
            has_bullish = "BULLISH" in values
            has_bearish = "BEARISH" in values
            if has_bullish and has_bearish:
                cluster.is_contradicted = True
                claims_bullish = [
                    c for c in cluster.claims if c.object_value.upper() == "BULLISH"
                ]
                claims_bearish = [
                    c for c in cluster.claims if c.object_value.upper() == "BEARISH"
                ]
                contradictions.append(
                    ContradictionRef(
                        description="Conflicting sentiment detected for entity.",
                        source_ref_1=claims_bullish[0].source_ids[0],
                        source_ref_2=claims_bearish[0].source_ids[0],
                        severity="warning",
                    )
                )

        # 2. Numeric contradictions (e.g., Target $50 vs Target $200)
        elif cluster.claims[0].predicate == "price_target":
            try:
                numeric_values = [(float(c.object_value), c) for c in cluster.claims]
                if len(numeric_values) >= 2:
                    min_val_tup = min(numeric_values, key=lambda x: x[0])
                    max_val_tup = max(numeric_values, key=lambda x: x[0])

                    # More than 2x divergence is a strict contradiction
                    if min_val_tup[0] > 0 and (max_val_tup[0] / min_val_tup[0]) > 2.0:
                        cluster.is_contradicted = True
                        contradictions.append(
                            ContradictionRef(
                                description=f"Price targets severely diverge: {min_val_tup[0]} vs {max_val_tup[0]}",
                                source_ref_1=min_val_tup[1].source_ids[0],
                                source_ref_2=max_val_tup[1].source_ids[0],
                                severity="warning",  # or blocker based on rules
                            )
                        )
            except ValueError:
                pass  # Not parseable as float

    return contradictions

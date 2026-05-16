"""
Deterministic claim extraction logic (Stage 1).
"""

import uuid
import re
from typing import List
from ..contracts.claims import Claim, Provenance
from .normalizer import NormalizedDocument


def extract_claims(doc: NormalizedDocument, entity_id: str) -> List[Claim]:
    """
    Extract deterministic claims from a normalized document.
    """
    claims = []

    if doc.source_type == "structured":
        claims.append(
            Claim(
                id=str(uuid.uuid4()),
                subject_entity_id=entity_id,
                predicate=doc.metadata.get("fact_type", "has_value"),
                object_value=str(doc.content),
                claim_type="fact",
                origin="deterministic",
                source_ids=[doc.source_ref],
                timestamp=doc.timestamp,
                confidence=0.95,  # Structural deterministic facts are very confident
                freshness_score=1.0,  # Freshness is scored separately later, starting max
                provenance=Provenance(
                    source_table=doc.source_type,
                    source_id=doc.source_ref,
                    extraction_method="deterministic_structured_mapping",
                    author=doc.author,
                ),
            )
        )
        return claims

    if doc.source_type in ("news", "reddit", "youtube"):
        # Very simple deterministic template extractions as placeholders
        content_upper = doc.content.upper()

        # Determine bullish/bearish simple mentions
        if (
            "BEAT EARNINGS" in content_upper
            or "BULLISH" in content_upper
            or "REVENUE INCREASED" in content_upper
        ):
            claims.append(
                Claim(
                    id=str(uuid.uuid4()),
                    subject_entity_id=entity_id,
                    predicate="sentiment",
                    object_value="bullish",
                    claim_type="inference",
                    origin="deterministic",
                    source_ids=[doc.source_ref],
                    timestamp=doc.timestamp,
                    confidence=0.7,
                    freshness_score=1.0,
                    provenance=Provenance(
                        source_table=doc.source_type,
                        source_id=doc.source_ref,
                        extraction_method="deterministic_regex_sentiment",
                        author=doc.author,
                    ),
                )
            )
        elif (
            "MISSED EARNINGS" in content_upper
            or "BEARISH" in content_upper
            or "REVENUE DECREASED" in content_upper
        ):
            claims.append(
                Claim(
                    id=str(uuid.uuid4()),
                    subject_entity_id=entity_id,
                    predicate="sentiment",
                    object_value="bearish",
                    claim_type="inference",
                    origin="deterministic",
                    source_ids=[doc.source_ref],
                    timestamp=doc.timestamp,
                    confidence=0.7,
                    freshness_score=1.0,
                    provenance=Provenance(
                        source_table=doc.source_type,
                        source_id=doc.source_ref,
                        extraction_method="deterministic_regex_sentiment",
                        author=doc.author,
                    ),
                )
            )

        # Price target extraction placeholder
        pt_match = re.search(
            r"PRICE TARGET\s*(?:OF|TO|AT)?\s*\$?(\d+(?:\.\d+)?)", content_upper
        )
        if pt_match:
            claims.append(
                Claim(
                    id=str(uuid.uuid4()),
                    subject_entity_id=entity_id,
                    predicate="price_target",
                    object_value=pt_match.group(1),
                    claim_type="forecast",
                    origin="deterministic",
                    source_ids=[doc.source_ref],
                    timestamp=doc.timestamp,
                    confidence=0.8,
                    freshness_score=1.0,
                    provenance=Provenance(
                        source_table=doc.source_type,
                        source_id=doc.source_ref,
                        extraction_method="deterministic_regex_price_target",
                        author=doc.author,
                    ),
                )
            )

    return claims

"""
LLM Claim Enrichment logic (Stage 2).
Runs only if `ENABLE_LLM_CLAIM_ENRICHMENT` is true.
"""

import logging
from typing import List
from ..contracts.claims import Claim
from .normalizer import NormalizedDocument

logger = logging.getLogger(__name__)


async def enrich_claims_with_llm(
    doc: NormalizedDocument, entity_id: str, existing_claims: List[Claim]
) -> List[Claim]:
    """
    Optional LLM pass to extract implicit causal relations and thesis summaries.
    This is behind the ENABLE_LLM_CLAIM_ENRICHMENT flag.
    Returns new claims to append.
    """
    from app.config.config_cognition import cognition_settings

    if not cognition_settings.ENABLE_LLM_CLAIM_ENRICHMENT:
        return []

    # In a full implementation, this uses vllm_client to run:
    # "Identify implicit relationships or downstream consequences from the following text..."

    # Placeholder: currently returns empty. Will implement full LLM call if needed.
    # LLM queries should focus on:
    # - causal claims (X caused Y)
    # - implied impacts
    logger.debug(f"[LLM Enrichment] Skipped for {doc.source_ref} (stubbed)")
    return []

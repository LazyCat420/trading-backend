"""
Evidence packet definitions for the Cognition V2 architecture.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from .claims import Claim
from .retrieval import StructuredFact, SourceDocRef, ContradictionRef, FreshnessSummary


class SourceQuality(BaseModel):
    source_diversity: int
    freshness_score: float
    contradiction_count: int
    numeric_completeness: float
    entity_confidence: float
    stale_data_severity: float
    teaser_artifact_risk: float


class EvidencePacket(BaseModel):
    """
    The fused, deduplicated, and scored bundle of evidence ready for verification.
    """

    entity_id: str
    claims: List[Claim] = Field(default_factory=list)
    structured_facts: List[StructuredFact] = Field(default_factory=list)
    source_summaries: List[SourceDocRef] = Field(default_factory=list)
    contradictions: List[ContradictionRef] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    tool_cache: dict = Field(default_factory=dict)
    freshness_summary: Optional[FreshnessSummary] = None
    source_quality_summary: Optional[SourceQuality] = None
    company_story: Optional[str] = None
    key_themes: Optional[List[dict]] = Field(default_factory=list)
    pillar_profiles: Optional[dict] = None

    class Config:
        frozen = True

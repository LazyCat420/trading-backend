"""
Claim definitions for the Cognition V2 architecture.
"""

from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime


class Provenance(BaseModel):
    source_table: str
    source_id: str
    extraction_method: str
    author: Optional[str] = None

    class Config:
        frozen = True


ClaimOrigin = Literal["deterministic", "llm_enriched", "llm_inferred", "human_defined"]
ClaimType = Literal["fact", "inference", "forecast"]


class Claim(BaseModel):
    """
    A single factual assertion or inference extracted from evidence.
    """

    id: str
    subject_entity_id: str
    predicate: str
    object_value: str
    claim_type: ClaimType
    origin: ClaimOrigin
    source_ids: List[str]
    timestamp: datetime
    confidence: float
    freshness_score: float
    provenance: Provenance

    class Config:
        frozen = True

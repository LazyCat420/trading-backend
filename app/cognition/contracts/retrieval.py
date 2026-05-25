"""
Retrieval context definitions for the Cognition V2 architecture.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


class SemanticHit(BaseModel):
    content: str
    score: float
    source_ref: str


class GraphHit(BaseModel):
    subject: str
    predicate: str
    object_val: str
    source_ref: Optional[str] = None


class StructuredFact(BaseModel):
    fact_type: str
    value: Any
    timestamp: datetime


class SourceDocRef(BaseModel):
    source_type: str
    source_id: str
    summary: str
    timestamp: Optional[datetime] = None
    url: Optional[str] = None
    metadata: Optional[dict] = None


class ContradictionRef(BaseModel):
    description: str
    source_ref_1: str
    source_ref_2: str
    severity: str


class FreshnessSummary(BaseModel):
    oldest_data_age_hours: float
    newest_data_age_hours: float
    is_stale: bool
    oldest_timestamp: Optional[datetime] = None
    newest_timestamp: Optional[datetime] = None


class RetrievalContext(BaseModel):
    """
    The aggregated raw information gathered for a given entity.
    Populated by the graph and retrieval layers, consumed by evidence and debate.
    """

    entity_id: str
    query_time: datetime
    semantic_hits: List[SemanticHit] = Field(default_factory=list)
    graph_hits: List[GraphHit] = Field(default_factory=list)
    structured_facts: List[StructuredFact] = Field(default_factory=list)
    source_docs: List[SourceDocRef] = Field(default_factory=list)
    contradictions: List[ContradictionRef] = Field(default_factory=list)
    freshness_summary: Optional[FreshnessSummary] = None

    class Config:
        frozen = True

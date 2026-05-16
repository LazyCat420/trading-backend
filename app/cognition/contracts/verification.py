"""
Verification gate definitions for the Cognition V2 architecture.
"""

from pydantic import BaseModel, Field
from typing import List, Literal

VerificationStatus = Literal["pass", "refine", "recollect", "abstain"]
SufficiencyStatus = Literal["sufficient", "insufficient", "critical_gap"]


class VerificationReport(BaseModel):
    """
    The output of the verification gate, assessing a hypothesis against the evidence.
    """

    status: VerificationStatus
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    verified_claims: List[str] = Field(default_factory=list)
    unverified_claims: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    missing_critical_data: List[str] = Field(default_factory=list)
    score: float

    class Config:
        frozen = True


class SufficiencyResult(BaseModel):
    """
    The output of the data sufficiency gate, verifying the packet has enough
    breadth and freshness before allowing analysis to proceed.
    """

    status: SufficiencyStatus
    blockers: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    class Config:
        frozen = True

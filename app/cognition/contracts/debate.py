"""
Debate definitions for the Cognition V2 architecture.
"""

from pydantic import BaseModel, Field
from typing import List


class HypothesisDraft(BaseModel):
    """
    A proposed trading thesis or analysis conclusion waiting for verification or debate.
    """

    thesis: str
    supporting_claims: List[str] = Field(default_factory=list)
    action: str

    class Config:
        frozen = True


class ThesisDraft(BaseModel):
    action: str
    confidence: int
    core_claims: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    rationale: str
    iteration: int = 0


class ArgumentDraft(BaseModel):
    persona: str
    argument: str
    countered_claims: List[str] = Field(default_factory=list)

    class Config:
        frozen = True


class DecisionDraft(BaseModel):
    final_action: str
    confidence: float
    rationale: str

    class Config:
        frozen = True


class DebateResult(BaseModel):
    """Output of the full adversarial debate pipeline.

    Contains bull/bear claims (raw and verified), cross-examination
    findings, and the judge's final weighted verdict.
    """

    bull_claims: List[dict] = Field(default_factory=list)
    bear_claims: List[dict] = Field(default_factory=list)
    verified_bull_claims: List[dict] = Field(default_factory=list)
    verified_bear_claims: List[dict] = Field(default_factory=list)
    unverified_claims: List[dict] = Field(default_factory=list)
    cross_exam_findings: str = ""
    judge_action: str = "HOLD"
    judge_confidence: int = 0
    judge_rationale: str = ""
    winning_side: str = "split"  # "bull" | "bear" | "split"
    key_deciding_factor: str = ""
    rejected_claim_impact: str = ""
    integrity_status: str = "HIGH"  # "HIGH" | "LOW_INTEGRITY"
    transcript: str = ""
    total_tokens: int = 0
    persona_outcomes: dict = Field(default_factory=dict)
    minority_report: str = ""

    class Config:
        frozen = True

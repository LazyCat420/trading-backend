"""Pydantic schemas for the ASI-Evolve strategy evolution system."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class EvolutionMetrics(BaseModel):
    """Structured backtest metrics returned by the sandbox executor."""

    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0


class EvolutionNode(BaseModel):
    """A single node in the strategy evolution tree."""

    id: str = Field(..., description="UUID4 primary key")
    session_id: str = Field(..., description="Date-based tag e.g. 'apr05'")
    round: int = Field(..., description="Round number within the session")
    parent_id: Optional[str] = Field(
        None, description="ID of the parent node this was derived from"
    )
    motivation: str = Field("", description="Why the agent tried this strategy")
    code: str = Field("", description="Full strategy_candidate.py content")
    metrics: Optional[EvolutionMetrics] = None
    score: Optional[float] = Field(None, description="Primary scalar — Sharpe ratio")
    status: str = Field(
        "DISCARD", description="KEEP | DISCARD | SYNTAX_ERROR | RUNTIME_ERROR | TIMEOUT"
    )
    analysis: str = Field("", description="LLM-generated lesson text")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EvolutionSessionSummary(BaseModel):
    """Summary of an evolution session."""

    session_id: str
    total_rounds: int = 0
    best_score: Optional[float] = None
    best_node_id: Optional[str] = None
    kept_count: int = 0
    discarded_count: int = 0
    error_count: int = 0
    timeout_count: int = 0

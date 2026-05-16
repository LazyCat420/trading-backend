"""
Cognition V2 — Memory Models.

Dataclass definitions for the four memory types:
  - SemanticMemory  — stable facts, relations, domain knowledge
  - EpisodicMemory  — cycle inputs, evidence, debate outcomes, decisions
  - ReflectiveMemory — postmortems, failure patterns, missed signals
  - ProceduralMemory — process rules, heuristics, mitigation patterns

All memories are wrapped in a MemoryEnvelope for unified storage/retrieval.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Discriminator for memory classification."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    REFLECTIVE = "reflective"
    PROCEDURAL = "procedural"


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory record."""

    ACTIVE = "active"
    TENTATIVE = "tentative"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


# ── Semantic Memory ────────────────────────────────────────────────
@dataclass
class SemanticFact:
    """A single stable fact about an entity or domain concept."""

    subject: str  # e.g. "NVDA"
    predicate: str  # e.g. "sector"
    object: str  # e.g. "Technology"
    source: str = ""  # provenance
    confidence: float = 1.0


@dataclass
class SemanticMemory:
    """Stable entity facts, graph relations, and long-term domain knowledge.

    Examples:
      - "NVDA is in the Technology sector"
      - "NVDA's primary revenue driver is data center GPUs"
      - "Oil prices correlate inversely with airline profitability"
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = ""
    facts: list[SemanticFact] = field(default_factory=list)
    domain_knowledge: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.8
    evidence_count: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Episodic Memory ───────────────────────────────────────────────
@dataclass
class EpisodicMemory:
    """Records of specific cognition cycle events.

    Captures cycle inputs, evidence packets, debate outcomes,
    decisions, and timestamps for replay and analysis.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = ""
    cycle_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # What happened
    event_type: str = ""  # "decision", "verification", "debate", "outcome"
    action: str = ""  # BUY/SELL/HOLD
    confidence: float = 0.0

    # Evidence snapshot
    evidence_summary: str = ""
    evidence_sources: list[str] = field(default_factory=list)

    # Debate outcome (if applicable)
    debate_thesis_won: bool | None = None
    debate_persona: str = ""
    debate_challenges: list[str] = field(default_factory=list)

    # Outcome (filled post-resolution)
    outcome_label: str = ""  # WIN/LOSS/FLAT
    outcome_pnl_pct: float | None = None

    # Context
    context_summary: str = ""
    tags: list[str] = field(default_factory=list)


# ── Reflective Memory ─────────────────────────────────────────────
@dataclass
class ReflectionRecord:
    """Structured postmortem analysis of an episode.

    Answers: What went wrong? What was missed? What should have blocked this?
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str = ""
    entity_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Analysis
    failure_patterns: list[str] = field(default_factory=list)
    missed_signals: list[str] = field(default_factory=list)
    late_verifiers: list[str] = field(default_factory=list)
    what_should_have_blocked: str = ""
    root_cause: str = ""

    # Recommendations
    recommended_changes: list[str] = field(default_factory=list)
    severity: str = "medium"  # low/medium/high/critical

    # Outcome context
    expected_outcome: str = ""
    actual_outcome: str = ""
    outcome_delta: float = 0.0


@dataclass
class ReflectiveMemory:
    """Collection of reflections that form institutional learning.

    Aggregates postmortems and failure patterns into reusable
    knowledge about what goes wrong and why.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = ""
    pattern_name: str = ""
    description: str = ""
    frequency: int = 1  # how often this pattern has been observed
    reflections: list[str] = field(default_factory=list)  # reflection_ids
    tags: list[str] = field(default_factory=list)
    severity: str = "medium"
    confidence: float = 0.5
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Procedural Memory ─────────────────────────────────────────────
@dataclass
class ProceduralRule:
    """A process rule or heuristic derived from experience.

    Examples:
      - "Always check 13F filings before buying semiconductor stocks"
      - "If macro memo warns of rate hike, reduce position sizes by 50%"
      - "YouTube sentiment below -0.3 on earnings day is noise, not signal"
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_text: str = ""
    category: str = ""  # "data_collection", "analysis", "trading", "risk"
    applies_to: list[str] = field(default_factory=list)  # tickers/sectors/all
    confidence: float = 0.5
    success_count: int = 0
    failure_count: int = 0
    source_episodes: list[str] = field(default_factory=list)  # episode_ids
    tags: list[str] = field(default_factory=list)
    active: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Memory Envelope ────────────────────────────────────────────────
@dataclass
class MemoryEnvelope:
    """Unified wrapper for all memory types.

    Provides a consistent interface for storage, retrieval, and lifecycle
    management regardless of the underlying memory type.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.SEMANTIC
    entity_id: str = ""
    content_hash: str = ""  # for idempotency checks
    status: MemoryStatus = MemoryStatus.ACTIVE
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)
    ttl_days: int | None = None  # None = no expiry

    # The actual memory payload (serialized)
    payload: dict[str, Any] = field(default_factory=dict)

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_accessed_at: str | None = None

    @staticmethod
    def compute_content_hash(memory_type: str, content: str) -> str:
        """Deterministic hash for idempotency checks."""
        raw = f"{memory_type}:{content}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

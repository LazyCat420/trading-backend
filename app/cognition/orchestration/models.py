"""
Cognition V2 — Orchestration Models.

Data structures for V2 pipeline run results.
Used by memory/writer.py to persist episodic memories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CognitionRunResult:
    """Output of a complete V2 cognition pipeline run.

    Shape is consumed by:
      - memory/writer.py (write_episode)
      - orchestration/runner.py (build final result)
      - pipeline_service.py (convert to V1-compatible dict)
    """

    entity_id: str = ""
    cycle_id: str = ""
    final_action: str = "HOLD"
    final_confidence: float = 0.0
    summary: str = ""
    rationale: str = ""
    tags: list[str] = field(default_factory=list)

    # Pipeline artifacts
    evidence_packet: Any = None  # EvidencePacket (avoid circular import)
    thesis: Any = None  # ThesisDraft
    sufficiency: Any = None  # SufficiencyResult
    memory_context: dict[str, Any] = field(default_factory=dict)

    # Telemetry
    total_tokens: int = 0
    total_ms: int = 0
    stages_completed: list[str] = field(default_factory=list)
    retrieval_retries: int = 0
    fell_back_to_v1: bool = False

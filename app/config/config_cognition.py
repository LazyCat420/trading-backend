"""
Configuration flags for the Cognition V2 multi-agent architecture.
Controls the staged rollout of new v2 components alongside the legacy pipeline.
"""

from pydantic_settings import BaseSettings


class CognitionSettings(BaseSettings):
    # Core V2 toggles
    ENABLE_COGNITION_V2: bool = False
    COGNITION_V2_STAGE: int = 4

    # Layer 1: Ontology & Graph (Dev 1)
    ENABLE_ONTOLOGY_GRAPH: bool = True

    # Layer 2: Evidence Fusion & Verification (Dev 2)
    ENABLE_EVIDENCE_FUSION: bool = True
    ENABLE_VERIFICATION_GATE: bool = True

    # Layer 3: Debate & Adjudication (Dev 3)
    ENABLE_DEBATE_REFINEMENT: bool = True
    DEBATE_ENABLED: bool = True  # toggle adversarial bull/bear debate
    DEBATE_MAX_TOOL_TURNS: int = 1  # max tool-calling turns per debate agent (reduced from 3 — agents already have evidence packet)
    CLAIM_REJECT_THRESHOLD: int = 8  # max unverified claims before LOW_INTEGRITY (3 personas × 4 turns = 24 agent turns)
    FAST_DEBATE_MODE: bool = True  # Halve debate latency with capped prompt sizes
    MAX_DEBATE_HISTORY_AGE_HOURS: int = 4  # Don't use debates older than this for context
    CONFIRMATION_LOOP_THRESHOLD: int = 3  # Force skepticism if N+ consecutive same verdicts

    # Layer 4: Reflective Memory (Dev 5)
    ENABLE_REFLECTIVE_MEMORY: bool = True

    # Specific Feature Flags for Dev 2
    ENABLE_LLM_CLAIM_ENRICHMENT: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


cognition_settings = CognitionSettings()

# Static Data - Not overridable via environment variables
LLM_TEMPERATURES = {
    "thesis_generation": 0.5,
    "debate": 0.7,
    "creative": 0.8,
    "factual": 0.0,
    # Adversarial debate agents
    "bull_agent": 0.4,
    "bear_agent": 0.4,
    "cross_examiner": 0.2,
    "debate_judge": 0.2,
    "thesis_synthesis": 0.3,
}

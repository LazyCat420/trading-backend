"""
Context Budget Registry — Model-aware token budgeting for prompt assembly.

Maps each model to its *effective* context budget — the portion of the raw
context window where the model maintains high-quality attention.  Research
shows open-weight models degrade significantly past 50–60% of their raw
``max_model_len`` (the "Lost in the Middle" problem).

SCOPE:
    This budget system ONLY governs the **autonomous trading pipeline**:
      - agent_loop.py (tool-calling agents)
      - base_agent.py (analysis agents)
      - debate_coordinator.py (bull/bear debates)
      - war_context_builder.py (geopolitical context)
      - context_compressor.py (history compression)

    It does NOT affect:
      - Hermes Gateway (app/routers/hermes.py) — passes raw messages through
      - Strategy Chat (app/routers/collaboration.py) — calls llm.chat() directly
      - Any external consumer that hits the vLLM /v1/chat/completions API

    The vLLM servers still serve their full ``--max-model-len`` (e.g. 64K, 128K).
    This module only constrains how much context the *trading pipeline* chooses
    to assemble, preventing the autonomous agents from degrading in quality.

Usage:
    from app.config.context_budget import get_context_budget, ContextBudget
    budget = get_context_budget("google/gemma-4-26B-A4B-it")
    if len(my_data_block) > budget.data_context_chars:
        my_data_block = my_data_block[:budget.data_context_chars]
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Rough heuristic: 1 token ≈ 4 characters for English text.
# Used to convert between token budgets and character budgets.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ContextBudget:
    """Immutable context budget for a model/endpoint.

    All ``*_tokens`` fields are in *estimated* tokens.
    Corresponding ``*_chars`` properties convert via CHARS_PER_TOKEN.
    """

    model_id: str
    raw_context_tokens: int          # Model's max_model_len from vLLM
    effective_context_tokens: int    # Usable portion (≈50–60% of raw)

    # ── Per-slice budgets (in tokens) ────────────────────────────
    # Calibrated from empirical DB analysis (2026-05-05):
    #   - 57.7% of real LLM calls use 8K-12K prompt tokens
    #   - Debate T3/T4 agents spike to 15-21K at P99
    #   - YouTube transcripts P95 = 58K chars (14K tokens)
    #   - RAG already capped at 3K chars by retriever.py
    #   - Trading memory total = 762 bytes; lessons avg 137 chars
    system_prompt_budget: int = 2000
    data_context_budget: int = 6000  # Market data, technicals, fundamentals (40%)
    tool_result_budget: int = 3200   # Per individual tool result (20%)
    rag_budget: int = 1500           # RAG/memory injection (8%)
    memory_budget: int = 1000        # Trading memory + canonical memory (5%)
    history_budget: int = 3000       # Conversation history before compression (12%)
    war_context_budget: int = 600    # Geopolitical intel block (3%)
    capsule_budget: int = 400        # AgentCapsule stack — already capped (2%)

    @property
    def compressor_threshold(self) -> int:
        """Trigger compression at 75% of effective context."""
        return int(self.effective_context_tokens * 0.75)

    # ── Character-based convenience properties ──────────────────
    @property
    def data_context_chars(self) -> int:
        return self.data_context_budget * CHARS_PER_TOKEN

    @property
    def tool_result_chars(self) -> int:
        return self.tool_result_budget * CHARS_PER_TOKEN

    @property
    def rag_chars(self) -> int:
        return self.rag_budget * CHARS_PER_TOKEN

    @property
    def memory_chars(self) -> int:
        return self.memory_budget * CHARS_PER_TOKEN

    @property
    def war_context_chars(self) -> int:
        return self.war_context_budget * CHARS_PER_TOKEN

    @property
    def system_prompt_chars(self) -> int:
        return self.system_prompt_budget * CHARS_PER_TOKEN

    @property
    def effective_context_chars(self) -> int:
        return self.effective_context_tokens * CHARS_PER_TOKEN

    @property
    def total_allocated(self) -> int:
        """Sum of all per-slice budgets (sanity check vs effective)."""
        return (
            self.system_prompt_budget
            + self.data_context_budget
            + self.tool_result_budget
            + self.rag_budget
            + self.memory_budget
            + self.history_budget
            + self.war_context_budget
            + self.capsule_budget
        )


def _effective_from_raw(raw_tokens: int) -> int:
    """Return the *effective* context budget — the portion of the raw context
    window where the model maintains high-quality attention.

    Research ("Lost in the Middle", Liu et al. 2024) shows open-weight models
    degrade significantly past 50-60% of their raw max_model_len.  We apply
    a 50% discount and cap at 65536 tokens (64K) to keep all pipeline agents
    in the high-quality attention zone.

    This triggers earlier compression and head-tail truncation, which is
    strictly better than letting the model silently degrade or timeout.
    """
    EFFECTIVE_RATIO = 0.50       # 50% of raw context
    MAX_BUDGET_CEILING = 65536   # 64K tokens hard cap
    return min(int(raw_tokens * EFFECTIVE_RATIO), MAX_BUDGET_CEILING)


def _compute_slice_budgets(effective: int) -> dict:
    """Distribute effective context across slice categories.

    Allocation percentages calibrated from empirical DB analysis (2026-05-05):
      - System prompt:  10%  (stable, rarely varies)
      - Data context:   40%  (T1 agents need ~11K tokens; this is the bulk)
      - Tool results:   20%  (YouTube P95 = 14K tokens raw; needs headroom)
      - RAG:             8%  (already capped at 3K chars by retriever.py)
      - Memory:          5%  (trading memory = 762B; lessons avg 137 chars)
      - History:        12%  (compressed by compressor.py before growth)
      - War context:     3%  (structured intel, compact)
      - Capsules:        2%  (already capped at 600 tokens by capsule.py)
    """
    return {
        "system_prompt_budget": int(effective * 0.10),
        "data_context_budget": int(effective * 0.40),
        "tool_result_budget": int(effective * 0.20),
        "rag_budget": int(effective * 0.08),
        "memory_budget": int(effective * 0.05),
        "history_budget": int(effective * 0.12),
        "war_context_budget": int(effective * 0.03),
        "capsule_budget": int(effective * 0.02),
    }


# ── Runtime cache of discovered model budgets ─────────────────────────
_budget_cache: dict[str, ContextBudget] = {}


def register_model_context(model_id: str, raw_context_tokens: int) -> ContextBudget:
    """Register a discovered model's raw context length and compute its budget.

    Called by ``vllm_client.discover_roles()`` after querying ``/v1/models``.
    """
    effective = _effective_from_raw(raw_context_tokens)
    slices = _compute_slice_budgets(effective)

    budget = ContextBudget(
        model_id=model_id,
        raw_context_tokens=raw_context_tokens,
        effective_context_tokens=effective,
        **slices,
    )

    _budget_cache[model_id] = budget
    logger.info(
        "[BUDGET] Registered model %s: raw=%d, effective=%d, compressor=%d, "
        "allocated=%d (%.0f%% of effective)",
        model_id,
        raw_context_tokens,
        effective,
        budget.compressor_threshold,
        budget.total_allocated,
        (budget.total_allocated / effective * 100) if effective else 0,
    )
    return budget


# ── Default budget (used when model discovery hasn't run yet) ─────────
# Default budget scaled to 50% of a 32K raw model (16K effective).
_DEFAULT_BUDGET = ContextBudget(
    model_id="default",
    raw_context_tokens=32768,
    effective_context_tokens=16384,
    system_prompt_budget=1638,   # 10%
    data_context_budget=6553,    # 40%
    tool_result_budget=3276,     # 20%
    rag_budget=1310,             # 8%
    memory_budget=819,           # 5%
    history_budget=1966,         # 12%
    war_context_budget=491,      # 3%
    capsule_budget=327,          # 2%
)


def get_context_budget(model_id: str | None = None) -> ContextBudget:
    """Return the context budget for a model.

    Falls back to a conservative default if the model hasn't been discovered.
    """
    if model_id and model_id in _budget_cache:
        return _budget_cache[model_id]

    # Try partial match (model IDs can be long paths)
    if model_id:
        for cached_id, budget in _budget_cache.items():
            if model_id in cached_id or cached_id in model_id:
                return budget

    # Return any cached budget if we have one (all models are same in this setup)
    if _budget_cache:
        return next(iter(_budget_cache.values()))

    return _DEFAULT_BUDGET


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimation (~4 chars per token)."""
    return len(str(text)) // CHARS_PER_TOKEN

"""
AgentBudget — Governor for agent execution costs and loops.
Ensures agents don't get stuck in infinite tool-calling loops.
"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentBudget:
    max_turns: int = 15
    current_turns: int = 0
    max_tokens: int = 64000
    current_tokens: int = 0
    max_usd: float = 1.00
    current_usd: float = 0.0

    def consume_turn(self) -> bool:
        """Consume a turn. Returns False if budget exhausted."""
        self.current_turns += 1
        if self.current_turns >= self.max_turns:
            logger.warning("[BUDGET] Agent exhausted max turns (%d)", self.max_turns)
            return False
        return True

    def consume_tokens(self, tokens: int, cost_per_1k: float = 0.001) -> bool:
        """Consume tokens and track estimated cost. Returns False if exhausted."""
        self.current_tokens += tokens
        self.current_usd += (tokens / 1000.0) * cost_per_1k

        if self.current_tokens >= self.max_tokens:
            logger.warning("[BUDGET] Agent exhausted max tokens (%d)", self.max_tokens)
            return False
        if self.current_usd >= self.max_usd:
            logger.warning("[BUDGET] Agent exhausted max budget ($%.2f)", self.max_usd)
            return False
        return True

    def is_exhausted(self) -> bool:
        return (
            self.current_turns >= self.max_turns
            or self.current_tokens >= self.max_tokens
            or self.current_usd >= self.max_usd
        )

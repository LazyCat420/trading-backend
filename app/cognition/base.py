"""
Base classes and generic utilities for the V2 Cognition framework.
"""

from typing import Any
import logging


class BaseCognitionModule:
    """Base class for any V2 Pipeline Module (Stage)."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"cognition.{self.name}")

    async def execute(self, payload: Any, context: dict) -> Any:
        self.logger.info(f"[{self.name}] Executing logic...")
        return await self._execute(payload, context)

    async def _execute(self, payload: Any, context: dict) -> Any:
        raise NotImplementedError("Subclasses must implement _execute")

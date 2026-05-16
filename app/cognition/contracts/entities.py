"""
Entity definitions for the Cognition V2 architecture.
"""

from pydantic import BaseModel
from typing import List, Optional


class ResolvedEntity(BaseModel):
    """
    A canonical entity resolved from unstructured or structured data.
    e.g., A specific company ticker like 'NVDA'.
    """

    entity_id: str
    entity_type: str  # e.g., 'ticker', 'person', 'sector'
    aliases: List[str] = []
    canonical_name: Optional[str] = None
    description: Optional[str] = None

    class Config:
        frozen = True

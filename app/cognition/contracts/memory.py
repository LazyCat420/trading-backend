"""
Memory definitions for the Cognition V2 architecture.
"""

from pydantic import BaseModel
from datetime import datetime


class EpisodicMemory(BaseModel):
    id: str
    timestamp: datetime
    content: str

    class Config:
        frozen = True

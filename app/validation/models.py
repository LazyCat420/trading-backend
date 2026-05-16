from enum import Enum
from typing import Optional
from pydantic import BaseModel

class ValidationStatus(str, Enum):
    PENDING = "pending"
    VALID = "valid"
    QUARANTINE = "quarantine"

class QuarantineReason(str, Enum):
    NO_DATA = "no_data"
    DELISTED = "delisted"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MANUAL = "manual"
    OTHER = "other"

class ValidationResult(BaseModel):
    ticker: str
    status: ValidationStatus
    reason: Optional[QuarantineReason] = None
    details: Optional[str] = None
    yfinance_pass: bool = False
    finviz_pass: bool = False
    content_pass: bool = False
    wikipedia_pass: bool = False

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal

class FundAlert(BaseModel):
    """Pydantic model for a fund alert, ensuring strict data validation before DB insertion."""
    id: str = Field(..., description="Unique identifier for the alert")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the alert")
    alert_type: Literal["stop_loss", "margin_call", "anomaly", "system_error", "massive_drop"] = Field(..., description="Type of the alert")
    ticker: Optional[str] = Field(None, description="Ticker associated with the alert, if any")
    entity_name: str = Field(..., description="Bot ID or system component that generated the alert")
    detail: str = Field(..., description="Detailed explanation of the alert")
    severity: Literal["high", "medium", "low"] = Field(..., description="Severity level of the alert")
    llm_summary: Optional[str] = Field(None, description="Optional LLM generated summary")
    is_read: bool = Field(False, description="Whether the alert has been read by the user")

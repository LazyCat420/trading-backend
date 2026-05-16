from pydantic import BaseModel
from typing import List, Optional


class FundListRow(BaseModel):
    cik: str
    name: Optional[str] = None
    latest_quarter: Optional[str] = None
    is_active: bool
    holding_count: int
    total_value_usd: float


class FundListResponse(BaseModel):
    funds: List[FundListRow]
    count: int


class HoldingRow(BaseModel):
    ticker: str
    name_of_issuer: Optional[str] = None
    shares: int
    value_usd: float
    pct_of_portfolio: float
    qoq_change: str
    share_change: int
    cusip: Optional[str] = None
    trend_direction: Optional[str] = None
    trend_streak: Optional[int] = None
    total_change_pct: Optional[float] = None


class FundHoldingsResponse(BaseModel):
    cik: str
    filer_name: str
    quarter: Optional[str]
    prior_quarter: Optional[str]
    total_value_usd: float
    holdings: List[HoldingRow]
    count: int
    total: int
    page: int
    limit: int
    message: Optional[str] = None


class OverlapRow(BaseModel):
    ticker: str
    name_of_issuer: Optional[str] = None
    fund_count: int
    total_value_usd: float
    total_shares: int
    fund_names: str


class FundOverlapResponse(BaseModel):
    overlap: List[OverlapRow]
    count: int
    min_funds: int


class HistoryRow(BaseModel):
    quarter: str
    filing_date: Optional[str] = None
    shares: int
    value_usd: float
    share_change: Optional[int] = None


class HoldingHistoryResponse(BaseModel):
    cik: str
    filer_name: str
    ticker: str
    history: List[HistoryRow]
    quarters_held: int

from dataclasses import dataclass


@dataclass
class PipelineContext:
    """Proof-of-concept context object to replace repetitive positional arguments."""

    tickers: list[str]
    collect: bool
    analyze: bool
    trade: bool
    cycle_id: str
    trigger_type: str = "manual"
    schedule_id: str | None = None
    resume_from: str | None = None
    already_analyzed: list[str] | None = None
    existing_results: list[dict] | None = None
    macro_memo: str = ""
    max_tickers: int | None = None  # user-specified cap on total tickers processed

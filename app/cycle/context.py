from dataclasses import dataclass

@dataclass
class CycleContext:
    tickers: list[str]
    collect: bool
    analyze: bool
    trade: bool
    cycle_id: str
    bot_id: str = "default"
    execution_mode: str = "production"
    trigger_type: str = "manual"
    schedule_id: str | None = None
    resume_from: str | None = None
    already_analyzed: list[str] | None = None
    existing_results: list[dict] | None = None
    macro_memo: str = ""
    max_tickers: int | None = None

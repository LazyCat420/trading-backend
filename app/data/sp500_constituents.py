"""
S&P 500 Constituents — loaded from JSON.
Source: Wikipedia, fetched 2026-05-08
Update periodically by re-running scripts/gen_sp500_list.py
"""

import json
from pathlib import Path

SP500_TICKERS: list[dict[str, str]] = json.loads(
    (Path(__file__).parent / "sp500_constituents.json").read_text()
)

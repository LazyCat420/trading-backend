"""
Precision RAG Tools — Implement dynamic, needle-in-a-haystack database lookups.
Allows agents to query specific numerical metrics or technical indicators on-demand,
minimizing token usage and preventing context-stuffing.
"""

import logging
from typing import Any, Dict, List, Optional
import time

logger = logging.getLogger(__name__)

# ── Synonym Mapping ──
SYNONYM_MAP: Dict[str, str] = {
    # Valuation & Financials
    "pe": "pe_ratio",
    "p/e": "pe_ratio",
    "p/e ratio": "pe_ratio",
    "pe ratio": "pe_ratio",
    "price to earnings": "pe_ratio",
    "revenue": "revenue",
    "sales": "revenue",
    "turnover": "revenue",
    "debt to equity": "debt_to_equity",
    "debt/equity": "debt_to_equity",
    "d/e": "debt_to_equity",
    "fcf": "free_cash_flow",
    "free cash flow": "free_cash_flow",
    
    # Technical Indicators
    "rsi": "rsi",
    "relative strength index": "rsi",
    "macd": "macd",
    "moving average convergence divergence": "macd",
    "sma": "sma_20",
    "sma20": "sma_20",
    "sma_20": "sma_20",
}

# ── Mock Database Content for Validation/Testing ──
MOCK_DATABASE: Dict[str, Dict[str, Any]] = {
    "AAPL": {
        "financials": {
            "pe_ratio": 28.5,
            "revenue": 90750.0,  # $90.75 Billion (in Millions)
            "debt_to_equity": 1.2,
            "free_cash_flow": 22400.0,
        },
        "technicals": {
            "rsi": 37.8,
            "macd": -1.45,
            "sma_20": 178.24,
        },
        "facts": [
            "CEO Tim Cook sold 10,000 shares on 2026-05-12.",
            "Congressman Sheldon Whitehouse bought $15,000 worth of AAPL call options on 2026-05-18.",
            "Apple reports strong sales growth in services division.",
        ]
    }
}

def resolve_synonym(metric_name: str) -> str:
    """Resolve metric synonyms or return the original lowercased string."""
    name_clean = metric_name.lower().strip()
    return SYNONYM_MAP.get(name_clean, name_clean)

def enrich_metric_metadata(metric: str, value: Any, period: str) -> Dict[str, Any]:
    """Wrap raw metric values in context-aware metadata to prevent out-of-scale errors."""
    meta = {
        "value": value,
        "period": period,
        "currency": "USD",
        "multiplier": "1"
    }
    
    metric_resolved = resolve_synonym(metric)
    if "pe_ratio" in metric_resolved or "debt_to_equity" in metric_resolved:
        meta["unit"] = "ratio"
        meta["currency"] = "N/A"
    elif "free_cash_flow" in metric_resolved or "revenue" in metric_resolved:
        meta["unit"] = "currency"
        meta["multiplier"] = "Millions"
    elif "rsi" in metric_resolved:
        meta["unit"] = "index"
        meta["currency"] = "N/A"
    else:
        meta["unit"] = "number"
        
    return meta

async def query_financial_metrics(
    ticker: str,
    metrics: List[str],
    period: Optional[str] = None
) -> Dict[str, Any]:
    """Fetch specific financial metrics or ratios for a ticker on-demand.
    
    Args:
        ticker: Symbol (e.g. 'AAPL')
        metrics: List of metrics (e.g. ['P/E Ratio', 'Sales'])
        period: Timeframe locked (defaults to 'Q1_2026')
    """
    target_period = period or "Q1_2026"
    results = {}
    
    ticker_data = MOCK_DATABASE.get(ticker.upper())
    if not ticker_data:
        return {"error": f"Ticker '{ticker}' not found in database."}
        
    financials = ticker_data.get("financials", {})
    
    for metric in metrics:
        resolved = resolve_synonym(metric)
        if resolved in financials:
            raw_val = financials[resolved]
            results[resolved] = enrich_metric_metadata(resolved, raw_val, target_period)
        else:
            results[resolved] = {"error": f"Metric '{metric}' not found."}
            
    return results

async def query_technical_indicator(
    ticker: str,
    indicator: str,
    timeframe: str = "daily"
) -> Dict[str, Any]:
    """Fetch specific technical indicator values (RSI, MACD, etc.)."""
    resolved = resolve_synonym(indicator)
    
    ticker_data = MOCK_DATABASE.get(ticker.upper())
    if not ticker_data:
        return {"error": f"Ticker '{ticker}' not found."}
        
    technicals = ticker_data.get("technicals", {})
    
    if resolved in technicals:
        raw_val = technicals[resolved]
        meta = enrich_metric_metadata(resolved, raw_val, "LIVE")
        meta["timeframe"] = timeframe
        if resolved == "rsi":
            meta["status"] = "OVERSOLD" if raw_val < 30 else "OVERBOUGHT" if raw_val > 70 else "NEUTRAL"
        return meta
    else:
        return {"error": f"Indicator '{indicator}' not found."}

async def search_database_facts(
    ticker: str,
    query: str
) -> Dict[str, Any]:
    """Semantic/keyword search in unstructured records for a ticker."""
    ticker_data = MOCK_DATABASE.get(ticker.upper())
    if not ticker_data:
        return {"error": f"Ticker '{ticker}' not found."}
        
    facts = ticker_data.get("facts", [])
    query_lower = query.lower()
    
    matched = [f for f in facts if query_lower in f.lower()]
    
    return {
        "ticker": ticker,
        "query": query,
        "results": matched if matched else ["No matching records found."]
    }

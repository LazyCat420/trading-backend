"""
Precision RAG Tools — Dynamic, needle-in-a-haystack database lookups.

Allows agents to query specific numerical metrics or technical indicators on-demand,
minimizing token usage and preventing context-stuffing.

These tools are designed for the Hybrid Context Model (see plans/prism_agentic_efficiency_plan.md):
- Track A (Narrative): Full text loaded upfront in prompts
- Track B (Numerical): These tools query exact values on-demand via /agent agentic loop
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.db.connection import get_db
from app.tools.registry import registry

logger = logging.getLogger(__name__)

# ── Synonym Mapping (Mitigation D: Schema Resolution) ──
SYNONYM_MAP: Dict[str, str] = {
    # Valuation & Financials → fundamentals table columns
    "pe": "pe_ratio",
    "p/e": "pe_ratio",
    "p/e ratio": "pe_ratio",
    "pe ratio": "pe_ratio",
    "price to earnings": "pe_ratio",
    "forward pe": "forward_pe",
    "forward p/e": "forward_pe",
    "peg": "peg_ratio",
    "peg ratio": "peg_ratio",
    "price to book": "price_to_book",
    "p/b": "price_to_book",
    "profit margin": "profit_margin",
    "roe": "roe",
    "return on equity": "roe",
    "revenue": "revenue",
    "sales": "revenue",
    "turnover": "revenue",
    "revenue growth": "revenue_growth",
    "debt to equity": "debt_to_equity",
    "debt/equity": "debt_to_equity",
    "d/e": "debt_to_equity",
    "beta": "beta",
    "market cap": "market_cap",
    "market_cap": "market_cap",
    "short float": "short_float_pct",
    "short interest": "short_float_pct",
    "52 week high": "week_52_high",
    "52w high": "week_52_high",
    "52 week low": "week_52_low",
    "52w low": "week_52_low",
    # Financial history columns
    "fcf": "free_cash_flow",
    "free cash flow": "free_cash_flow",
    "gross profit": "gross_profit",
    "operating income": "operating_income",
    "net income": "net_income",
    "eps": "eps",
    "earnings per share": "eps",
    # Technical Indicators → technicals table columns
    "rsi": "rsi_14",
    "rsi 14": "rsi_14",
    "relative strength index": "rsi_14",
    "macd": "macd",
    "moving average convergence divergence": "macd",
    "macd signal": "macd_signal",
    "macd histogram": "macd_hist",
    "macd hist": "macd_hist",
    "sma": "sma_20",
    "sma20": "sma_20",
    "sma 20": "sma_20",
    "sma_20": "sma_20",
    "sma50": "sma_50",
    "sma 50": "sma_50",
    "sma_50": "sma_50",
    "sma200": "sma_200",
    "sma 200": "sma_200",
    "sma_200": "sma_200",
    "ema12": "ema_12",
    "ema 12": "ema_12",
    "ema_12": "ema_12",
    "ema26": "ema_26",
    "ema 26": "ema_26",
    "ema_26": "ema_26",
    "atr": "atr_14",
    "atr 14": "atr_14",
    "average true range": "atr_14",
    "adx": "adx_14",
    "adx 14": "adx_14",
    "bollinger upper": "bb_upper",
    "bb upper": "bb_upper",
    "bollinger lower": "bb_lower",
    "bb lower": "bb_lower",
    "bollinger mid": "bb_mid",
    "bb mid": "bb_mid",
    "stochastic k": "stoch_k",
    "stoch k": "stoch_k",
    "stochastic d": "stoch_d",
    "stoch d": "stoch_d",
    "obv": "obv",
    "on balance volume": "obv",
    "vwap": "vwap",
    "support": "support",
    "resistance": "resistance",
}

# ── Column Metadata for enrichment (Mitigation A: Scale Protection) ──
METRIC_METADATA: Dict[str, Dict[str, str]] = {
    # Ratios (dimensionless)
    "pe_ratio": {"unit": "ratio", "multiplier": "1", "currency": "N/A"},
    "forward_pe": {"unit": "ratio", "multiplier": "1", "currency": "N/A"},
    "peg_ratio": {"unit": "ratio", "multiplier": "1", "currency": "N/A"},
    "price_to_book": {"unit": "ratio", "multiplier": "1", "currency": "N/A"},
    "profit_margin": {"unit": "percentage", "multiplier": "1", "currency": "N/A"},
    "roe": {"unit": "percentage", "multiplier": "1", "currency": "N/A"},
    "debt_to_equity": {"unit": "ratio", "multiplier": "1", "currency": "N/A"},
    "beta": {"unit": "coefficient", "multiplier": "1", "currency": "N/A"},
    "revenue_growth": {"unit": "percentage", "multiplier": "1", "currency": "N/A"},
    "short_float_pct": {"unit": "percentage", "multiplier": "1", "currency": "N/A"},
    # Currency values
    "revenue": {"unit": "currency", "multiplier": "raw", "currency": "USD"},
    "market_cap": {"unit": "currency", "multiplier": "raw", "currency": "USD"},
    "free_cash_flow": {"unit": "currency", "multiplier": "raw", "currency": "USD"},
    "gross_profit": {"unit": "currency", "multiplier": "raw", "currency": "USD"},
    "operating_income": {"unit": "currency", "multiplier": "raw", "currency": "USD"},
    "net_income": {"unit": "currency", "multiplier": "raw", "currency": "USD"},
    "eps": {"unit": "currency_per_share", "multiplier": "1", "currency": "USD"},
    "week_52_high": {"unit": "price", "multiplier": "1", "currency": "USD"},
    "week_52_low": {"unit": "price", "multiplier": "1", "currency": "USD"},
    # Technical indicators
    "rsi_14": {"unit": "index", "multiplier": "1", "currency": "N/A"},
    "macd": {"unit": "price_delta", "multiplier": "1", "currency": "USD"},
    "macd_signal": {"unit": "price_delta", "multiplier": "1", "currency": "USD"},
    "macd_hist": {"unit": "price_delta", "multiplier": "1", "currency": "USD"},
    "adx_14": {"unit": "index", "multiplier": "1", "currency": "N/A"},
    "stoch_k": {"unit": "percentage", "multiplier": "1", "currency": "N/A"},
    "stoch_d": {"unit": "percentage", "multiplier": "1", "currency": "N/A"},
    "atr_14": {"unit": "price", "multiplier": "1", "currency": "USD"},
    "obv": {"unit": "volume", "multiplier": "raw", "currency": "N/A"},
    "vwap": {"unit": "price", "multiplier": "1", "currency": "USD"},
    "support": {"unit": "price", "multiplier": "1", "currency": "USD"},
    "resistance": {"unit": "price", "multiplier": "1", "currency": "USD"},
}

# Tables where each metric lives
FUNDAMENTALS_COLS = {
    "pe_ratio", "forward_pe", "peg_ratio", "price_to_book", "profit_margin",
    "roe", "revenue", "revenue_growth", "debt_to_equity", "beta",
    "market_cap", "short_float_pct", "week_52_high", "week_52_low",
}
FINANCIAL_HISTORY_COLS = {
    "free_cash_flow", "gross_profit", "operating_income", "net_income", "eps",
}
TECHNICALS_COLS = {
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "bb_upper", "bb_mid", "bb_lower", "atr_14", "adx_14",
    "stoch_k", "stoch_d", "obv", "vwap", "support", "resistance",
}
# SMA/EMA/BB are price-level indicators
for _col in ("sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "bb_upper", "bb_mid", "bb_lower"):
    METRIC_METADATA[_col] = {"unit": "price", "multiplier": "1", "currency": "USD"}


def resolve_synonym(metric_name: str) -> str:
    """Resolve metric synonyms to canonical DB column names."""
    name_clean = metric_name.lower().strip()
    return SYNONYM_MAP.get(name_clean, name_clean)


def enrich_metric_metadata(metric: str, value: Any, period: str) -> Dict[str, Any]:
    """Wrap raw metric values in context-aware metadata to prevent out-of-scale errors."""
    resolved = resolve_synonym(metric)
    defaults = METRIC_METADATA.get(resolved, {"unit": "number", "multiplier": "1", "currency": "N/A"})

    meta = {
        "metric": resolved,
        "value": value,
        "period": period,
        "unit": defaults["unit"],
        "multiplier": defaults["multiplier"],
        "currency": defaults["currency"],
    }

    # Add RSI status labels
    if resolved == "rsi_14" and value is not None:
        meta["status"] = "OVERSOLD" if value < 30 else "OVERBOUGHT" if value > 70 else "NEUTRAL"

    # Add ADX trend strength
    if resolved == "adx_14" and value is not None:
        meta["status"] = "STRONG_TREND" if value > 25 else "WEAK_TREND"

    # Add stochastic labels
    if resolved in ("stoch_k", "stoch_d") and value is not None:
        meta["status"] = "OVERBOUGHT" if value > 80 else "OVERSOLD" if value < 20 else "NEUTRAL"

    return meta


# ═════════════════════════════════════════════════════════════
# Tool 1: query_financial_metrics
# ═════════════════════════════════════════════════════════════

@registry.register(
    name="query_financial_metrics",
    description=(
        "Fetch specific financial metrics, ratios, or multiples for a stock ticker from the database. "
        "Use this when you need exact numerical values like P/E ratio, revenue, debt-to-equity, free cash flow, etc. "
        "Accepts multiple metrics in a single call for efficiency. "
        "Supports synonyms (e.g. 'P/E Ratio', 'Sales', 'D/E' all resolve correctly)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g. 'AAPL')",
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of metric names to fetch. Examples: "
                    "['pe_ratio', 'revenue', 'debt_to_equity', 'free_cash_flow', 'eps']. "
                    "Synonyms accepted: 'P/E Ratio', 'Sales', 'D/E', 'FCF'."
                ),
            },
            "period": {
                "type": "string",
                "description": "Optional timeframe context (e.g. 'Q1_2026', 'FY_2025'). Defaults to latest available.",
            },
        },
        "required": ["ticker", "metrics"],
    },
    tier=0,
    source="precision_rag",
    tags=["financial", "fundamental", "metrics", "precision"],
)
async def query_financial_metrics(
    ticker: str,
    metrics: List[str],
    period: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch specific financial metrics or ratios for a ticker on-demand from PostgreSQL."""
    ticker = ticker.upper().strip()
    results = {}

    # Resolve all synonyms
    resolved_metrics = {m: resolve_synonym(m) for m in metrics}

    # Partition metrics by source table
    fund_metrics = {orig: res for orig, res in resolved_metrics.items() if res in FUNDAMENTALS_COLS}
    fin_hist_metrics = {orig: res for orig, res in resolved_metrics.items() if res in FINANCIAL_HISTORY_COLS}
    unknown_metrics = {orig: res for orig, res in resolved_metrics.items()
                       if res not in FUNDAMENTALS_COLS and res not in FINANCIAL_HISTORY_COLS and res not in TECHNICALS_COLS}

    try:
        with get_db() as db:
            # Query fundamentals table
            if fund_metrics:
                cols_needed = list(set(fund_metrics.values()))
                col_str = ", ".join(cols_needed)
                row = db.execute(
                    f"SELECT snapshot_date, {col_str} FROM fundamentals "
                    f"WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                    [ticker],
                ).fetchone()

                if row:
                    snapshot_date = str(row[0])
                    period_label = period or snapshot_date
                    for orig, resolved in fund_metrics.items():
                        idx = cols_needed.index(resolved) + 1  # +1 for snapshot_date
                        val = row[idx]
                        results[resolved] = enrich_metric_metadata(resolved, val, period_label)
                else:
                    for orig, resolved in fund_metrics.items():
                        results[resolved] = {"error": f"No fundamentals data found for {ticker}."}

            # Query financial_history table
            if fin_hist_metrics:
                cols_needed = list(set(fin_hist_metrics.values()))
                col_str = ", ".join(cols_needed)
                row = db.execute(
                    f"SELECT period_end, {col_str} FROM financial_history "
                    f"WHERE ticker = %s ORDER BY period_end DESC LIMIT 1",
                    [ticker],
                ).fetchone()

                if row:
                    period_end = str(row[0])
                    period_label = period or period_end
                    for orig, resolved in fin_hist_metrics.items():
                        idx = cols_needed.index(resolved) + 1
                        val = row[idx]
                        results[resolved] = enrich_metric_metadata(resolved, val, period_label)
                else:
                    for orig, resolved in fin_hist_metrics.items():
                        results[resolved] = {"error": f"No financial history found for {ticker}."}

            # Handle unknown metrics
            for orig, resolved in unknown_metrics.items():
                results[resolved] = {"error": f"Metric '{orig}' not recognized. Use query_technical_indicator for technical data."}

    except Exception as e:
        logger.error("[PrecisionRAG] query_financial_metrics failed for %s: %s", ticker, e)
        return {"error": f"Database query failed: {str(e)}"}

    results["_ticker"] = ticker
    results["_timestamp"] = datetime.now(timezone.utc).isoformat()
    return results


# ═════════════════════════════════════════════════════════════
# Tool 2: query_technical_indicator
# ═════════════════════════════════════════════════════════════

@registry.register(
    name="query_technical_indicator",
    description=(
        "Fetch specific technical indicators (RSI, MACD, SMA, Bollinger Bands, ATR, ADX, etc.) for a stock. "
        "Use this to verify momentum claims, check exact RSI/MACD values, or compare price to moving averages. "
        "Returns the latest computed value with status labels (OVERSOLD/OVERBOUGHT/NEUTRAL)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g. 'AAPL')",
            },
            "indicator": {
                "type": "string",
                "description": (
                    "The indicator to fetch. Examples: 'RSI', 'MACD', 'SMA_20', 'SMA_50', 'SMA_200', "
                    "'ATR', 'ADX', 'Bollinger Upper', 'Stochastic K', 'OBV', 'VWAP', 'Support', 'Resistance'."
                ),
            },
            "timeframe": {
                "type": "string",
                "description": "Timeframe context (e.g. 'daily', 'weekly'). Currently only daily is computed.",
            },
        },
        "required": ["ticker", "indicator"],
    },
    tier=0,
    source="precision_rag",
    tags=["technical", "indicator", "momentum", "precision"],
)
async def query_technical_indicator(
    ticker: str,
    indicator: str,
    timeframe: str = "daily",
) -> Dict[str, Any]:
    """Fetch a specific technical indicator value from the technicals table."""
    ticker = ticker.upper().strip()
    resolved = resolve_synonym(indicator)

    if resolved not in TECHNICALS_COLS:
        return {
            "error": f"Indicator '{indicator}' not recognized as a technical indicator. "
                     f"Use query_financial_metrics for fundamental data.",
            "available_indicators": sorted(TECHNICALS_COLS),
        }

    try:
        # Ensure technicals are computed
        from app.processors.technical_processor import compute_technicals
        try:
            compute_technicals(ticker)
        except Exception:
            pass  # Non-fatal, data may already exist

        with get_db() as db:
            row = db.execute(
                f"SELECT date, {resolved} FROM technicals "
                f"WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                [ticker],
            ).fetchone()

            if not row:
                return {"error": f"No technical data available for {ticker}. Run get_market_data first."}

            date_val = str(row[0])
            value = row[1]

            result = enrich_metric_metadata(resolved, value, date_val)
            result["timeframe"] = timeframe
            result["as_of"] = date_val
            result["_ticker"] = ticker

            # Add price context for moving averages
            if resolved in ("sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "bb_upper", "bb_lower", "bb_mid"):
                price_row = db.execute(
                    "SELECT close FROM price_history WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if price_row and price_row[0] and value:
                    result["current_price"] = float(price_row[0])
                    result["price_vs_indicator"] = "ABOVE" if float(price_row[0]) > float(value) else "BELOW"

            return result

    except Exception as e:
        logger.error("[PrecisionRAG] query_technical_indicator failed for %s/%s: %s", ticker, indicator, e)
        return {"error": f"Database query failed: {str(e)}"}


# ═════════════════════════════════════════════════════════════
# Tool 3: search_database_facts
# ═════════════════════════════════════════════════════════════

@registry.register(
    name="search_database_facts",
    description=(
        "Search the internal database for specific facts about a stock using semantic/keyword search. "
        "Use this to verify claims from news articles, find insider trading activity, congressional trades, "
        "or any other specific data point mentioned in narratives. "
        "Searches across news articles, Reddit posts, YouTube transcripts, and other scraped content."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g. 'AAPL')",
            },
            "query": {
                "type": "string",
                "description": (
                    "The topic or claim to search for. Be specific. "
                    "Examples: 'insider buying', 'CEO selling shares', 'congressional trades', "
                    "'earnings beat', 'supply chain issues'."
                ),
            },
        },
        "required": ["ticker", "query"],
    },
    tier=1,
    source="precision_rag",
    tags=["search", "facts", "verification", "precision"],
)
async def search_database_facts(
    ticker: str,
    query: str,
) -> Dict[str, Any]:
    """Semantic/keyword search for specific facts about a ticker across all internal data sources."""
    ticker = ticker.upper().strip()
    results_list = []

    try:
        # Strategy 1: Vector semantic search (if embedding service is available)
        try:
            from app.services.embedding_service import embedder
            from app.db.vector_store import vector_store

            query_vec = embedder.embed_text(
                query, prefix="Represent this sentence for searching relevant passages: "
            )
            vector_results = vector_store.search_cosine(
                query_embedding=query_vec, ticker=ticker, top_k=5
            )
            if vector_results:
                for r in vector_results:
                    results_list.append({
                        "source": r.get("source_table", "unknown"),
                        "text": r.get("content_preview", ""),
                        "relevance_score": round(r.get("score", 0), 3),
                    })
        except Exception as e:
            logger.debug("[PrecisionRAG] Vector search unavailable, falling back to keyword: %s", e)

        # Strategy 2: Keyword search in news articles
        with get_db() as db:
            query_pattern = f"%{query}%"
            news_rows = db.execute(
                """
                SELECT title, publisher, published_at, COALESCE(llm_summary, summary)
                FROM news_articles
                WHERE ticker = %s AND (
                    title ILIKE %s OR summary ILIKE %s OR llm_summary ILIKE %s
                )
                ORDER BY published_at DESC LIMIT 5
                """,
                [ticker, query_pattern, query_pattern, query_pattern],
            ).fetchall()

            for row in news_rows:
                results_list.append({
                    "source": "news_articles",
                    "text": f"{row[0]} ({row[1]}, {row[2]}): {row[3] or 'No summary'}",
                    "date": str(row[2]) if row[2] else None,
                })

        if not results_list:
            return {
                "ticker": ticker,
                "query": query,
                "results": [],
                "message": "No matching records found in internal database.",
            }

        return {
            "ticker": ticker,
            "query": query,
            "results": results_list,
            "_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error("[PrecisionRAG] search_database_facts failed for %s: %s", ticker, e)
        return {"error": f"Search failed: {str(e)}"}

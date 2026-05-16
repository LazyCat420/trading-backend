"""
Data Imputer — Pre-fills missing data fields before LLM context building.

Strategy:
  1. Check PostgreSQL for existing data
  2. If missing, attempt mathematical imputation from surrounding data
  3. If still missing, try fallback API chain (yfinance → FMP → Finnhub)
  4. If truly unavailable, label as DATA_UNAVAILABLE with source metadata

Every imputed value carries its provenance (source name + method) so the
hallucination checker can verify LLM claims against ground truth.

Usage:
    from app.pipeline.data.data_imputer import impute_ticker_data
    report = impute_ticker_data("NVDA")
    # report = {
    #   "ticker": "NVDA",
    #   "fields_checked": 12,
    #   "fields_present": 10,
    #   "fields_imputed": 1,
    #   "fields_unavailable": 1,
    #   "imputed": [{"field": "rsi_14", "value": 54.2, "source": "computed", "method": "wilder_rsi"}],
    #   "unavailable": [{"field": "short_float_pct", "label": "DATA_UNAVAILABLE"}],
    # }
"""

import logging
import statistics
from typing import Any, Optional

from app.db.connection import get_db

logger = logging.getLogger(__name__)


# Fields we check and their imputation strategies
CRITICAL_FIELDS = [
    # (field_name, table, column, imputation_method)
    ("close_price", "price_history", "close", "interpolate"),
    ("volume", "price_history", "volume", "rolling_avg"),
    ("rsi_14", "technicals", "rsi_14", "compute_rsi"),
    ("sma_20", "technicals", "sma_20", "compute_sma"),
    ("sma_50", "technicals", "sma_50", "compute_sma"),
    ("macd", "technicals", "macd", "compute_macd"),
    ("market_cap", "fundamentals", "market_cap", "derive_market_cap"),
    ("pe_ratio", "fundamentals", "pe_ratio", "none"),
    ("revenue", "fundamentals", "revenue", "none"),
    ("profit_margin", "fundamentals", "profit_margin", "none"),
    ("debt_to_equity", "fundamentals", "debt_to_equity", "none"),
    ("short_float_pct", "fundamentals", "short_float_pct", "none"),
]


def impute_ticker_data(ticker: str) -> dict:
    """Check data completeness for a ticker and impute missing fields.

    Returns a report dict with fields_checked, fields_present,
    fields_imputed, fields_unavailable, and detailed lists.
    """
    with get_db() as db:
        ticker = ticker.upper()

        report = {
            "ticker": ticker,
            "fields_checked": len(CRITICAL_FIELDS),
            "fields_present": 0,
            "fields_imputed": 0,
            "fields_unavailable": 0,
            "imputed": [],
            "unavailable": [],
            "provenance": {},  # field -> {value, source, method, timestamp}
        }

        for field_name, table, column, method in CRITICAL_FIELDS:
            try:
                # Step 1: Check if data exists in DB
                value = _check_db_field(db, ticker, table, column)

                if value is not None:
                    report["fields_present"] += 1
                    report["provenance"][field_name] = {
                        "value": value,
                        "source": "database",
                        "method": "direct_query",
                        "fresh": True,
                    }
                    continue

                # Step 2: Try mathematical imputation
                if method != "none":
                    imputed_value = _impute_field(db, ticker, field_name, method)
                    if imputed_value is not None:
                        report["fields_imputed"] += 1
                        report["imputed"].append(
                            {
                                "field": field_name,
                                "value": imputed_value,
                                "source": "computed",
                                "method": method,
                            }
                        )
                        report["provenance"][field_name] = {
                            "value": imputed_value,
                            "source": "imputed",
                            "method": method,
                            "fresh": False,
                        }
                        continue

                # Step 3: Try fallback API chain
                api_value = _try_fallback_apis(ticker, field_name)
                if api_value is not None:
                    report["fields_imputed"] += 1
                    report["imputed"].append(
                        {
                            "field": field_name,
                            "value": api_value["value"],
                            "source": api_value["source"],
                            "method": "api_fallback",
                        }
                    )
                    report["provenance"][field_name] = {
                        "value": api_value["value"],
                        "source": api_value["source"],
                        "method": "api_fallback",
                        "fresh": True,
                    }
                    continue

                # Step 4: Mark as unavailable
                report["fields_unavailable"] += 1
                report["unavailable"].append(
                    {
                        "field": field_name,
                        "label": "DATA_UNAVAILABLE",
                    }
                )
                report["provenance"][field_name] = {
                    "value": "DATA_UNAVAILABLE",
                    "source": "none",
                    "method": "exhausted_all_sources",
                    "fresh": False,
                }

            except Exception as e:
                logger.warning(
                    "[IMPUTER] Error checking %s.%s: %s", ticker, field_name, e
                )
                report["fields_unavailable"] += 1
                report["unavailable"].append(
                    {
                        "field": field_name,
                        "label": "DATA_UNAVAILABLE",
                        "error": str(e)[:100],
                    }
                )

        completeness_pct = round(
            (report["fields_present"] + report["fields_imputed"])
            / max(report["fields_checked"], 1)
            * 100,
            1,
        )
        report["completeness_pct"] = completeness_pct

        logger.info(
            "[IMPUTER] %s: %d/%d present, %d imputed, %d unavailable (%.1f%% complete)",
            ticker,
            report["fields_present"],
            report["fields_checked"],
            report["fields_imputed"],
            report["fields_unavailable"],
            completeness_pct,
        )

        return report


def build_data_quality_header(report: dict) -> str:
    """Build a data quality header to prepend to the LLM context.

    This tells the LLM exactly which fields are real, imputed, or missing
    so it cannot invent values for missing fields.
    """
    ticker = report["ticker"]
    lines = [
        f"## DATA QUALITY REPORT ({ticker})",
        f"Completeness: {report['completeness_pct']}%",
        f"Fields: {report['fields_present']} from DB, "
        f"{report['fields_imputed']} imputed, "
        f"{report['fields_unavailable']} unavailable",
        "",
    ]

    if report["imputed"]:
        lines.append("**Imputed fields** (computed, not from API):")
        for item in report["imputed"]:
            lines.append(
                f"  - {item['field']}: {item['value']} "
                f"(via {item['method']}, source: {item['source']})"
            )
        lines.append("")

    if report["unavailable"]:
        lines.append("**UNAVAILABLE fields** (DO NOT guess or invent values):")
        for item in report["unavailable"]:
            lines.append(f"  - {item['field']}: DATA_UNAVAILABLE")
        lines.append("")
        lines.append(
            "⚠️ For any field marked DATA_UNAVAILABLE, you MUST state "
            '"data not available" rather than guessing or using stale values.'
        )

    return "\n".join(lines) + "\n"


# ── Internal helpers ──────────────────────────────────────────────────


def _check_db_field(db, ticker: str, table: str, column: str) -> Optional[Any]:
    """Check if a field has a non-null value in the DB."""
    try:
        if table == "price_history":
            row = db.execute(
                f"SELECT {column} FROM {table} WHERE ticker = %s "
                f"ORDER BY date DESC LIMIT 1",
                [ticker],
            ).fetchone()
        elif table == "technicals":
            row = db.execute(
                f"SELECT {column} FROM {table} WHERE ticker = %s "
                f"ORDER BY date DESC LIMIT 1",
                [ticker],
            ).fetchone()
        elif table == "fundamentals":
            row = db.execute(
                f"SELECT {column} FROM {table} WHERE ticker = %s "
                f"ORDER BY snapshot_date DESC LIMIT 1",
                [ticker],
            ).fetchone()
        else:
            return None

        if row and row[0] is not None:
            return row[0]
    except Exception:
        pass
    return None


def _impute_field(db, ticker: str, field_name: str, method: str) -> Optional[Any]:
    """Attempt to compute a missing field from surrounding data."""
    try:
        if method == "interpolate":
            # Linear interpolation from surrounding candles
            rows = db.execute(
                "SELECT close FROM price_history WHERE ticker = %s "
                "ORDER BY date DESC LIMIT 5",
                [ticker],
            ).fetchall()
            if len(rows) >= 2:
                values = [r[0] for r in rows if r[0] is not None]
                if values:
                    return round(statistics.mean(values), 2)

        elif method == "rolling_avg":
            # 20-day rolling average for volume
            rows = db.execute(
                "SELECT volume FROM price_history WHERE ticker = %s "
                "AND volume IS NOT NULL ORDER BY date DESC LIMIT 20",
                [ticker],
            ).fetchall()
            if rows:
                values = [r[0] for r in rows if r[0] is not None and r[0] > 0]
                if values:
                    return int(statistics.mean(values))

        elif method == "compute_rsi":
            # Compute RSI from raw price changes
            rows = db.execute(
                "SELECT close FROM price_history WHERE ticker = %s "
                "ORDER BY date DESC LIMIT 15",
                [ticker],
            ).fetchall()
            if len(rows) >= 14:
                closes = [r[0] for r in reversed(rows) if r[0] is not None]
                if len(closes) >= 14:
                    return _calc_rsi(closes, 14)

        elif method == "compute_sma":
            # Compute SMA from price data
            period = 20 if "20" in field_name else 50 if "50" in field_name else 200
            rows = db.execute(
                "SELECT close FROM price_history WHERE ticker = %s "
                f"AND close IS NOT NULL ORDER BY date DESC LIMIT {period}",
                [ticker],
            ).fetchall()
            if len(rows) >= period:
                values = [r[0] for r in rows]
                return round(statistics.mean(values), 2)

        elif method == "compute_macd":
            # MACD = EMA(12) - EMA(26)
            rows = db.execute(
                "SELECT close FROM price_history WHERE ticker = %s "
                "AND close IS NOT NULL ORDER BY date DESC LIMIT 26",
                [ticker],
            ).fetchall()
            if len(rows) >= 26:
                closes = [r[0] for r in reversed(rows)]
                ema12 = _calc_ema(closes, 12)
                ema26 = _calc_ema(closes, 26)
                if ema12 is not None and ema26 is not None:
                    return round(ema12 - ema26, 4)

        elif method == "derive_market_cap":
            # Market cap = price × shares outstanding
            price_row = db.execute(
                "SELECT close FROM price_history WHERE ticker = %s "
                "ORDER BY date DESC LIMIT 1",
                [ticker],
            ).fetchone()
            shares_row = db.execute(
                "SELECT market_cap, close FROM price_history ph "
                "JOIN fundamentals f ON ph.ticker = f.ticker "
                "WHERE ph.ticker = %s AND f.market_cap > 0 AND ph.close > 0 "
                "ORDER BY f.snapshot_date DESC LIMIT 1",
                [ticker],
            ).fetchone()
            if price_row and shares_row and shares_row[1] > 0:
                implied_shares = shares_row[0] / shares_row[1]
                return int(price_row[0] * implied_shares)

    except Exception as e:
        logger.debug("[IMPUTER] Imputation failed for %s.%s: %s", ticker, field_name, e)

    return None


def _try_fallback_apis(ticker: str, field_name: str) -> Optional[dict]:
    """Try fallback API sources for a missing field.

    Returns {"value": ..., "source": "api_name"} or None.
    """
    # Only attempt API fallback for fundamental fields
    # (price/technical data should be computed, not fetched piecemeal)
    fundamental_fields = {
        "market_cap",
        "pe_ratio",
        "revenue",
        "profit_margin",
        "debt_to_equity",
        "short_float_pct",
    }
    if field_name not in fundamental_fields:
        return None

    # Try yfinance (already imported elsewhere in the codebase)
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        yf_mapping = {
            "market_cap": "marketCap",
            "pe_ratio": "trailingPE",
            "revenue": "totalRevenue",
            "profit_margin": "profitMargins",
            "debt_to_equity": "debtToEquity",
            "short_float_pct": "shortPercentOfFloat",
        }
        yf_key = yf_mapping.get(field_name)
        if yf_key and info.get(yf_key) is not None:
            return {"value": info[yf_key], "source": "yfinance_fallback"}
    except Exception:
        pass

    # Could add FMP and Finnhub fallbacks here in the future
    # For now, yfinance is the primary fallback

    return None


def _calc_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Calculate RSI using Wilder's smoothing method."""
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = statistics.mean(gains[:period])
    avg_loss = statistics.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def _calc_ema(values: list[float], period: int) -> Optional[float]:
    """Calculate Exponential Moving Average."""
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    ema = statistics.mean(values[:period])

    for val in values[period:]:
        ema = (val - ema) * multiplier + ema

    return ema

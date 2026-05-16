"""
Hallucination Checker — Post-LLM verification layer.

Validates LLM output claims against the source data that was sent to it.
Catches three types of hallucinations:

  1. INVENTED_FIELD — LLM references a data field never sent in the context
  2. WRONG_VALUE   — LLM cites a specific value that doesn't match source data
  3. STALE_REFERENCE — LLM uses data marked DATA_UNAVAILABLE as if it exists

If hallucination_rate > threshold → signal is auto-rejected with a full
audit trail. Rejected signals are logged for training data.

Usage:
    from app.pipeline.analysis.hallucination_checker import check_hallucinations
    result = check_hallucinations(
        llm_output={"action": "BUY", "rationale": "RSI is 32..."},
        context_provenance=imputer_report["provenance"],
        raw_context=context_blob,
        ticker="NVDA",
    )
    if result["rejected"]:
        # Signal was hallucinated — don't trade
"""

import logging
import re
import json
import uuid
import dataclasses
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Default threshold: reject if >20% of verifiable claims are hallucinated
DEFAULT_HALLUCINATION_THRESHOLD = 0.20


def check_hallucinations(
    llm_output: dict,
    context_provenance: dict,
    raw_context: str,
    ticker: str,
    threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
) -> dict:
    """Verify LLM output against source data.

    Args:
        llm_output: The LLM's response dict (action, confidence, rationale)
        context_provenance: Field provenance from data_imputer
        raw_context: The exact context string sent to the LLM
        ticker: Ticker being analyzed
        threshold: Max acceptable hallucination rate (0.0-1.0)

    Returns:
        {
            "ticker": str,
            "total_claims": int,
            "verified_claims": int,
            "hallucinations": [{"type": str, "field": str, "detail": str}],
            "hallucination_rate": float,
            "rejected": bool,
            "rejection_reason": str | None,
        }
    """
    result = {
        "ticker": ticker,
        "total_claims": 0,
        "verified_claims": 0,
        "hallucinations": [],
        "hallucination_rate": 0.0,
        "rejected": False,
        "rejection_reason": None,
    }

    rationale = llm_output.get("rationale", "")
    if not rationale:
        return result

    # ── Check 1: STALE_REFERENCE — using DATA_UNAVAILABLE fields ──
    unavailable_fields = [
        field
        for field, prov in context_provenance.items()
        if prov.get("value") == "DATA_UNAVAILABLE"
    ]

    for field in unavailable_fields:
        # Check if the LLM mentions this field with a specific value
        field_patterns = _get_field_patterns(field)
        for pattern in field_patterns:
            matches = re.findall(pattern, rationale, re.IGNORECASE)
            for match in matches:
                # LLM mentioned the field — check if it assigned a value
                if _has_numeric_claim(rationale, field):
                    result["hallucinations"].append(
                        {
                            "type": "STALE_REFERENCE",
                            "field": field,
                            "detail": f"LLM cited a value for '{field}' which was "
                            f"marked DATA_UNAVAILABLE in the context",
                        }
                    )
                    result["total_claims"] += 1

    # ── Check 2: WRONG_VALUE — numeric claims that don't match source ──
    for field, prov in context_provenance.items():
        if prov.get("value") == "DATA_UNAVAILABLE":
            continue

        source_value = prov.get("value")
        if source_value is None or not isinstance(source_value, (int, float)):
            continue

        # Look for the LLM mentioning this field with a number
        claimed_value = _extract_claimed_value(rationale, field)
        if claimed_value is not None:
            result["total_claims"] += 1

            # Allow 5% tolerance for rounding
            tolerance = abs(source_value * 0.05) if source_value != 0 else 1.0
            if abs(claimed_value - source_value) <= tolerance:
                result["verified_claims"] += 1
            else:
                result["hallucinations"].append(
                    {
                        "type": "WRONG_VALUE",
                        "field": field,
                        "detail": f"LLM claimed {field}={claimed_value}, "
                        f"but source data shows {field}={source_value} "
                        f"(source: {prov.get('source', '?')})",
                        "claimed": claimed_value,
                        "actual": source_value,
                    }
                )

    # ── Check 3: INVENTED_FIELD — references to data not in context ──
    # Check for specific financial metrics mentioned but not in our data
    invented_patterns = {
        "earnings_surprise": r"earnings?\s+surpris",
        "insider_buying": r"insider\s+(?:buying|purchasing|accumulating)",
        "analyst_upgrade": r"analyst\s+(?:upgrade|downgrade|rating)",
        "dividend_yield": r"dividend\s+yield\s+(?:of|at|is)\s+(\d+\.?\d*)",
    }

    for field, pattern in invented_patterns.items():
        if field not in context_provenance:
            matches = re.findall(pattern, rationale, re.IGNORECASE)
            if matches:
                # Check if this info was actually in the raw context
                if not _field_in_context(field, raw_context):
                    result["hallucinations"].append(
                        {
                            "type": "INVENTED_FIELD",
                            "field": field,
                            "detail": f"LLM referenced '{field}' which was not "
                            f"provided in the context data",
                        }
                    )
                    result["total_claims"] += 1

    # ── Calculate hallucination rate ──
    total = result["total_claims"]
    if total > 0:
        hallucinated = len(result["hallucinations"])
        result["hallucination_rate"] = round(hallucinated / total, 3)
        result["verified_claims"] = total - hallucinated

    # ── Reject if above threshold ──
    if total > 0 and result["hallucination_rate"] > threshold:
        result["rejected"] = True
        result["rejection_reason"] = (
            f"Hallucination rate {result['hallucination_rate']:.0%} "
            f"exceeds threshold {threshold:.0%}. "
            f"{len(result['hallucinations'])} of {total} claims failed verification."
        )
        logger.warning(
            "[HALLUCINATION] %s REJECTED: %s",
            ticker,
            result["rejection_reason"],
        )
        _log_rejection(ticker, llm_output, result)
    elif result["hallucinations"]:
        logger.info(
            "[HALLUCINATION] %s: %d hallucinations found but below threshold "
            "(%.0f%% < %.0f%%)",
            ticker,
            len(result["hallucinations"]),
            result["hallucination_rate"] * 100,
            threshold * 100,
        )

    return result


def get_hallucination_stats() -> dict:
    """Get aggregate hallucination stats from the database."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            # Ensure table exists
            db.execute("""
                CREATE TABLE IF NOT EXISTS hallucination_log (
                    id VARCHAR PRIMARY KEY,
                    ticker VARCHAR,
                    cycle_id VARCHAR,
                    hallucination_count INTEGER,
                    total_claims INTEGER,
                    hallucination_rate FLOAT,
                    rejected BOOLEAN,
                    details_json VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Recent stats
            rows = db.execute("""
                SELECT
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN rejected THEN 1 ELSE 0 END) as total_rejected,
                    AVG(hallucination_rate) as avg_rate,
                    MAX(hallucination_rate) as max_rate,
                    SUM(hallucination_count) as total_hallucinations
                FROM hallucination_log
                WHERE created_at > NOW() - INTERVAL '24 HOURS'
            """).fetchone()

            recent = db.execute("""
                SELECT ticker, hallucination_count, hallucination_rate, rejected, created_at
                FROM hallucination_log
                ORDER BY created_at DESC
                LIMIT 10
            """).fetchall()

            return {
                "last_24h": {
                    "total_checks": rows[0] if rows else 0,
                    "total_rejected": rows[1] if rows else 0,
                    "avg_hallucination_rate": round(rows[2] * 100, 1)
                    if rows and rows[2]
                    else 0,
                    "max_hallucination_rate": round(rows[3] * 100, 1)
                    if rows and rows[3]
                    else 0,
                    "total_hallucinations": rows[4] if rows else 0,
                },
                "recent": [
                    {
                        "ticker": r[0],
                        "hallucinations": r[1],
                        "rate_pct": round(r[2] * 100, 1) if r[2] else 0,
                        "rejected": r[3],
                        "created_at": str(r[4]),
                    }
                    for r in (recent or [])
                ],
            }
    except Exception as e:
        logger.warning("[HALLUCINATION] Failed to get stats: %s", e)
        return {"error": str(e)}


# ── Internal helpers ──────────────────────────────────────────────────


def audit_numeric_divergence(
    llm_rationale: str, ticker: str, cycle_id: str, source_file: str
):
    """Checks for numeric divergence in text and logs to hallucination_audit table."""
    try:
        from app.data.market_data_store import get_latest_snapshot

        snapshot = get_latest_snapshot(ticker)
        if not snapshot:
            return

        snapshot_dict = dataclasses.asdict(snapshot)
        context_provenance = {
            k: {"value": v, "source": "MarketSnapshot"}
            for k, v in snapshot_dict.items()
            if v is not None
        }

        result = check_hallucinations(
            llm_output={"action": "HOLD", "confidence": 0, "rationale": llm_rationale},
            context_provenance=context_provenance,
            raw_context="",
            ticker=ticker,
            threshold=1.0,  # don't reject here
        )

        if result.get("hallucinations"):
            from app.db.connection import get_db
            import uuid

            with get_db() as db:
                for h in result["hallucinations"]:
                    # Only log WRONG_VALUE which is numeric divergence
                    if h["type"] == "WRONG_VALUE":
                        db.execute(
                            """
                            INSERT INTO hallucination_audit (id, cycle_id, ticker, source_file, foreign_value, context_snippet)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                str(uuid.uuid4()),
                                cycle_id,
                                ticker,
                                source_file,
                                str(h.get("claimed")),
                                h.get("detail", ""),
                            ),
                        )
    except Exception as e:
        logger.warning(f"Failed to audit numeric divergence: {e}")


def _get_field_patterns(field: str) -> list[str]:
    """Get regex patterns that match mentions of a data field."""
    clean = field.replace("_", r"[\s_-]?")
    return [
        rf"\b{clean}\b",
        rf"\b{field}\b",
    ]


def _has_numeric_claim(text: str, field: str) -> bool:
    """Check if the text assigns a specific numeric value to a field."""
    clean_field = field.replace("_", r"[\s_-]?")
    # Patterns like "RSI is 45" or "RSI of 45" or "RSI: 45"
    pattern = rf"{clean_field}\s*(?:is|of|at|:|=|stands?\s+at)\s*[\$]?\d"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _extract_claimed_value(text: str, field: str) -> float | None:
    """Extract a numeric value claimed for a specific field."""
    clean_field = field.replace("_", r"[\s_-]?")

    # Match patterns like "RSI is 45.2" or "RSI of 45" or "RSI at $45.2B"
    patterns = [
        rf"{clean_field}\s*(?:is|of|at|:|=|stands?\s+at)\s*[\$]?([\d,]+\.?\d*)",
        rf"{clean_field}\s*\(?[\$]?([\d,]+\.?\d*)\)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val_str = match.group(1).replace(",", "")
                return float(val_str)
            except (ValueError, IndexError):
                continue

    return None


def _field_in_context(field: str, raw_context: str) -> bool:
    """Check if a field concept appears anywhere in the raw context."""
    field_synonyms = {
        "earnings_surprise": [
            "earnings surprise",
            "beat estimates",
            "missed estimates",
        ],
        "insider_buying": ["insider", "director purchase", "officer buy"],
        "analyst_upgrade": ["analyst", "upgrade", "downgrade", "price target"],
        "dividend_yield": ["dividend yield", "div yield", "annual dividend"],
    }

    synonyms = field_synonyms.get(field, [field.replace("_", " ")])
    context_lower = raw_context.lower()

    return any(syn.lower() in context_lower for syn in synonyms)


def _log_rejection(ticker: str, llm_output: dict, check_result: dict):
    """Log a hallucination rejection to PostgreSQL for audit."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS hallucination_log (
                    id VARCHAR PRIMARY KEY,
                    ticker VARCHAR,
                    cycle_id VARCHAR,
                    hallucination_count INTEGER,
                    total_claims INTEGER,
                    hallucination_rate FLOAT,
                    rejected BOOLEAN,
                    details_json VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            db.execute(
                """
                INSERT INTO hallucination_log
                (id, ticker, hallucination_count, total_claims, hallucination_rate,
                 rejected, details_json, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    str(uuid.uuid4()),
                    ticker,
                    len(check_result["hallucinations"]),
                    check_result["total_claims"],
                    check_result["hallucination_rate"],
                    check_result["rejected"],
                    json.dumps(
                        {
                            "hallucinations": check_result["hallucinations"],
                            "llm_action": llm_output.get("action"),
                            "llm_confidence": llm_output.get("confidence"),
                        }
                    ),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            logger.info("[HALLUCINATION] Rejection logged for %s", ticker)
    except Exception as e:
        logger.warning("[HALLUCINATION] Failed to log rejection: %s", e)

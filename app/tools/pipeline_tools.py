"""
Pipeline Tools — Exposes key pipeline phases as callable agent tools.

Inspired by Claude Code's 'skills' system: reusable workflows that
agents can invoke as high-level actions instead of being locked into
the rigid pipeline sequence.

This lets the CIO agent say "audit data quality for NVDA" as a tool
call during a debate, instead of waiting for the pipeline to run.
"""

import json
import logging

from app.tools.registry import registry

logger = logging.getLogger(__name__)


# ── Tool 1: Data Quality Audit ─────────────────────────────────────────
@registry.register(
    name="audit_data_quality",
    description=(
        "Run a data quality audit for a specific ticker. "
        "Checks price history, technicals, fundamentals, and news completeness. "
        "Returns a quality score (0-1) and lists any data gaps or missing sources. "
        "Also identifies junk stocks (penny stocks, micro-caps, zero volume) for pruning."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker to audit (e.g. 'NVDA').",
            },
        },
        "required": ["ticker"],
    },
    tier=1,
    source="internal_db",
    tags=["audit", "data", "quality", "gaps", "completeness"],
)
async def audit_data_quality(ticker: str) -> str:
    """Run a data quality audit for a single ticker and return structured results."""
    try:
        from app.pipeline.analysis.autoresearch import _audit_data_quality

        result = _audit_data_quality([ticker])
        return json.dumps(
            {
                "status": "success",
                "ticker": ticker,
                "avg_score": result.get("avg_score", 0),
                "gaps": result.get("gaps", []),
                "purged_tickers": result.get("purged_tickers", []),
                "per_ticker": result.get("per_ticker", {}),
            }
        )
    except Exception as e:
        logger.exception("[PipelineTools] audit_data_quality failed for %s", ticker)
        return json.dumps({"status": "error", "ticker": ticker, "message": str(e)})


# ── Tool 2: Decision Quality Audit ─────────────────────────────────────
@registry.register(
    name="audit_decision_quality",
    description=(
        "Audit the decision quality for a specific trading cycle. "
        "Checks BUY/SELL/HOLD ratios, confidence distribution, and identifies "
        "issues like high HOLD ratio or uniform confidence (sign of LLM laziness)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cycle_id": {
                "type": "string",
                "description": "The cycle ID to audit (e.g. 'abc123def456').",
            },
        },
        "required": ["cycle_id"],
    },
    tier=1,
    source="internal_db",
    tags=["audit", "decision", "confidence", "quality"],
)
async def audit_decision_quality(cycle_id: str) -> str:
    """Audit decision quality for a specific cycle."""
    try:
        from app.pipeline.analysis.autoresearch import _audit_decisions

        # Build a minimal cycle_summary from the DB
        from app.db.connection import get_db

        with get_db() as db:
            rows = db.execute(
                "SELECT (result_json::jsonb)->>'action' AS action, confidence FROM analysis_results WHERE cycle_id = %s",
                [cycle_id],
            ).fetchall()

        buy_count = sum(1 for r in rows if r[0] and r[0].upper() == "BUY")
        sell_count = sum(1 for r in rows if r[0] and r[0].upper() == "SELL")
        hold_count = sum(1 for r in rows if r[0] and r[0].upper() == "HOLD")

        cycle_summary = {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
        }

        result = _audit_decisions(cycle_id, cycle_summary)
        return json.dumps({"status": "success", "cycle_id": cycle_id, **result})
    except Exception as e:
        logger.exception("[PipelineTools] audit_decision_quality failed")
        return json.dumps({"status": "error", "cycle_id": cycle_id, "message": str(e)})


# ── Tool 3: Hallucination Check ────────────────────────────────────────
@registry.register(
    name="check_hallucination",
    description=(
        "Run a hallucination check on an LLM claim by cross-referencing it "
        "against the actual data in our database. Useful during debates when "
        "one agent suspects another agent is citing fabricated numbers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker the claim is about.",
            },
            "claim": {
                "type": "string",
                "description": "The specific claim to verify (e.g. 'RSI is 28' or 'P/E ratio is 15.3').",
            },
        },
        "required": ["ticker", "claim"],
    },
    tier=2,
    source="internal_db",
    tags=["hallucination", "verify", "fact-check", "ground-truth"],
)
async def check_hallucination(ticker: str, claim: str) -> str:
    """Verify an LLM claim against ground-truth data in the database."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            # Gather ground-truth data for comparison
            ground_truth = {}

            # Price data
            try:
                price_row = db.execute(
                    "SELECT close, volume FROM price_history WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if price_row:
                    ground_truth["latest_close"] = price_row[0]
                    ground_truth["latest_volume"] = price_row[1]
            except Exception:
                pass

            # Technical indicators
            try:
                tech_row = db.execute(
                    "SELECT rsi_14, macd, macd_signal, macd_hist, sma_20, sma_50, sma_200, "
                    "atr_14, adx_14, stoch_k, stoch_d, bb_upper, bb_lower "
                    "FROM technicals WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if tech_row:
                    labels = ["rsi_14", "macd", "macd_signal", "macd_hist",
                              "sma_20", "sma_50", "sma_200", "atr_14", "adx_14",
                              "stoch_k", "stoch_d", "bb_upper", "bb_lower"]
                    ground_truth["indicators"] = {
                        labels[i]: tech_row[i] for i in range(len(labels)) if tech_row[i] is not None
                    }
            except Exception:
                pass

            # Fundamentals
            try:
                fund_row = db.execute(
                    "SELECT pe_ratio, market_cap, forward_pe, peg_ratio, price_to_book, "
                    "profit_margin, revenue_growth, debt_to_equity, beta "
                    "FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                    [ticker],
                ).fetchone()
                if fund_row:
                    fund_labels = ["pe_ratio", "market_cap", "forward_pe", "peg_ratio",
                                   "price_to_book", "profit_margin", "revenue_growth",
                                   "debt_to_equity", "beta"]
                    for i, label in enumerate(fund_labels):
                        if fund_row[i] is not None:
                            ground_truth[label] = fund_row[i]
            except Exception:
                pass

        if not ground_truth:
            return json.dumps(
                {
                    "status": "inconclusive",
                    "ticker": ticker,
                    "claim": claim,
                    "message": "No ground-truth data found for this ticker. Cannot verify.",
                }
            )

        return json.dumps(
            {
                "status": "success",
                "ticker": ticker,
                "claim": claim,
                "ground_truth": ground_truth,
                "instruction": (
                    "Compare the claim against the ground_truth data. "
                    "If the claim cites numbers not present in ground_truth, it may be hallucinated."
                ),
            }
        )
    except Exception as e:
        logger.exception("[PipelineTools] check_hallucination failed for %s", ticker)
        return json.dumps({"status": "error", "ticker": ticker, "message": str(e)})


# ── Tool 4: Strategy Performance Lookup ────────────────────────────────
@registry.register(
    name="get_strategy_performance",
    description=(
        "Get historical trading performance for a specific ticker. "
        "Returns past trade outcomes, win rate, and P&L to inform whether "
        "this stock has been profitable or lossy in previous cycles."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker to check performance for.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of past trades to return. Default 10.",
            },
        },
        "required": ["ticker"],
    },
    tier=1,
    source="internal_db",
    tags=["performance", "strategy", "history", "pnl", "win-rate"],
)
async def get_strategy_performance(ticker: str, limit: int = 10) -> str:
    """Look up historical trading performance for a ticker."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            rows = db.execute(
                """SELECT (result_json::jsonb)->>'action' AS action, confidence, (result_json::jsonb)->>'rationale' AS rationale, created_at
                   FROM analysis_results
                   WHERE ticker = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                [ticker, limit],
            ).fetchall()

        if not rows:
            return json.dumps(
                {
                    "status": "success",
                    "ticker": ticker,
                    "message": "No historical trades found for this ticker.",
                    "trades": [],
                }
            )

        trades = []
        for r in rows:
            trades.append(
                {
                    "action": r[0],
                    "confidence": r[1],
                    "rationale": (r[2] or "")[:200],
                    "date": r[3],
                }
            )

        # Compute basic stats
        buy_count = sum(
            1 for t in trades if t["action"] and t["action"].upper() == "BUY"
        )
        sell_count = sum(
            1 for t in trades if t["action"] and t["action"].upper() == "SELL"
        )
        hold_count = sum(
            1 for t in trades if t["action"] and t["action"].upper() == "HOLD"
        )
        avg_confidence = (
            sum(t["confidence"] for t in trades if t["confidence"]) / len(trades)
            if trades
            else 0
        )

        return json.dumps(
            {
                "status": "success",
                "ticker": ticker,
                "total_trades": len(trades),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
                "avg_confidence": round(avg_confidence, 1),
                "recent_trades": trades[:5],
            }
        )
    except Exception as e:
        logger.exception(
            "[PipelineTools] get_strategy_performance failed for %s", ticker
        )
        return json.dumps({"status": "error", "ticker": ticker, "message": str(e)})


# ── Tool 5: AutoResearch Report Lookup ─────────────────────────────────
@registry.register(
    name="get_autoresearch_report",
    description=(
        "Get the latest AutoResearch report, which contains automated audit results "
        "including data quality scores, decision quality scores, LLM performance, "
        "and AI-generated recommendations for system improvement."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cycle_id": {
                "type": "string",
                "description": "Optional: specific cycle_id. If omitted, returns the latest report.",
            },
        },
        "required": [],
    },
    tier=1,
    source="internal_db",
    tags=["autoresearch", "report", "audit", "recommendations"],
)
async def get_autoresearch_report(cycle_id: str = "") -> str:
    """Retrieve an AutoResearch audit report."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            if cycle_id:
                row = db.execute(
                    "SELECT * FROM autoresearch_reports WHERE cycle_id = %s LIMIT 1",
                    [cycle_id],
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM autoresearch_reports ORDER BY rowid DESC LIMIT 1"
                ).fetchone()

        if not row:
            return json.dumps(
                {
                    "status": "success",
                    "message": "No AutoResearch reports found.",
                }
            )

        # Convert row to dict (database returns tuples)
        columns = [
            "id",
            "cycle_id",
            "data_quality_score",
            "decision_quality_score",
            "llm_performance_score",
            "overall_score",
            "data_gaps",
            "decision_issues",
            "llm_issues",
            "performance_metrics",
            "reflection",
            "recovery_stats",
            "status",
        ]
        report = {}
        for i, col in enumerate(columns):
            if i < len(row):
                val = row[i]
                # Try to parse JSON strings
                if isinstance(val, str) and val.startswith(("[", "{")):
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        pass
                report[col] = val

        return json.dumps({"status": "success", **report})
    except Exception as e:
        logger.exception("[PipelineTools] get_autoresearch_report failed")
        return json.dumps({"status": "error", "message": str(e)})


# ── Tool 6: Trigger Deep Research (Mid-Cycle Autoresearch) ─────────────
@registry.register(
    name="trigger_deep_research",
    description=(
        "Trigger a mid-cycle deep research pass for a specific ticker. "
        "Use this when you identify critical data gaps (e.g., missing fundamentals or news) "
        "and need to wait for the data collectors to fill the gaps before making a decision."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker to research.",
            },
            "cycle_id": {
                "type": "string",
                "description": "The current cycle ID.",
            },
        },
        "required": ["ticker", "cycle_id"],
    },
    tier=1,
    source="internal_db",
    tags=["research", "gaps", "collect", "autoresearch"],
)
async def trigger_deep_research(ticker: str, cycle_id: str) -> str:
    """Invoke mid-cycle partial autoresearch to fill data gaps."""
    try:
        from app.pipeline.analysis.autoresearch import run_partial_autoresearch
        from app.pipeline.analysis.dynamic_tool_router import resolve_missing_data

        # Run partial autoresearch to find gaps
        result = await run_partial_autoresearch(cycle_id, [ticker])

        # Dynamically trigger data collectors for the found gaps
        gaps = result.get("data_quality", {}).get("gaps", [])
        if gaps:
            for gap in gaps:
                if gap.get("ticker") == ticker:
                    missing_fields = gap.get("missing_sources", [])
                    await resolve_missing_data(ticker, missing_fields)
                    return json.dumps(
                        {
                            "status": "success",
                            "ticker": ticker,
                            "message": f"Triggered deep research and data collection for missing fields: {missing_fields}",
                            "action_required": "Please use read tools (e.g., get_market_data, get_fundamentals) to verify the new data.",
                        }
                    )

        return json.dumps(
            {
                "status": "success",
                "ticker": ticker,
                "message": "Deep research complete. No critical data gaps found.",
            }
        )
    except Exception as e:
        logger.exception("[PipelineTools] trigger_deep_research failed for %s", ticker)
        return json.dumps({"status": "error", "ticker": ticker, "message": str(e)})


# ── Tool 7: Search Trading Skills (Lazy-Load Skills) ───────────────────
@registry.register(
    name="search_trading_skills",
    description=(
        "Search and load specific trading skills or sector instructions mid-cycle. "
        "Use this to dynamically fetch expert analysis guidelines for a specific stock or sector."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker to load skills for.",
            },
        },
        "required": ["ticker"],
    },
    tier=1,
    source="internal_db",
    tags=["skills", "instructions", "sector", "strategy"],
)
async def search_trading_skills(ticker: str) -> str:
    """Lazy-load sector-specific trading skills for the agent."""
    try:
        from app.services.trading_skills import load_skill_for_ticker

        skill_content = load_skill_for_ticker(ticker)
        if skill_content:
            return json.dumps(
                {
                    "status": "success",
                    "ticker": ticker,
                    "skill_instructions": skill_content,
                }
            )

        return json.dumps(
            {
                "status": "success",
                "ticker": ticker,
                "message": "No specific trading skills found for this ticker or sector.",
            }
        )
    except Exception as e:
        logger.exception("[PipelineTools] search_trading_skills failed for %s", ticker)
        return json.dumps({"status": "error", "ticker": ticker, "message": str(e)})



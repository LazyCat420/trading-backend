"""
Risk Assessment Tools — Agentic Risk Environment for Portfolio Sizing.

Provides tools that give agents full visibility into macro regime, brain graph
connections, and portfolio state in a single composite call. This eliminates
the need for 3+ separate tool calls when the allocator needs to assess the
risk environment before sizing positions.

Design:
  - get_market_regime: Wraps market_regime.py as a callable tool
  - query_brain_graph: Fetches pre-computed ontology subgraph for tickers
  - assess_risk_environment: Composite tool that bundles regime + graph + portfolio
    state + constitution rules into a single response. The tool_playbook system
    will learn this composite pattern over cycles via AutoResearch.

References:
  - Agentic Risk Management (FLAG-TRADER, SAPPO frameworks, 2025)
  - RLHF-style feedback via AutoResearch → Benchmark Agent → Constitution amendments
"""

import json
import logging
from typing import Optional

from app.tools.registry import registry
from app.config import settings

logger = logging.getLogger(__name__)


# ── Tool: get_market_regime ──────────────────────────────────────────────
@registry.register(
    name="get_market_regime",
    description=(
        "Classify the current macro market regime as BULL, BEAR, or SIDEWAYS "
        "using SPY trend vs SMA200, VIX levels, golden cross, and 20-day returns. "
        "Returns the regime label, a bull_score (0-100), a position_multiplier "
        "(1.0 for BULL, 0.5 for SIDEWAYS, 0.3 for BEAR), and supporting metrics. "
        "Use this to adjust position sizing aggressiveness based on macro conditions."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    tier=0,
    source="computed",
    tags=["regime", "macro", "vix", "spy", "risk", "sizing"],
)
async def get_market_regime_tool() -> str:
    """Return the current market regime classification as JSON."""
    try:
        from app.processors.market_regime import get_market_regime

        regime_data = get_market_regime()

        return json.dumps({
            "status": "success",
            **regime_data,
            "sizing_guidance": _regime_sizing_guidance(regime_data.get("regime", "UNKNOWN")),
        })
    except Exception as e:
        logger.exception("[RiskTools] get_market_regime failed")
        return json.dumps({"status": "error", "message": str(e)})


def _regime_sizing_guidance(regime: str) -> str:
    """Return human-readable sizing guidance for the current regime."""
    guidance = {
        "BULL": (
            "Market is in BULL regime. Normal position sizing is appropriate. "
            "Favor BUY signals with conviction. Position multiplier: 1.0x."
        ),
        "SIDEWAYS": (
            "Market is SIDEWAYS. Reduce position sizes by ~50%. Be selective — "
            "only take high-conviction BUY signals. Position multiplier: 0.5x."
        ),
        "BEAR": (
            "Market is in BEAR regime. Reduce position sizes by ~70%. "
            "Favor capital preservation and defensive/SELL postures. "
            "Position multiplier: 0.3x. Require extra conviction for any BUY."
        ),
        "UNKNOWN": (
            "Market regime is UNKNOWN (no SPY data). Default to cautious mode. "
            "Position multiplier: 0.5x."
        ),
    }
    return guidance.get(regime, guidance["UNKNOWN"])


# ── Tool: query_brain_graph ──────────────────────────────────────────────
@registry.register(
    name="query_brain_graph",
    description=(
        "Query the Brain Graph (ontology) to retrieve sector connections, "
        "supply chain relationships, competitor links, and correlation data "
        "for a list of tickers. Returns pre-computed subgraph data showing "
        "how tickers are connected in the knowledge graph. Use this to identify "
        "portfolio concentration risks and hidden correlations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of ticker symbols to query connections for.",
            },
        },
        "required": ["tickers"],
    },
    tier=1,
    source="internal_db",
    tags=["ontology", "brain_graph", "sector", "correlation", "risk"],
)
async def query_brain_graph_tool(tickers: list[str]) -> str:
    """Fetch brain graph connections for the given tickers."""
    try:
        from app.db.connection import get_db

        results = {}
        with get_db() as db:
            for ticker in tickers[:10]:  # Cap at 10 tickers to limit query time
                ticker = ticker.upper().strip()
                connections = []

                # Fetch direct ontology edges
                try:
                    rows = db.execute(
                        """
                        SELECT n2.label, n2.node_type, e.edge_type, e.weight
                        FROM ontology_edges e
                        JOIN ontology_nodes n1 ON e.source_id = n1.id
                        JOIN ontology_nodes n2 ON e.target_id = n2.id
                        WHERE n1.label = %s
                        ORDER BY e.weight DESC
                        LIMIT 15
                        """,
                        [ticker],
                    ).fetchall()

                    for row in rows:
                        connections.append({
                            "connected_to": row[0],
                            "type": row[1],
                            "relationship": row[2],
                            "weight": round(row[3], 3) if row[3] else 0,
                        })
                except Exception:
                    pass

                # Fetch correlation data with other tickers in the batch
                correlations = []
                try:
                    if len(tickers) > 1:
                        other_tickers = [t.upper().strip() for t in tickers if t.upper().strip() != ticker]
                        if other_tickers:
                            placeholders = ",".join(["%s"] * len(other_tickers))
                            rows = db.execute(
                                f"""
                                SELECT ticker_a, ticker_b, correlation
                                FROM ticker_correlations
                                WHERE (ticker_a = %s AND ticker_b IN ({placeholders}))
                                   OR (ticker_b = %s AND ticker_a IN ({placeholders}))
                                ORDER BY ABS(correlation) DESC
                                """,
                                [ticker] + other_tickers + [ticker] + other_tickers,
                            ).fetchall()

                            for row in rows:
                                other = row[1] if row[0] == ticker else row[0]
                                correlations.append({
                                    "ticker": other,
                                    "correlation": round(row[2], 3) if row[2] else 0,
                                })
                except Exception:
                    pass

                # Fetch sector info
                sector = "Unknown"
                try:
                    row = db.execute(
                        "SELECT sector FROM ticker_metadata WHERE ticker = %s",
                        [ticker],
                    ).fetchone()
                    if row:
                        sector = row[0]
                except Exception:
                    pass

                results[ticker] = {
                    "sector": sector,
                    "graph_connections": connections,
                    "correlations": correlations,
                }

        return json.dumps({
            "status": "success",
            "tickers_queried": len(results),
            "graph_data": results,
        })
    except Exception as e:
        logger.exception("[RiskTools] query_brain_graph failed")
        return json.dumps({"status": "error", "message": str(e)})


# ── Tool: assess_risk_environment (COMPOSITE) ────────────────────────────
@registry.register(
    name="assess_risk_environment",
    description=(
        "COMPOSITE TOOL: Assess the full risk environment in a single call. "
        "Bundles market regime classification, brain graph connections, "
        "portfolio state, and Trading Constitution sizing rules into one "
        "unified response. This saves 3-4 separate tool calls and gives "
        "you everything needed to make informed position sizing decisions. "
        "RECOMMENDED: Call this FIRST before any sizing calculations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of ticker symbols being evaluated for trades.",
            },
        },
        "required": ["tickers"],
    },
    tier=0,
    source="computed",
    tags=["composite", "regime", "risk", "sizing", "portfolio", "constitution"],
)
async def assess_risk_environment_tool(tickers: list[str]) -> str:
    """Composite tool: regime + brain graph + portfolio + constitution in one shot."""
    try:
        # 1. Market Regime
        regime_data = {}
        try:
            from app.processors.market_regime import get_market_regime
            regime_data = get_market_regime()
        except Exception as e:
            logger.warning("[RiskTools] Regime fetch failed: %s", e)
            regime_data = {"regime": "UNKNOWN", "position_multiplier": 0.5}

        # 2. Portfolio State
        portfolio_data = {}
        try:
            from app.trading.paper_trader import get_portfolio
            portfolio = get_portfolio(settings.BOT_ID)
            positions = portfolio.get("positions", [])

            # Enrich with current prices
            from app.tools.portfolio_tools import get_position_context
            enriched = []
            for pos in positions:
                ctx = get_position_context(pos["ticker"], settings.BOT_ID)
                enriched.append({
                    "ticker": pos["ticker"],
                    "qty": pos["qty"],
                    "avg_entry": pos["avg_entry_price"],
                    "current_price": ctx.get("current_price"),
                    "pnl_pct": ctx.get("unrealized_pnl_pct", 0),
                    "holding_days": ctx.get("holding_days", 0),
                })

            # Sector breakdown
            sectors = {}
            try:
                from app.db.connection import get_db
                with get_db() as db:
                    for pos in enriched:
                        row = db.execute(
                            "SELECT sector FROM ticker_metadata WHERE ticker = %s",
                            [pos["ticker"]],
                        ).fetchone()
                        sector = row[0] if row else "Unknown"
                        sectors[sector] = sectors.get(sector, 0) + 1
            except Exception:
                pass

            portfolio_data = {
                "cash": portfolio["cash"],
                "total_pnl": portfolio["total_pnl"],
                "position_count": len(enriched),
                "positions": enriched,
                "sector_breakdown": sectors,
            }
        except Exception as e:
            logger.warning("[RiskTools] Portfolio fetch failed: %s", e)
            portfolio_data = {"error": str(e)}

        # 3. Brain Graph Connections
        graph_data = {}
        try:
            graph_result = await query_brain_graph_tool(tickers)
            graph_parsed = json.loads(graph_result)
            if graph_parsed.get("status") == "success":
                graph_data = graph_parsed.get("graph_data", {})
        except Exception as e:
            logger.warning("[RiskTools] Brain graph fetch failed: %s", e)

        # 4. Trading Constitution Sizing Rules
        constitution_rules = {}
        try:
            from app.pipeline.trading_constitution import load_constitution, get_constitution_param

            # Get specific sizing parameters
            constitution_rules = {
                "max_positions": get_constitution_param("position_limits", "max_positions", 8),
                "max_sector_pct": get_constitution_param("sector", "max_sector_pct", 30.0),
                "min_sizing_pct": get_constitution_param("sizing", "min_pct", 2),
                "max_sizing_pct": get_constitution_param("sizing", "max_pct", 15),
                "min_confidence_threshold": get_constitution_param("sizing", "min_confidence", 70),
            }

            # Load full rules for context
            all_rules = load_constitution()
            sizing_rules = [r for r in all_rules if r.get("rule_category") in ("sizing", "position_limits", "sector")]
            constitution_rules["active_rules"] = [
                {"category": r["rule_category"], "text": r["rule_text"], "params": r["rule_params"]}
                for r in sizing_rules
            ]
        except Exception as e:
            logger.warning("[RiskTools] Constitution fetch failed: %s", e)
            constitution_rules = {
                "max_positions": 8,
                "max_sector_pct": 30.0,
                "min_sizing_pct": 2,
                "max_sizing_pct": 15,
                "min_confidence_threshold": 70,
                "note": "Using defaults — constitution unavailable",
            }

        # 5. Compose the unified response
        regime = regime_data.get("regime", "UNKNOWN")
        multiplier = regime_data.get("position_multiplier", 0.5)

        # Calculate regime-adjusted sizing bounds
        min_size = constitution_rules.get("min_sizing_pct", 2)
        max_size = constitution_rules.get("max_sizing_pct", 15)
        regime_adjusted_max = round(max_size * multiplier, 1)

        return json.dumps({
            "status": "success",

            "market_regime": {
                "regime": regime,
                "bull_score": regime_data.get("bull_score", 0),
                "position_multiplier": multiplier,
                "spy_price": regime_data.get("spy_price"),
                "vix": regime_data.get("vix"),
                "sizing_guidance": _regime_sizing_guidance(regime),
            },

            "portfolio": portfolio_data,

            "brain_graph": graph_data,

            "constitution_sizing_rules": constitution_rules,

            "regime_adjusted_sizing": {
                "min_pct": min_size,
                "max_pct": regime_adjusted_max,
                "raw_max_pct": max_size,
                "multiplier_applied": multiplier,
                "explanation": (
                    f"Constitution allows {min_size}%-{max_size}% sizing. "
                    f"Regime ({regime}) applies {multiplier}x multiplier → "
                    f"effective range: {min_size}%-{regime_adjusted_max}%."
                ),
            },
        })

    except Exception as e:
        logger.exception("[RiskTools] assess_risk_environment failed")
        return json.dumps({"status": "error", "message": str(e)})

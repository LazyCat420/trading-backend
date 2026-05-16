"""
Ontology Schema — Unified type system for the Trading Brain Graph.

Supports multiple graph families (code, trading flow, config, runtime,
feature lineage, execution, risk, provenance) inside one typed graph.

Every edge carries a ConfidenceLevel so hard facts are never confused
with inferred relationships.

Build order (phased rollout):
    Phase 1: Code dependency + Trading flow + Config impact
    Phase 2: Risk control + Execution path + Feature lineage
    Phase 3: Runtime incidents + Alerts + Backtest/live provenance
    Phase 4: Clustering, overlap, inference layers
"""

from enum import Enum
from typing import Dict, Any, FrozenSet


# ── Confidence / provenance classification ──────────────────────────────


class ConfidenceLevel(str, Enum):
    """How an edge was established — never mix inferred with hard facts."""

    FACT = "fact"  # from code, config, or registry (deterministic)
    DERIVED = "derived"  # computed from data (clustering, overlap scoring)
    INFERRED = "inferred"  # LLM / heuristic guess (may be wrong)


class GraphFamily(str, Enum):
    """Logical graph families — used for filtering and typed queries."""

    CODE = "code"  # code dependency graph
    TRADING_FLOW = "trading_flow"  # market event → signal → strategy → execution
    CONFIG = "config"  # config blast radius graph
    RUNTIME = "runtime"  # failure / incident graph
    FEATURE_LINEAGE = "feature"  # feature derivation lineage
    EXECUTION = "execution"  # order routing / execution path
    RISK = "risk"  # risk control graph
    PROVENANCE = "provenance"  # backtest / live provenance
    MARKET_ANALYSIS = "market"  # original market-analysis graph (v1)
    KNOWLEDGE = "knowledge"  # LLM-generated claims / hypotheses


# ── Node types ──────────────────────────────────────────────────────────
# Grouped by family.  A node may participate in multiple graph families.


class NodeType(str, Enum):
    # ── Market analysis (original v1) ──
    ASSET = "Asset"
    COMPANY = "Company"
    SECTOR = "Sector"
    INDUSTRY = "Industry"
    THEME = "Theme"
    EVENT = "Event"
    CLAIM = "Claim"
    SOURCE = "Source"
    PERSON = "Person"
    INSTITUTION = "Institution"
    RISK = "Risk"
    SIGNAL = "Signal"
    MACRO_REGIME = "MacroRegime"
    TRADE_DECISION = "TradeDecision"
    HYPOTHESIS = "Hypothesis"

    # ── Code dependency ──
    FILE = "File"
    FUNCTION = "Function"
    CLASS = "Class"
    MODULE = "Module"
    CONFIG_KEY = "ConfigKey"

    # ── Trading domain ──
    STRATEGY = "Strategy"
    FEATURE = "Feature"
    INDICATOR = "Indicator"
    RISK_RULE = "RiskRule"
    EXECUTION_PATH = "ExecutionPath"
    VENUE = "Venue"
    DATA_FEED = "DataFeed"
    ORDER_EVENT = "OrderEvent"
    MARKET_EVENT = "MarketEvent"
    EXECUTION_STEP = "ExecutionStep"

    # ── Runtime / incident ──
    ALERT = "Alert"
    EXCEPTION = "Exception"
    INCIDENT = "Incident"
    METRIC_SNAPSHOT = "MetricSnapshot"
    SERVICE_COMPONENT = "ServiceComponent"

    # ── Provenance / experiment ──
    BACKTEST_RUN = "BacktestRun"
    DEPLOYMENT = "Deployment"
    MODEL_VERSION = "ModelVersion"
    STRATEGY_VERSION = "StrategyVersion"


# ── Edge types ──────────────────────────────────────────────────────────
# Grouped by family.


class EdgeType(str, Enum):
    # ── Market analysis (original v1) ──
    MENTIONS = "MENTIONS"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    BELONGS_TO = "BELONGS_TO"
    COMPETES_WITH = "COMPETES_WITH"
    SUPPLIES = "SUPPLIES"
    IMPACTS = "IMPACTS"
    EXPOSED_TO = "EXPOSED_TO"
    HELD_BY = "HELD_BY"
    PREDICTED = "PREDICTED"
    RESOLVED_AS = "RESOLVED_AS"
    CAUSES = "CAUSES"
    CORRELATES_WITH = "CORRELATES_WITH"
    LEADS_LAGS = "LEADS_LAGS"

    # ── Code / structural ──
    CALLS = "CALLS"
    IMPORTS = "IMPORTS"
    DEFINES = "DEFINES"
    USES_CONFIG = "USES_CONFIG"

    # ── Trading flow / domain ──
    PRODUCES = "PRODUCES"
    CONSUMES = "CONSUMES"
    GATES = "GATES"
    ROUTES_TO = "ROUTES_TO"
    TRIGGERS = "TRIGGERS"
    AFFECTS = "AFFECTS"
    DEFAULTED_BY = "DEFAULTED_BY"
    OVERRIDDEN_BY = "OVERRIDDEN_BY"
    SHARES_FEATURE_WITH = "SHARES_FEATURE_WITH"
    TRADES_MARKET = "TRADES_MARKET"
    USES_SIGNAL = "USES_SIGNAL"

    # ── Feature lineage ──
    DERIVED_FROM = "DERIVED_FROM"
    NORMALIZED_FROM = "NORMALIZED_FROM"
    CONSUMED_BY = "CONSUMED_BY"

    # ── Runtime / incident ──
    TRIGGERED_ALERT = "TRIGGERED_ALERT"
    FAILED_WITH = "FAILED_WITH"
    ALERTED = "ALERTED"
    DEGRADED = "DEGRADED"
    EXECUTED_ON = "EXECUTED_ON"
    CAUSED_BY = "CAUSED_BY"
    DEPENDS_ON = "DEPENDS_ON"

    # ── Execution path ──
    SUBMITS_TO = "SUBMITS_TO"
    ACKED_BY = "ACKED_BY"
    FILLED_BY = "FILLED_BY"
    REJECTED_BY = "REJECTED_BY"

    # ── Risk control ──
    GATED_BY = "GATED_BY"
    LIMITED_BY = "LIMITED_BY"
    DISABLED_BY = "DISABLED_BY"

    # ── Provenance ──
    TESTED_BY = "TESTED_BY"
    DEPLOYED_AS = "DEPLOYED_AS"
    GENERATED = "GENERATED"
    DRIFTED_FROM = "DRIFTED_FROM"
    PRODUCED_RESULT = "PRODUCED_RESULT"


# ── Node lifecycle classification ───────────────────────────────────────
# PERMANENT nodes are never decayed or pruned.
# EPHEMERAL nodes decay and get pruned by the knowledge purge.

PERMANENT_NODE_TYPES: FrozenSet[NodeType] = frozenset(
    {
        # Market analysis structural
        NodeType.ASSET,
        NodeType.COMPANY,
        NodeType.SECTOR,
        NodeType.INDUSTRY,
        NodeType.SOURCE,
        NodeType.PERSON,
        NodeType.INSTITUTION,
        # Code structural
        NodeType.FILE,
        NodeType.FUNCTION,
        NodeType.CLASS,
        NodeType.MODULE,
        NodeType.CONFIG_KEY,
        # Trading infrastructure
        NodeType.STRATEGY,
        NodeType.FEATURE,
        NodeType.INDICATOR,
        NodeType.RISK_RULE,
        NodeType.EXECUTION_PATH,
        NodeType.VENUE,
        NodeType.DATA_FEED,
        NodeType.EXECUTION_STEP,
        # Runtime infrastructure
        NodeType.SERVICE_COMPONENT,
    }
)

EPHEMERAL_NODE_TYPES: FrozenSet[NodeType] = frozenset(
    {
        # LLM-generated knowledge
        NodeType.CLAIM,
        NodeType.SIGNAL,
        NodeType.HYPOTHESIS,
        NodeType.THEME,
        NodeType.EVENT,
        NodeType.RISK,
        NodeType.TRADE_DECISION,
        NodeType.MACRO_REGIME,
        # Runtime events (transient)
        NodeType.ALERT,
        NodeType.EXCEPTION,
        NodeType.INCIDENT,
        NodeType.METRIC_SNAPSHOT,
        NodeType.ORDER_EVENT,
        NodeType.MARKET_EVENT,
        # Provenance snapshots
        NodeType.BACKTEST_RUN,
        NodeType.DEPLOYMENT,
        NodeType.MODEL_VERSION,
        NodeType.STRATEGY_VERSION,
    }
)


# ── Property schemas for documentation / validation ────────────────────

EDGE_PROPERTIES: Dict[EdgeType, Dict[str, Any]] = {
    # Market analysis (original)
    EdgeType.CORRELATES_WITH: {"correlation": float, "period": str},
    EdgeType.LEADS_LAGS: {"lead_days": int, "correlation": float},
    EdgeType.BELONGS_TO: {"weight": float},
    EdgeType.MENTIONS: {"sentiment": float},
    # Code
    EdgeType.CALLS: {"call_count": int, "is_async": bool},
    EdgeType.IMPORTS: {"import_path": str},
    EdgeType.USES_CONFIG: {"default_value": str},
    # Trading flow
    EdgeType.PRODUCES: {"output_type": str},
    EdgeType.CONSUMES: {"input_type": str},
    EdgeType.GATES: {"condition": str},
    EdgeType.ROUTES_TO: {"priority": int},
    # Feature lineage
    EdgeType.DERIVED_FROM: {"transform": str, "lag": int},
    EdgeType.NORMALIZED_FROM: {"method": str},
    # Runtime
    EdgeType.CAUSED_BY: {"root_cause": bool},
    EdgeType.DEPENDS_ON: {"criticality": str},
    EdgeType.DEGRADED: {"severity": str, "duration_s": float},
    # Execution
    EdgeType.FILLED_BY: {"fill_pct": float},
    EdgeType.REJECTED_BY: {"reject_reason": str},
    # Risk
    EdgeType.GATED_BY: {"threshold": float},
    EdgeType.LIMITED_BY: {"limit_value": float, "limit_type": str},
    # Provenance
    EdgeType.DRIFTED_FROM: {"drift_metric": str, "drift_value": float},
    EdgeType.PRODUCED_RESULT: {"sharpe": float, "pnl": float},
    # Shared
    EdgeType.SHARES_FEATURE_WITH: {"overlap_score": float},
    EdgeType.TRADES_MARKET: {"position_type": str},
}

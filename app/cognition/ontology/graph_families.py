"""
Graph Families — Registry of logical graph families in the unified brain.

Each family defines:
    - which node types participate
    - which edge types participate
    - the default confidence level for edges in this family
    - sample questions the family can answer

Usage:
    from app.cognition.ontology.graph_families import GRAPH_FAMILIES, get_family
    fam = get_family(GraphFamily.TRADING_FLOW)
    print(fam.questions)
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List

from app.cognition.ontology.schema import (
    ConfidenceLevel,
    EdgeType,
    GraphFamily,
    NodeType,
)


@dataclass(frozen=True)
class GraphFamilySpec:
    """Specification for a logical graph family."""

    family: GraphFamily
    description: str
    node_types: FrozenSet[NodeType]
    edge_types: FrozenSet[EdgeType]
    default_confidence: ConfidenceLevel
    phase: int  # build phase (1–4)
    questions: List[str] = field(default_factory=list)


# ── Phase 1 families ────────────────────────────────────────────────────

CODE_FAMILY = GraphFamilySpec(
    family=GraphFamily.CODE,
    description="Code dependency graph — foundation for safe edits and tracing",
    node_types=frozenset(
        {
            NodeType.FILE,
            NodeType.FUNCTION,
            NodeType.CLASS,
            NodeType.MODULE,
            NodeType.CONFIG_KEY,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.CALLS,
            EdgeType.IMPORTS,
            EdgeType.DEFINES,
            EdgeType.USES_CONFIG,
        }
    ),
    default_confidence=ConfidenceLevel.FACT,
    phase=1,
    questions=[
        "What depends on this function?",
        "Which modules import this file?",
        "What configs does this strategy use?",
    ],
)

TRADING_FLOW_FAMILY = GraphFamilySpec(
    family=GraphFamily.TRADING_FLOW,
    description="Trading flow graph — the actual bot brain",
    node_types=frozenset(
        {
            NodeType.MARKET_EVENT,
            NodeType.FEATURE,
            NodeType.SIGNAL,
            NodeType.STRATEGY,
            NodeType.RISK_RULE,
            NodeType.EXECUTION_STEP,
            NodeType.ORDER_EVENT,
            NodeType.INDICATOR,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.PRODUCES,
            EdgeType.CONSUMES,
            EdgeType.GATES,
            EdgeType.ROUTES_TO,
            EdgeType.TRIGGERS,
            EdgeType.USES_SIGNAL,
            EdgeType.TRADES_MARKET,
        }
    ),
    default_confidence=ConfidenceLevel.FACT,
    phase=1,
    questions=[
        "What strategies use this feature?",
        "What signal triggers this order?",
        "What is the full path from market event to execution?",
    ],
)

CONFIG_FAMILY = GraphFamilySpec(
    family=GraphFamily.CONFIG,
    description="Config blast radius graph — what breaks when a config changes",
    node_types=frozenset(
        {
            NodeType.CONFIG_KEY,
            NodeType.STRATEGY,
            NodeType.RISK_RULE,
            NodeType.EXECUTION_PATH,
            NodeType.SERVICE_COMPONENT,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.AFFECTS,
            EdgeType.USES_CONFIG,
            EdgeType.DEFAULTED_BY,
            EdgeType.OVERRIDDEN_BY,
        }
    ),
    default_confidence=ConfidenceLevel.FACT,
    phase=1,
    questions=[
        "What breaks if I change this config key?",
        "What configs affect order sizing?",
        "Which strategies are affected by this risk setting?",
    ],
)

# ── Phase 2 families ────────────────────────────────────────────────────

RISK_FAMILY = GraphFamilySpec(
    family=GraphFamily.RISK,
    description="Risk control graph — gating, limits, kill switches",
    node_types=frozenset(
        {
            NodeType.STRATEGY,
            NodeType.RISK_RULE,
            NodeType.EXECUTION_PATH,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.GATED_BY,
            EdgeType.LIMITED_BY,
            EdgeType.DISABLED_BY,
            EdgeType.OVERRIDDEN_BY,
        }
    ),
    default_confidence=ConfidenceLevel.FACT,
    phase=2,
    questions=[
        "What risk rules gate this strategy?",
        "What breaks if I disable this risk limit?",
        "Which strategies share the same risk rule?",
    ],
)

EXECUTION_FAMILY = GraphFamilySpec(
    family=GraphFamily.EXECUTION,
    description="Execution path graph — order routing and fills",
    node_types=frozenset(
        {
            NodeType.STRATEGY,
            NodeType.EXECUTION_PATH,
            NodeType.VENUE,
            NodeType.ORDER_EVENT,
            NodeType.EXECUTION_STEP,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.SUBMITS_TO,
            EdgeType.ROUTES_TO,
            EdgeType.ACKED_BY,
            EdgeType.FILLED_BY,
            EdgeType.REJECTED_BY,
        }
    ),
    default_confidence=ConfidenceLevel.FACT,
    phase=2,
    questions=[
        "Where does this order get routed?",
        "Which venue rejected this order and why?",
        "What is the full execution path for ETH perpetuals?",
    ],
)

FEATURE_LINEAGE_FAMILY = GraphFamilySpec(
    family=GraphFamily.FEATURE_LINEAGE,
    description="Feature lineage graph — data derivation and feed tracking",
    node_types=frozenset(
        {
            NodeType.DATA_FEED,
            NodeType.FEATURE,
            NodeType.INDICATOR,
            NodeType.SIGNAL,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.DERIVED_FROM,
            EdgeType.NORMALIZED_FROM,
            EdgeType.CONSUMED_BY,
        }
    ),
    default_confidence=ConfidenceLevel.FACT,
    phase=2,
    questions=[
        "Where does this feature come from?",
        "What feeds into this indicator?",
        "Which signals depend on this raw feed?",
    ],
)

# ── Phase 3 families ────────────────────────────────────────────────────

RUNTIME_FAMILY = GraphFamilySpec(
    family=GraphFamily.RUNTIME,
    description="Runtime failure / incident graph — why did trading degrade?",
    node_types=frozenset(
        {
            NodeType.EXCEPTION,
            NodeType.ALERT,
            NodeType.SERVICE_COMPONENT,
            NodeType.STRATEGY,
            NodeType.VENUE,
            NodeType.DATA_FEED,
            NodeType.INCIDENT,
            NodeType.METRIC_SNAPSHOT,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.CAUSED_BY,
            EdgeType.TRIGGERED_ALERT,
            EdgeType.DEGRADED,
            EdgeType.DEPENDS_ON,
            EdgeType.FAILED_WITH,
            EdgeType.ALERTED,
            EdgeType.EXECUTED_ON,
        }
    ),
    default_confidence=ConfidenceLevel.DERIVED,
    phase=3,
    questions=[
        "Why is this strategy idle?",
        "Which feed failure is blocking trading?",
        "What incidents impacted live execution today?",
    ],
)

PROVENANCE_FAMILY = GraphFamilySpec(
    family=GraphFamily.PROVENANCE,
    description="Backtest/live provenance graph — why didn't live match backtest?",
    node_types=frozenset(
        {
            NodeType.STRATEGY_VERSION,
            NodeType.MODEL_VERSION,
            NodeType.BACKTEST_RUN,
            NodeType.DEPLOYMENT,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.TESTED_BY,
            EdgeType.DEPLOYED_AS,
            EdgeType.PRODUCED_RESULT,
            EdgeType.DRIFTED_FROM,
            EdgeType.GENERATED,
        }
    ),
    default_confidence=ConfidenceLevel.FACT,
    phase=3,
    questions=[
        "What config version was used in this backtest?",
        "Why did live performance drift from backtest?",
        "Which model version produced this result set?",
    ],
)

# ── Phase 4 families (original v1 + LLM knowledge) ─────────────────────

MARKET_ANALYSIS_FAMILY = GraphFamilySpec(
    family=GraphFamily.MARKET_ANALYSIS,
    description="Original market analysis graph — assets, sectors, correlations",
    node_types=frozenset(
        {
            NodeType.ASSET,
            NodeType.COMPANY,
            NodeType.SECTOR,
            NodeType.INDUSTRY,
            NodeType.THEME,
            NodeType.EVENT,
            NodeType.SOURCE,
            NodeType.PERSON,
            NodeType.INSTITUTION,
            NodeType.MACRO_REGIME,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.MENTIONS,
            EdgeType.BELONGS_TO,
            EdgeType.COMPETES_WITH,
            EdgeType.SUPPLIES,
            EdgeType.IMPACTS,
            EdgeType.EXPOSED_TO,
            EdgeType.HELD_BY,
            EdgeType.CORRELATES_WITH,
            EdgeType.LEADS_LAGS,
        }
    ),
    default_confidence=ConfidenceLevel.DERIVED,
    phase=4,
    questions=[
        "What sectors does this company belong to?",
        "What assets are correlated with NVDA?",
        "Who holds a significant position in this stock?",
    ],
)

KNOWLEDGE_FAMILY = GraphFamilySpec(
    family=GraphFamily.KNOWLEDGE,
    description="LLM-generated knowledge — claims, hypotheses, trade decisions",
    node_types=frozenset(
        {
            NodeType.CLAIM,
            NodeType.SIGNAL,
            NodeType.HYPOTHESIS,
            NodeType.TRADE_DECISION,
            NodeType.RISK,
        }
    ),
    edge_types=frozenset(
        {
            EdgeType.SUPPORTS,
            EdgeType.CONTRADICTS,
            EdgeType.PREDICTED,
            EdgeType.RESOLVED_AS,
            EdgeType.CAUSES,
            EdgeType.SHARES_FEATURE_WITH,
        }
    ),
    default_confidence=ConfidenceLevel.INFERRED,
    phase=4,
    questions=[
        "What claims support this trade decision?",
        "Are there contradicting hypotheses for this ticker?",
        "What is the evidence chain behind this prediction?",
    ],
)


# ── Registry ────────────────────────────────────────────────────────────

GRAPH_FAMILIES: Dict[GraphFamily, GraphFamilySpec] = {
    spec.family: spec
    for spec in [
        CODE_FAMILY,
        TRADING_FLOW_FAMILY,
        CONFIG_FAMILY,
        RISK_FAMILY,
        EXECUTION_FAMILY,
        FEATURE_LINEAGE_FAMILY,
        RUNTIME_FAMILY,
        PROVENANCE_FAMILY,
        MARKET_ANALYSIS_FAMILY,
        KNOWLEDGE_FAMILY,
    ]
}


def get_family(family: GraphFamily) -> GraphFamilySpec:
    """Look up a graph family spec by enum value."""
    return GRAPH_FAMILIES[family]


def get_families_for_node(node_type: NodeType) -> List[GraphFamilySpec]:
    """Return all graph families that include a given node type."""
    return [spec for spec in GRAPH_FAMILIES.values() if node_type in spec.node_types]


def get_families_for_edge(edge_type: EdgeType) -> List[GraphFamilySpec]:
    """Return all graph families that include a given edge type."""
    return [spec for spec in GRAPH_FAMILIES.values() if edge_type in spec.edge_types]


def get_phase_families(phase: int) -> List[GraphFamilySpec]:
    """Return all graph families scheduled for a given build phase."""
    return [spec for spec in GRAPH_FAMILIES.values() if spec.phase == phase]

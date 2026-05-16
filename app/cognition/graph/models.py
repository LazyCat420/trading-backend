"""
Graph Models — Dataclasses for the unified trading brain graph.

Supports confidence levels (fact / derived / inferred) and graph-family
tagging so queries can be scoped to a single logical graph family.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class GraphNode:
    id: str  # deterministic UUID from (entity_type, canonical_name)
    entity_type: str  # NodeType enum value
    canonical_name: str  # resolved, deduplicated name
    properties: Dict[str, Any]  # type-specific metadata (JSON-serialized)
    created_at: str
    updated_at: str
    graph_family: Optional[str] = (
        None  # GraphFamily enum value (nullable for v1 compat)
    )
    version: Optional[str] = None  # semantic version / commit hash for provenance


@dataclass
class GraphEdge:
    id: str  # deterministic UUID from (source_id, target_id, edge_type)
    source_id: str  # FK -> GraphNode.id
    target_id: str  # FK -> GraphNode.id
    edge_type: str  # EdgeType enum value
    properties: Dict[str, Any]  # edge-specific metadata
    confidence: float  # 0.0-1.0
    source_ref: Optional[str]  # provenance (which pipeline step created this)
    created_at: str
    updated_at: str
    confidence_level: str = "fact"  # ConfidenceLevel: "fact" | "derived" | "inferred"
    graph_family: Optional[str] = None  # GraphFamily enum value
    version: Optional[str] = None  # version tag for tracking drift


@dataclass
class GraphSubgraph:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    graph_family: Optional[str] = None  # optional scope


@dataclass
class GraphWriteResult:
    nodes_created: int
    nodes_updated: int
    edges_created: int
    edges_updated: int


@dataclass
class ResolvedEntity:
    entity_id: str
    canonical_name: str
    entity_type: str
    confidence: float  # resolution confidence
    aliases_matched: List[str]


@dataclass
class GraphNeighborhood:
    center: GraphNode
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    depth: int


@dataclass
class ClaimRef:
    claim_id: str
    summary: str
    confidence: float
    supporting_sources: List[str]



@dataclass
class GraphFamilyStats:
    """Summary statistics for a single graph family."""

    family: str
    node_count: int = 0
    edge_count: int = 0
    fact_edges: int = 0
    derived_edges: int = 0
    inferred_edges: int = 0

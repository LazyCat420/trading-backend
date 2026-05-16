"""
Graph Learn Tool — RLM-injectable function for LLM to create graph edges.

The LLM calls this during analysis when it discovers associations:
    graph_learn(
        edges=[{"source": "NVDA", "target": "AMD", "relation": "CAUSES",
                "weight": 0.8, "reason": "AI capex driving both"}]
    )

Validates against NodeType/EdgeType enums. Writes to ontology_nodes/edges
via BrainGraph.upsert_node()/upsert_edge() with source_cycle_id tracking.

Injected into the RLM REPL via rlm_tools.py TRADING_TOOLS dict.
"""

import json
import logging
from datetime import datetime, timezone

from app.cognition.ontology.schema import EdgeType, NodeType, PERMANENT_NODE_TYPES
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# Valid types the LLM is allowed to create.
# Structural / permanent node types (File, Asset, Sector, Strategy, etc.) are
# managed by extraction pipelines, not the LLM.
_LLM_NODE_TYPES = {nt.value for nt in NodeType if nt not in PERMANENT_NODE_TYPES}

_VALID_RELATIONS = {e.value for e in EdgeType}


def graph_learn(
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
) -> dict:
    """Record new associations or insights into the brain graph.

    Args:
        nodes: List of node dicts: [{"id": str, "type": str, "label": str, "metadata": dict}]
               type must be one of: Claim, Signal, Hypothesis, Theme, Event, Risk
        edges: List of edge dicts: [{"source": str, "target": str, "relation": str,
                                      "weight": float, "reason": str}]
               relation must be a valid EdgeType enum value

    Returns:
        dict with counts of nodes/edges created and any validation errors.
    """
    nodes = nodes or []
    edges = edges or []

    created_nodes = 0
    created_edges = 0
    errors: list[str] = []

    with get_db() as db:
        now = datetime.now(timezone.utc)

        # ── Create nodes ──
        for node_spec in nodes[:10]:  # Cap at 10 nodes per call
            try:
                node_id = str(node_spec.get("id", "")).strip()
                node_type = str(node_spec.get("type", "")).strip()
                label = str(node_spec.get("label", node_id)).strip()
                metadata = node_spec.get("metadata", {})

                if not node_id:
                    errors.append("Node missing 'id'")
                    continue
                if node_type not in _LLM_NODE_TYPES:
                    errors.append(
                        f"Node type '{node_type}' not allowed. "
                        f"Use one of: {', '.join(sorted(_LLM_NODE_TYPES))}"
                    )
                    continue

                meta_json = json.dumps(metadata) if metadata else None

                existing = db.execute(
                    "SELECT id FROM ontology_nodes WHERE id = %s", [node_id]
                ).fetchone()

                if existing:
                    db.execute(
                        "UPDATE ontology_nodes SET node_type=%s, label=%s, "
                        "metadata_json=%s, updated_at=%s WHERE id=%s",
                        [node_type, label[:80], meta_json, now, node_id],
                    )
                else:
                    db.execute(
                        "INSERT INTO ontology_nodes "
                        "(id, node_type, label, activation, metadata_json, "
                        "created_at, updated_at) "
                        "VALUES (%s, %s, %s, 0.0, %s, %s, %s)",
                        [node_id, node_type, label[:80], meta_json, now, now],
                    )
                created_nodes += 1

            except Exception as e:
                errors.append(f"Node error: {e}")

        # ── Create edges ──
        for edge_spec in edges[:10]:  # Cap at 10 edges per call
            try:
                source = str(edge_spec.get("source", "")).strip()
                target = str(edge_spec.get("target", "")).strip()
                relation = str(edge_spec.get("relation", "")).strip().upper()
                weight = float(edge_spec.get("weight", 0.5))
                reason = str(edge_spec.get("reason", ""))

                if not source or not target:
                    errors.append("Edge missing 'source' or 'target'")
                    continue
                if relation not in _VALID_RELATIONS:
                    errors.append(
                        f"Relation '{relation}' not valid. "
                        f"Use one of: {', '.join(sorted(_VALID_RELATIONS))}"
                    )
                    continue

                weight = max(0.0, min(1.0, weight))

                # Use BrainGraph.upsert_edge for EMA reinforcement
                from app.cognition.ontology.ontology_builder import BrainGraph

                BrainGraph.upsert_edge(
                    source_id=source,
                    target_id=target,
                    relation=relation,
                    weight=weight,
                    metadata={"reason": reason[:200], "origin": "llm_graph_learn"},
                )
                created_edges += 1

            except Exception as e:
                errors.append(f"Edge error: {e}")

        result = {
            "nodes_created": created_nodes,
            "edges_created": created_edges,
            "errors": errors if errors else None,
        }

        if created_nodes or created_edges:
            logger.info(
                "[GRAPH LEARN] LLM created %d nodes, %d edges",
                created_nodes,
                created_edges,
            )

        return result

"""
Graph Writer — Operations to modify the unified brain graph.

Supports confidence_level, graph_family, and version metadata on
both nodes and edges for the multi-family graph system.
"""

import json
import logging
from datetime import datetime, timezone
from app.db.connection import get_db
from app.cognition.graph.models import (
    GraphNode,
    GraphEdge,
    GraphSubgraph,
    GraphWriteResult,
)
from app.cognition.graph.storage import _ensure_schema

logger = logging.getLogger(__name__)


def upsert_node(node: GraphNode) -> str:
    """Insert or update a node. Returns the node ID."""
    _ensure_schema()
    with get_db() as db:
        props_json = json.dumps(node.properties)
        now_str = datetime.now(timezone.utc).isoformat()

        row = db.execute(
            "SELECT id FROM cognition.ontology_entities WHERE id = %s", [node.id]
        ).fetchone()
        if row:
            db.execute(
                """UPDATE cognition.ontology_entities 
                   SET entity_type = %s, canonical_name = %s, properties = %s,
                       graph_family = %s, version = %s, updated_at = %s
                   WHERE id = %s""",
                [
                    node.entity_type,
                    node.canonical_name,
                    props_json,
                    node.graph_family,
                    node.version,
                    now_str,
                    node.id,
                ],
            )
        else:
            db.execute(
                """INSERT INTO cognition.ontology_entities
                   (id, entity_type, canonical_name, properties, graph_family, version, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    node.id,
                    node.entity_type,
                    node.canonical_name,
                    props_json,
                    node.graph_family,
                    node.version,
                    node.created_at or now_str,
                    node.updated_at or now_str,
                ],
            )
        return node.id


def upsert_edge(edge: GraphEdge) -> str:
    """Insert or update an edge. Returns the edge ID."""
    _ensure_schema()
    with get_db() as db:
        props_json = json.dumps(edge.properties)
        now_str = datetime.now(timezone.utc).isoformat()

        row = db.execute(
            "SELECT id FROM cognition.ontology_edges WHERE id = %s", [edge.id]
        ).fetchone()
        if row:
            db.execute(
                """UPDATE cognition.ontology_edges
                   SET source_id = %s, target_id = %s, edge_type = %s, properties = %s,
                       confidence = %s, confidence_level = %s, source_ref = %s,
                       graph_family = %s, version = %s, updated_at = %s
                   WHERE id = %s""",
                [
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    props_json,
                    edge.confidence,
                    edge.confidence_level,
                    edge.source_ref,
                    edge.graph_family,
                    edge.version,
                    now_str,
                    edge.id,
                ],
            )
        else:
            db.execute(
                """INSERT INTO cognition.ontology_edges
                   (id, source_id, target_id, edge_type, properties, confidence,
                    confidence_level, source_ref, graph_family, version, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    edge.id,
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    props_json,
                    edge.confidence,
                    edge.confidence_level,
                    edge.source_ref,
                    edge.graph_family,
                    edge.version,
                    edge.created_at or now_str,
                    edge.updated_at or now_str,
                ],
            )
        return edge.id


def upsert_subgraph(subgraph: GraphSubgraph) -> GraphWriteResult:
    """Batch upsert nodes then edges. Returns counts."""
    nodes_c = 0
    nodes_u = 0
    edges_c = 0
    edges_u = 0
    _ensure_schema()
    with get_db() as db:
        db.execute("BEGIN TRANSACTION;")
        try:
            now_str = datetime.now(timezone.utc).isoformat()
            for node in subgraph.nodes:
                props_json = json.dumps(node.properties)
                row = db.execute(
                    "SELECT id FROM cognition.ontology_entities WHERE id = %s",
                    [node.id],
                ).fetchone()
                if row:
                    db.execute(
                        """UPDATE cognition.ontology_entities 
                           SET entity_type = %s, canonical_name = %s, properties = %s,
                               graph_family = %s, version = %s, updated_at = %s
                           WHERE id = %s""",
                        [
                            node.entity_type,
                            node.canonical_name,
                            props_json,
                            node.graph_family or subgraph.graph_family,
                            node.version,
                            now_str,
                            node.id,
                        ],
                    )
                    nodes_u += 1
                else:
                    db.execute(
                        """INSERT INTO cognition.ontology_entities
                           (id, entity_type, canonical_name, properties, graph_family, version, created_at, updated_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        [
                            node.id,
                            node.entity_type,
                            node.canonical_name,
                            props_json,
                            node.graph_family or subgraph.graph_family,
                            node.version,
                            node.created_at or now_str,
                            node.updated_at or now_str,
                        ],
                    )
                    nodes_c += 1

            for edge in subgraph.edges:
                props_json = json.dumps(edge.properties)
                row = db.execute(
                    "SELECT id FROM cognition.ontology_edges WHERE id = %s", [edge.id]
                ).fetchone()
                if row:
                    db.execute(
                        """UPDATE cognition.ontology_edges
                           SET source_id = %s, target_id = %s, edge_type = %s, properties = %s,
                               confidence = %s, confidence_level = %s, source_ref = %s,
                               graph_family = %s, version = %s, updated_at = %s
                           WHERE id = %s""",
                        [
                            edge.source_id,
                            edge.target_id,
                            edge.edge_type,
                            props_json,
                            edge.confidence,
                            edge.confidence_level,
                            edge.source_ref,
                            edge.graph_family or subgraph.graph_family,
                            edge.version,
                            now_str,
                            edge.id,
                        ],
                    )
                    edges_u += 1
                else:
                    db.execute(
                        """INSERT INTO cognition.ontology_edges
                           (id, source_id, target_id, edge_type, properties, confidence,
                            confidence_level, source_ref, graph_family, version, created_at, updated_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        [
                            edge.id,
                            edge.source_id,
                            edge.target_id,
                            edge.edge_type,
                            props_json,
                            edge.confidence,
                            edge.confidence_level,
                            edge.source_ref,
                            edge.graph_family or subgraph.graph_family,
                            edge.version,
                            edge.created_at or now_str,
                            edge.updated_at or now_str,
                        ],
                    )
                    edges_c += 1
            db.execute("COMMIT;")
        except Exception as e:
            db.execute("ROLLBACK;")
            raise e

        return GraphWriteResult(
            nodes_created=nodes_c,
            nodes_updated=nodes_u,
            edges_created=edges_c,
            edges_updated=edges_u,
        )

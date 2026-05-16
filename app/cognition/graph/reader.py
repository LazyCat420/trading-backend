"""
Graph Reader — Read queries and traversals for the unified brain graph.

Supports optional graph_family scoping so queries can target a single
logical graph family (e.g. only code dependency edges, or only runtime
incident edges).
"""

import json
import logging
from typing import List, Optional
from app.db.connection import get_db
from app.cognition.graph.models import GraphNode, GraphEdge, GraphNeighborhood, ClaimRef
from app.cognition.contracts.retrieval import ContradictionRef
from app.cognition.graph.storage import _ensure_schema

logger = logging.getLogger(__name__)


def _row_to_node(row) -> GraphNode:
    props = json.loads(row[3]) if row[3] else {}
    return GraphNode(
        id=row[0],
        entity_type=row[1],
        canonical_name=row[2],
        properties=props,
        created_at=row[4].isoformat() if hasattr(row[4], "isoformat") else row[4],
        updated_at=row[5].isoformat() if hasattr(row[5], "isoformat") else row[5],
        graph_family=row[6] if len(row) > 6 else None,
        version=row[7] if len(row) > 7 else None,
    )


def _row_to_edge(row) -> GraphEdge:
    props = json.loads(row[4]) if row[4] else {}
    return GraphEdge(
        id=row[0],
        source_id=row[1],
        target_id=row[2],
        edge_type=row[3],
        properties=props,
        confidence=row[5],
        source_ref=row[6],
        created_at=row[7].isoformat() if hasattr(row[7], "isoformat") else row[7],
        updated_at=row[8].isoformat() if hasattr(row[8], "isoformat") else row[8],
        confidence_level=row[9] if len(row) > 9 else "fact",
        graph_family=row[10] if len(row) > 10 else None,
        version=row[11] if len(row) > 11 else None,
    )


def get_local_neighborhood(
    entity_id: str, hops: int = 2, graph_family: Optional[str] = None
) -> GraphNeighborhood:
    """BFS traversal from entity_id up to N hops. Returns all nodes + edges.

    Args:
        entity_id: Starting node ID.
        hops: Number of BFS hops (default 2).
        graph_family: If provided, only traverse edges in this graph family.
    """
    _ensure_schema()
    with get_db() as db:
        center_row = db.execute(
            "SELECT * FROM cognition.ontology_entities WHERE id = %s", [entity_id]
        ).fetchone()
        if not center_row:
            raise ValueError(f"Entity not found: {entity_id}")

        center_node = _row_to_node(center_row)

        frontier = {entity_id}
        visited_nodes = {entity_id: center_node}
        visited_edges = {}

        family_filter = ""
        family_params: list = []
        if graph_family:
            family_filter = " AND graph_family = %s"
            family_params = [graph_family]

        for _ in range(hops):
            if not frontier:
                break

            placeholders = ",".join(["%s"] * len(frontier))
            query = f"""
                SELECT * FROM cognition.ontology_edges 
                WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders})){family_filter}
            """
            params = list(frontier) + list(frontier) + family_params
            edges_rows = db.execute(query, params).fetchall()

            next_frontier = set()

            for row in edges_rows:
                edge = _row_to_edge(row)
                visited_edges[edge.id] = edge
                if edge.source_id not in visited_nodes:
                    next_frontier.add(edge.source_id)
                if edge.target_id not in visited_nodes:
                    next_frontier.add(edge.target_id)

            if next_frontier:
                nf_placeholders = ",".join(["%s"] * len(next_frontier))
                nodes_rows = db.execute(
                    f"""
                    SELECT * FROM cognition.ontology_entities
                    WHERE id IN ({nf_placeholders})
                """,
                    list(next_frontier),
                ).fetchall()

                for row in nodes_rows:
                    node = _row_to_node(row)
                    visited_nodes[node.id] = node

            frontier = next_frontier

        return GraphNeighborhood(
            center=center_node,
            nodes=list(visited_nodes.values()),
            edges=list(visited_edges.values()),
            depth=hops,
        )


def get_related_claims(entity_id: str) -> List[ClaimRef]:
    """All Claim nodes connected to entity_id via any edge."""
    _ensure_schema()
    with get_db() as db:
        rows = db.execute(
            """
            SELECT e.id, e.canonical_name, e.properties 
            FROM cognition.ontology_entities e
            JOIN cognition.ontology_edges eg 
              ON (eg.source_id = e.id OR eg.target_id = e.id)
            WHERE (eg.source_id = %s OR eg.target_id = %s)
              AND e.entity_type = 'Claim'
        """,
            [entity_id, entity_id],
        ).fetchall()

        res = []
        # Using a set to deduplicate claims if multiple edges exist
        seen = set()
        for r in rows:
            cid = r[0]
            if cid in seen:
                continue
            seen.add(cid)
            props = json.loads(r[2]) if r[2] else {}
            res.append(
                ClaimRef(
                    claim_id=cid,
                    summary=r[1],
                    confidence=props.get("confidence", 1.0),
                    supporting_sources=props.get("sources", []),
                )
            )
        return res


def get_contradictions(entity_id: str) -> List[ContradictionRef]:
    """All pairs of Claims connected to entity_id that have CONTRADICTS edges."""
    _ensure_schema()
    with get_db() as db:
        # We find claims connected to entity_id
        # Then we find CONTRADICTS edges between them
        rows = db.execute(
            """
            WITH EntityClaims AS (
                SELECT DISTINCT e.id, e.canonical_name
                FROM cognition.ontology_entities e
                JOIN cognition.ontology_edges eg 
                  ON (eg.source_id = e.id OR eg.target_id = e.id)
                WHERE (eg.source_id = %s OR eg.target_id = %s)
                  AND e.entity_type = 'Claim'
            )
            SELECT eg.source_id, eg.target_id, eg.confidence, 
                   c1.canonical_name, c2.canonical_name
            FROM cognition.ontology_edges eg
            JOIN EntityClaims c1 ON c1.id = eg.source_id
            JOIN EntityClaims c2 ON c2.id = eg.target_id
            WHERE eg.edge_type = 'CONTRADICTS'
        """,
            [entity_id, entity_id],
        ).fetchall()

        res = []
        for r in rows:
            res.append(
                ContradictionRef(
                    description=f"{r[3]} vs {r[4]}",
                    source_ref_1=r[0],
                    source_ref_2=r[1],
                    severity=f"confidence={r[2]:.2f}",
                )
            )
        return res

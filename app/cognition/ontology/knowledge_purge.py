"""
Knowledge Purge — Automated lifecycle management for Living Graph.

Runs as a post-cycle hook alongside existing run_purge_pass().
Implements the 3-tier knowledge lifecycle:
    1. Accumulation: LLM creates Claims, Signals, Hypotheses
    2. Validation: Outcomes strengthen/weaken via reinforcement
    3. Autopurge: Stale, low-confidence, or disproven knowledge pruned

Uses PERMANENT_NODE_TYPES and EPHEMERAL_NODE_TYPES from schema.py as the
canonical source of truth for lifecycle classification.

Usage:
    from app.cognition.ontology.knowledge_purge import purge_stale_knowledge
    result = await purge_stale_knowledge()
"""

import logging
from datetime import datetime, timezone, timedelta

from app.db.connection import get_db
from app.cognition.ontology.schema import EPHEMERAL_NODE_TYPES

logger = logging.getLogger(__name__)

# Decay settings
STALE_DAYS = 7  # Edges not reinforced in 7 days start decaying
DECAY_RATE = 0.05  # 5% weight loss per purge cycle
EDGE_KILL_THRESHOLD = 0.05  # Edges below 5% weight are removed
DISPROVE_RATIO = 3  # contradicted > 3x validated → disproven


async def purge_stale_knowledge() -> dict:
    """Remove low-confidence, unvalidated, or stale graph knowledge.

    Lifecycle rules:
        1. Time decay: LLM-created edges not reinforced in 7 days lose 5% weight
        2. Kill dead edges: weight < 0.05
        3. Kill orphan ephemeral nodes with no edges
        4. Contradiction resolution: contradicted > 3x validated → disproven
        5. Kill disproven Claims with very low activation

    Returns dict with counts of decayed/killed/disproven items.
    """
    try:
        with get_db() as db:
            now = datetime.now(timezone.utc)
            stale_cutoff = now - timedelta(days=STALE_DAYS)

            decayed_edges = 0
            killed_edges = 0
            killed_nodes = 0
            disproven_nodes = 0

            # ── 1. Time decay: weaken stale ephemeral edges ──
            # Only decay edges that connect to ephemeral nodes
            _eph_list = [nt.value for nt in EPHEMERAL_NODE_TYPES]
            ephemeral_ids = db.execute(
                "SELECT id FROM ontology_nodes WHERE node_type IN "
                f"({','.join('%s' for _ in _eph_list)})",
                _eph_list,
            ).fetchall()
            ephemeral_set = {r[0] for r in ephemeral_ids}

            if ephemeral_set:
                stale_edges = db.execute(
                    "SELECT id, weight, source_id, target_id FROM ontology_edges "
                    "WHERE updated_at < %s AND weight > %s",
                    [stale_cutoff, EDGE_KILL_THRESHOLD],
                ).fetchall()

                for edge_id, weight, src, tgt in stale_edges:
                    # Only decay if at least one endpoint is ephemeral
                    if src in ephemeral_set or tgt in ephemeral_set:
                        new_weight = max(0.0, weight - DECAY_RATE)
                        db.execute(
                            "UPDATE ontology_edges SET weight = %s, updated_at = %s "
                            "WHERE id = %s",
                            [new_weight, now, edge_id],
                        )
                        decayed_edges += 1

            # ── 2. Kill dead edges (weight below threshold) ──
            dead = db.execute(
                "SELECT id FROM ontology_edges WHERE weight < %s",
                [EDGE_KILL_THRESHOLD],
            ).fetchall()
            for (edge_id,) in dead:
                db.execute("DELETE FROM ontology_edges WHERE id = %s", [edge_id])
                killed_edges += 1

            # ── 3. Contradiction resolution: auto-disprove ──
            contradict_rows = db.execute(
                "SELECT id, validated_count, contradicted_count FROM ontology_nodes "
                "WHERE node_type = 'Claim' "
                "AND (disproven IS NULL OR disproven = FALSE) "
                "AND contradicted_count > %s * validated_count + 1",
                [DISPROVE_RATIO],
            ).fetchall()

            for claim_id, v, c in contradict_rows:
                db.execute(
                    "UPDATE ontology_nodes SET disproven = TRUE, updated_at = %s "
                    "WHERE id = %s",
                    [now, claim_id],
                )
                disproven_nodes += 1
                logger.info(
                    "[PURGE] Disproven claim %s (validated=%d, contradicted=%d)",
                    claim_id[:8],
                    v or 0,
                    c or 0,
                )

            # ── 4. Kill orphan ephemeral nodes (no edges) ──
            _eph_list2 = [nt.value for nt in EPHEMERAL_NODE_TYPES]
            orphans = db.execute(
                "SELECT n.id FROM ontology_nodes n "
                f"WHERE n.node_type IN ({','.join('%s' for _ in _eph_list2)}) "
                "AND NOT EXISTS (SELECT 1 FROM ontology_edges e WHERE e.source_id = n.id) "
                "AND NOT EXISTS (SELECT 1 FROM ontology_edges e WHERE e.target_id = n.id) "
                "AND n.updated_at < %s",
                _eph_list2 + [stale_cutoff],
            ).fetchall()

            for (node_id,) in orphans:
                db.execute("DELETE FROM ontology_nodes WHERE id = %s", [node_id])
                killed_nodes += 1

            result = {
                "decayed_edges": decayed_edges,
                "killed_edges": killed_edges,
                "killed_nodes": killed_nodes,
                "disproven_nodes": disproven_nodes,
            }

            total_ops = decayed_edges + killed_edges + killed_nodes + disproven_nodes
            if total_ops > 0:
                logger.info(
                    "[PURGE] Knowledge purge: decayed=%d edges, killed=%d edges + %d nodes, "
                    "disproven=%d claims",
                    decayed_edges,
                    killed_edges,
                    killed_nodes,
                    disproven_nodes,
                )
            else:
                logger.debug("[PURGE] Knowledge purge: nothing to clean")

        return result

    except Exception as e:
        logger.warning("[PURGE] Knowledge purge failed (non-fatal): %s", e)
        return {"error": str(e)}

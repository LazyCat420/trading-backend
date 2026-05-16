"""
Ontology Builder — Neuro-Symbolic Brain Graph Engine.

Implements a mathematical knowledge graph with:
  1. Vector-weighted association edges (cosine similarity)
  2. Spreading activation for LLM context injection

The graph is stored in PostgreSQL (ontology_nodes / ontology_edges)
and can be queried to retrieve the most relevant sub-graph
for any given ticker before sending it to the LLM.

Mathematical Framework:
  Edge Weight:  W(A,B) = α·cos(E_A, E_B) + β·context_relevance
  Activation:   A_j(t) = Σ_i A_i(t-1) · W(i,j) · γ

References:
  - Neuro-Symbolic AI (Garcez et al., 2023)
  - Spreading Activation in Semantic Networks (Collins & Loftus, 1975)
  - RAG + KG grounding for LLMs (Pan et al., 2024)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from app.cognition.base import BaseCognitionModule
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
SEMANTIC_EMBEDDING_DIM = 384  # sentence-transformer semantic embeddings
STRUCTURAL_EMBEDDING_DIM = 16  # GNNEngine structural embeddings (SVD)
EMBEDDING_DIM = SEMANTIC_EMBEDDING_DIM  # backward compat alias
ALPHA = 0.7  # weight for cosine similarity component
BETA = 0.3  # weight for context relevance component
DEFAULT_DECAY = 0.85  # γ: per-hop activation decay
MAX_ACTIVATION_HOPS = 3
ACTIVATION_THRESHOLD = 0.05  # nodes below this are pruned from subgraph
MAX_SUBGRAPH_NODES = 50


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm < 1e-9:
        return 0.0
    return float(dot / norm)


def _compute_edge_weight(
    embedding_a: Optional[list[float]],
    embedding_b: Optional[list[float]],
    context_relevance: float = 0.5,
) -> float:
    """
    Compute the association strength W(A,B):
      W(A,B) = α · cos(E_A, E_B) + β · context_relevance

    Falls back to context_relevance alone if embeddings are missing.
    """
    if embedding_a and embedding_b:
        dim_a, dim_b = len(embedding_a), len(embedding_b)
        if dim_a != dim_b:
            logger.warning(
                "[BrainGraph] Embedding dimension mismatch: %d vs %d — falling back to context_relevance",
                dim_a,
                dim_b,
            )
            return context_relevance
        if dim_a not in (SEMANTIC_EMBEDDING_DIM, STRUCTURAL_EMBEDDING_DIM):
            logger.warning(
                "[BrainGraph] Unexpected embedding dim %d (expected %d or %d) — falling back",
                dim_a,
                SEMANTIC_EMBEDDING_DIM,
                STRUCTURAL_EMBEDDING_DIM,
            )
            return context_relevance
        cos_sim = _cosine_similarity(embedding_a, embedding_b)
        # Normalize cosine from [-1,1] to [0,1]
        cos_norm = (cos_sim + 1.0) / 2.0
        return ALPHA * cos_norm + BETA * context_relevance
    # No embeddings — use context relevance only
    return context_relevance


class BrainGraph:
    """
    In-memory brain graph engine backed by PostgreSQL.

    Provides CRUD for nodes/edges and the spreading activation algorithm.
    """

    # ── Node CRUD ─────────────────────────────────────────────────────

    @staticmethod
    def upsert_node(
        node_id: str,
        node_type: str,
        label: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert or update an ontology node."""
        now = datetime.now(timezone.utc)
        label = label or node_id
        meta_json = json.dumps(metadata) if metadata else None

        try:
            with get_db() as db:
                existing = db.execute(
                    "SELECT id FROM ontology_nodes WHERE id = %s", [node_id]
                ).fetchone()

                if existing:
                    if embedding:
                        db.execute(
                            "UPDATE ontology_nodes SET node_type=%s, label=%s, embedding=%s, "
                            "metadata_json=%s, updated_at=%s WHERE id=%s",
                            [node_type, label, embedding, meta_json, now, node_id],
                        )
                    else:
                        db.execute(
                            "UPDATE ontology_nodes SET node_type=%s, label=%s, "
                            "metadata_json=%s, updated_at=%s WHERE id=%s",
                            [node_type, label, meta_json, now, node_id],
                        )
                else:
                    db.execute(
                        "INSERT INTO ontology_nodes (id, node_type, label, activation, embedding, metadata_json, created_at, updated_at) "
                        "VALUES (%s, %s, %s, 0.0, %s, %s, %s, %s)",
                        [node_id, node_type, label, embedding, meta_json, now, now],
                    )
        except Exception as e:
            logger.error("[BrainGraph] upsert_node error: %s", e)

    # ── Edge CRUD ─────────────────────────────────────────────────────

    @staticmethod
    def upsert_edge(
        source_id: str,
        target_id: str,
        relation: str,
        weight: Optional[float] = None,
        decay: float = DEFAULT_DECAY,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert or reinforce an ontology edge.

        If the edge already exists, evidence_count is incremented and weight
        is re-averaged (exponential moving average) to strengthen repeated
        associations without clobbering the original.
        """
        now = datetime.now(timezone.utc)
        meta_json = json.dumps(metadata) if metadata else None

        try:
            with get_db() as db:
                # ── Validate that both nodes exist (prevent dangling edges) ────
                src_exists = db.execute(
                    "SELECT 1 FROM ontology_nodes WHERE id = %s", [source_id]
                ).fetchone()
                tgt_exists = db.execute(
                    "SELECT 1 FROM ontology_nodes WHERE id = %s", [target_id]
                ).fetchone()
                if not src_exists or not tgt_exists:
                    missing = []
                    if not src_exists:
                        missing.append(f"source={source_id}")
                    if not tgt_exists:
                        missing.append(f"target={target_id}")
                    logger.error(
                        "[BrainGraph] upsert_edge skipped — node(s) not found: %s",
                        ", ".join(missing),
                    )
                    return

                # Compute weight from embeddings if not provided
                if weight is None:
                    try:
                        src_row = db.execute(
                            "SELECT embedding FROM ontology_nodes WHERE id = %s",
                            [source_id],
                        ).fetchone()
                        tgt_row = db.execute(
                            "SELECT embedding FROM ontology_nodes WHERE id = %s",
                            [target_id],
                        ).fetchone()
                        emb_a = src_row[0] if src_row and src_row[0] else None
                        emb_b = tgt_row[0] if tgt_row and tgt_row[0] else None
                        weight = _compute_edge_weight(emb_a, emb_b)
                    except Exception:
                        weight = 0.5

                existing = db.execute(
                    "SELECT id, weight, evidence_count FROM ontology_edges "
                    "WHERE source_id = %s AND target_id = %s AND relation = %s",
                    [source_id, target_id, relation],
                ).fetchone()

                if existing:
                    old_weight = existing[1]
                    old_count = existing[2]
                    # Exponential moving average: strengthens repeated edges
                    new_weight = min(1.0, 0.7 * old_weight + 0.3 * weight)
                    db.execute(
                        "UPDATE ontology_edges SET weight=%s, evidence_count=%s, "
                        "metadata_json=%s, updated_at=%s WHERE id=%s",
                        [new_weight, old_count + 1, meta_json, now, existing[0]],
                    )
                else:
                    edge_id = str(uuid.uuid4())[:12]
                    db.execute(
                        "INSERT INTO ontology_edges "
                        "(id, source_id, target_id, relation, weight, decay, evidence_count, metadata_json, created_at, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, 1, %s, %s, %s)",
                        [
                            edge_id,
                            source_id,
                            target_id,
                            relation,
                            weight,
                            decay,
                            meta_json,
                            now,
                            now,
                        ],
                    )
        except Exception as e:
            logger.error("[BrainGraph] upsert_edge error: %s", e)

    # ── Spreading Activation ──────────────────────────────────────────

    @staticmethod
    def spreading_activation(
        seed_node_ids: list[str],
        max_hops: int = MAX_ACTIVATION_HOPS,
        threshold: float = ACTIVATION_THRESHOLD,
        max_nodes: int = MAX_SUBGRAPH_NODES,
    ) -> dict:
        """
        Run spreading activation from seed nodes.

        Algorithm (Collins & Loftus, 1975 adapted):
          1. Set seed nodes to activation = 1.0
          2. For each hop t:
             A_j(t) = Σ_i A_i(t-1) · W(i,j) · γ_ij
          3. Prune nodes below threshold
          4. Return the activated subgraph (nodes + edges + activations)

        Returns:
            {
                "nodes": [{id, type, label, activation, metadata}],
                "edges": [{source, target, relation, weight}],
                "stats": {total_activated, hops_used, seed_nodes}
            }
        """
        with get_db() as db:
            # Build adjacency from PostgreSQL — include evidence_count for weight boost
            all_edges = db.execute(
                "SELECT source_id, target_id, weight, decay, relation, evidence_count FROM ontology_edges"
            ).fetchall()

            # ── Fast Numpy GNNEngine Integration ──────
            from app.cognition.ontology.gnn_engine import GNNEngine
            import math

            nodes = list(
                set(
                    [src for src, _, _, _, _, _ in all_edges]
                    + [tgt for _, tgt, _, _, _, _ in all_edges]
                    + seed_node_ids
                )
            )
            # Boost edge weight by evidence_count: effective_w = w * log(1 + evidence_count)
            graph_edges = [
                (src, tgt, min(1.0, w * math.log(1 + (ec or 1))))
                for src, tgt, w, _, _, ec in all_edges
            ]

            try:
                gnn = GNNEngine(nodes, graph_edges)
                # Run graph convolutions
                activations = gnn.message_passing(
                    initial_activations={seed: 1.0 for seed in seed_node_ids},
                    layers=max_hops,
                    decay=DEFAULT_DECAY,
                )
                hops_used = max_hops
            except Exception as e:
                logger.error(f"[BrainGraph] GNNEngine failed, fallback to none: {e}")
                activations = {seed: 1.0 for seed in seed_node_ids}
                hops_used = 1

            # Prune below threshold + limit
            activated = {
                nid: act for nid, act in activations.items() if act >= threshold
            }
            # Sort by activation descending, take top N
            top_nodes = sorted(activated.items(), key=lambda x: -x[1])[:max_nodes]
            top_ids = {nid for nid, _ in top_nodes}

            # Fetch node details
            result_nodes = []
            for nid, act in top_nodes:
                row = db.execute(
                    "SELECT node_type, label, metadata_json, "
                    "validated_count, contradicted_count, disproven "
                    "FROM ontology_nodes WHERE id = %s",
                    [nid],
                ).fetchone()
                if row:
                    meta = json.loads(row[2]) if row[2] else {}
                    # Merge lifecycle columns into metadata for Claim nodes
                    if row[0] == "Claim":
                        meta["validated_count"] = row[3] or 0
                        meta["contradicted_count"] = row[4] or 0
                        meta["disproven"] = bool(row[5])
                    result_nodes.append(
                        {
                            "id": nid,
                            "type": row[0],
                            "label": row[1],
                            "activation": round(act, 4),
                            "metadata": meta or None,
                        }
                    )

        # Fetch relevant edges (both endpoints in subgraph)
        result_edges = []
        for src, tgt, w, _d, rel, _ec in all_edges:
            if src in top_ids and tgt in top_ids:
                result_edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "relation": rel,
                        "weight": round(w, 4),
                    }
                )

        return {
            "nodes": result_nodes,
            "edges": result_edges,
            "stats": {
                "total_activated": len(result_nodes),
                "hops_used": hops_used,
                "seed_nodes": seed_node_ids,
            },
        }

    # ── Context Builder Integration ───────────────────────────────────

    @staticmethod
    def get_activated_context(ticker: str, max_chars: int = 4000) -> str:
        """
        Generate a text summary of the activated subgraph for LLM context injection.

        This replaces the old "dump everything" approach with mathematical
        graph-based relevance filtering.
        """
        result = BrainGraph.spreading_activation(seed_node_ids=[ticker])

        if not result["nodes"]:
            return ""

        lines = [f"## Brain Graph Context for {ticker}"]
        lines.append(
            f"(Activated {result['stats']['total_activated']} nodes across {result['stats']['hops_used']} hops)\n"
        )

        # Group by type
        by_type: dict[str, list] = {}
        for n in result["nodes"]:
            by_type.setdefault(n["type"], []).append(n)

        for ntype, nodes in sorted(by_type.items()):
            lines.append(f"### {ntype}")
            for n in sorted(nodes, key=lambda x: -x["activation"]):
                act_pct = int(n["activation"] * 100)
                meta = n.get("metadata") or {}

                # Claim nodes get validation stats
                if ntype == "Claim":
                    validated = meta.get("validated_count", 0)
                    contradicted = meta.get("contradicted_count", 0)
                    disproven = meta.get("disproven", False)
                    if disproven:
                        lines.append(
                            f"  - ~~{n['label']}~~ (DISPROVEN, v={validated} c={contradicted})"
                        )
                    else:
                        lines.append(
                            f"  - {n['label']} (activation={act_pct}%, "
                            f"validated={validated}x, contradicted={contradicted}x)"
                        )
                else:
                    meta_str = ""
                    if meta:
                        meta_str = f" | {json.dumps(meta)}"
                    lines.append(f"  - {n['label']} (activation={act_pct}%{meta_str})")

        # Add key relationships
        if result["edges"]:
            lines.append("\n### Key Relationships")
            for e in sorted(result["edges"], key=lambda x: -x["weight"])[:20]:
                w_pct = int(e["weight"] * 100)
                lines.append(
                    f"  - {e['source']} --[{e['relation']}]--> {e['target']} (strength={w_pct}%)"
                )

        text = "\n".join(lines)
        return text[:max_chars]

    # ── Bulk Graph Seeding ────────────────────────────────────────────

    @staticmethod
    def seed_from_ticker_metadata(ticker: str) -> int:
        """
        Seed the brain graph with structural nodes from ticker_metadata,
        sector relationships, correlations, and news sources.

        Returns count of nodes+edges created.
        """
        count = 0

        with get_db() as db:
            # ── Core Asset node ───────────────────────────────────────────
            try:
                row = db.execute(
                    "SELECT name, sector, industry, market_cap_tier, asset_class "
                    "FROM ticker_metadata WHERE ticker = %s",
                    [ticker],
                ).fetchone()
            except Exception:
                row = None

        BrainGraph.upsert_node(ticker, "Asset", label=ticker)
        count += 1

        if row:
            # Defensive handling for missing fields or dictionary-like rows
            try:
                if isinstance(row, dict) or hasattr(row, "get"):
                    name = row.get("name")
                    sector = row.get("sector")
                    industry = row.get("industry")
                    cap_tier = row.get("market_cap_tier", "unknown")
                    asset_class = row.get("asset_class", "unknown")
                else:
                    name, sector, industry, cap_tier, asset_class = row
            except ValueError:
                name, sector, industry, cap_tier, asset_class = (
                    ticker,
                    None,
                    None,
                    "unknown",
                    "unknown",
                )

            meta = {
                "market_cap_tier": cap_tier if cap_tier is not None else "unknown",
                "asset_class": asset_class if asset_class is not None else "unknown",
            }
            BrainGraph.upsert_node(ticker, "Asset", label=name or ticker, metadata=meta)

            if sector:
                BrainGraph.upsert_node(sector, "Sector", label=sector)
                BrainGraph.upsert_edge(ticker, sector, "BELONGS_TO", weight=0.9)
                count += 2

            if industry:
                BrainGraph.upsert_node(industry, "Industry", label=industry)
                BrainGraph.upsert_edge(ticker, industry, "BELONGS_TO", weight=0.85)
                count += 2

        with get_db() as db:
            # ── Correlated tickers ────────────────────────────────────────
            try:
                corr_rows = db.execute(
                    "SELECT ticker_b, correlation, tier FROM ticker_correlations "
                    "WHERE ticker_a = %s AND period = '30d' ORDER BY ABS(correlation) DESC LIMIT 5",
                    [ticker],
                ).fetchall()
            except Exception:
                corr_rows = []

        for corr_ticker, corr_val, tier in corr_rows:
            BrainGraph.upsert_node(corr_ticker, "Asset", label=corr_ticker)
            weight = (abs(corr_val) + 1.0) / 2.0  # map [-1,1] -> [0,1]
            BrainGraph.upsert_edge(
                ticker,
                corr_ticker,
                "CORRELATES_WITH",
                weight=weight,
                metadata={"correlation": corr_val, "tier": tier},
            )
            count += 2

        with get_db() as db:
            # ── Recent news as Source nodes ────────────────────────────────
            try:
                news_rows = db.execute(
                    "SELECT id, title, publisher, url FROM news_articles "
                    "WHERE ticker = %s AND quality_status != 'discarded' "
                    "ORDER BY published_at DESC LIMIT 5",
                    [ticker],
                ).fetchall()
            except Exception:
                news_rows = []

        for news_id, title, publisher, url in news_rows:
            node_id = f"news_{news_id[:8]}"
            BrainGraph.upsert_node(
                node_id,
                "Source",
                label=(title or "")[:80],
                metadata={"publisher": publisher, "url": url},
            )
            BrainGraph.upsert_edge(node_id, ticker, "MENTIONS", weight=0.7)
            count += 2

        with get_db() as db:
            # ── Sector peers ──────────────────────────────────────────────
            if row and row[1]:  # sector exists
                try:
                    peer_rows = db.execute(
                        "SELECT ticker FROM ticker_metadata WHERE sector = %s AND ticker != %s LIMIT 5",
                        [row[1], ticker],
                    ).fetchall()
                except Exception:
                    peer_rows = []
            else:
                peer_rows = []

        for (peer,) in peer_rows:
            BrainGraph.upsert_node(peer, "Asset", label=peer)
            BrainGraph.upsert_edge(ticker, peer, "COMPETES_WITH", weight=0.5)
            count += 2

        logger.info("[BrainGraph] Seeded %d nodes+edges for %s", count, ticker)
        return count


class OntologyBuilder(BaseCognitionModule):
    """
    V2 Cognition Module: Ontology Graph Builder.

    Seeds the brain graph for a ticker and runs spreading activation
    to produce a mathematically-weighted context for the LLM.
    """

    def __init__(self):
        super().__init__("OntologyBuilder")

    async def _execute(self, ticker: str, context: dict) -> dict:
        """
        1. Seed the graph with structural data for this ticker.
        2. Run spreading activation from the ticker node.
        3. Return the activated subgraph for downstream consumption.
        """
        # Seed structural relationships
        seeded = BrainGraph.seed_from_ticker_metadata(ticker)

        # Run spreading activation
        subgraph = BrainGraph.spreading_activation(seed_node_ids=[ticker])

        # Generate text context for LLM injection
        context_text = BrainGraph.get_activated_context(ticker)

        return {
            "ticker": ticker,
            "ontology_nodes": subgraph["nodes"],
            "ontology_edges": subgraph["edges"],
            "ontology_stats": subgraph["stats"],
            "ontology_context": context_text,
            "nodes_seeded": seeded,
            "status": "ready",
        }

"""
Graph Mutations — Write-side operations for the Living Graph.

Creates and manages LLM-generated knowledge (Claims, TradeDecisions,
Signals, Hypotheses) and trading-domain entities (Strategies, Features,
RiskRules, Incidents) in the ontology graph.

Ephemeral nodes decay and get pruned; structural nodes persist.

Separated from ontology_builder.py (seeding + activation) to follow
the 1-file-1-job rule.

Usage:
    from app.cognition.ontology.graph_mutations import create_claim
    claim_id = create_claim("NVDA", "RSI < 35 + institutional buying = BUY", cycle_id)
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from app.db.connection import get_db

logger = logging.getLogger(__name__)


def create_claim(
    ticker: str,
    text: str,
    cycle_id: str,
    confidence: float = 0.5,
) -> str | None:
    """Create a Claim node + SUPPORTS edge from ticker to claim.

    Returns the claim node ID if created, None on error.
    """
    try:
        now = datetime.now(timezone.utc)

        # Deterministic ID: hash of ticker + text to avoid duplicates
        claim_id = (
            f"claim_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{ticker}:{text}').hex[:12]}"
        )

        meta = {
            "text": text[:200],
            "ticker": ticker,
            "validated_count": 0,
            "contradicted_count": 0,
        }

        with get_db() as db:
            # Check if claim already exists (same pattern re-observed)
            existing = db.execute(
                "SELECT id, evidence_count FROM ontology_edges "
                "WHERE source_id = %s AND target_id = %s AND relation = 'SUPPORTS'",
                [ticker, claim_id],
            ).fetchone()

            if existing:
                # Reinforce existing claim edge (EMA strengthening)
                old_count = existing[1] or 1
                db.execute(
                    "UPDATE ontology_edges SET evidence_count = %s, updated_at = %s "
                    "WHERE source_id = %s AND target_id = %s AND relation = 'SUPPORTS'",
                    [old_count + 1, now, ticker, claim_id],
                )
                logger.info(
                    "[GRAPH] Reinforced existing claim %s (count=%d)",
                    claim_id[:8],
                    old_count + 1,
                )
                return claim_id

            # Create Claim node
            db.execute(
                "INSERT INTO ontology_nodes "
                "(id, node_type, label, activation, metadata_json, "
                "source_cycle_id, validated_count, contradicted_count, "
                "created_at, updated_at) "
                "VALUES (%s, 'Claim', %s, 0.0, %s, %s, 0, 0, %s, %s)",
                [claim_id, text[:80], json.dumps(meta), cycle_id, now, now],
            )

            # Create SUPPORTS edge: ticker -> claim
            edge_id = str(uuid.uuid4())[:12]
            db.execute(
                "INSERT INTO ontology_edges "
                "(id, source_id, target_id, relation, weight, decay, "
                "evidence_count, metadata_json, source_cycle_id, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, 'SUPPORTS', %s, 0.85, 1, %s, %s, %s, %s)",
                [
                    edge_id,
                    ticker,
                    claim_id,
                    confidence,
                    json.dumps({"origin": "post_cycle_learn"}),
                    cycle_id,
                    now,
                    now,
                ],
            )

        logger.info(
            "[GRAPH] Created Claim %s for %s: %s",
            claim_id[:8],
            ticker,
            text[:60],
        )
        return claim_id

    except Exception as e:
        logger.warning("[GRAPH] create_claim failed for %s: %s", ticker, e)
        return None


def create_trade_decision(
    ticker: str,
    action: str,
    confidence: int,
    cycle_id: str,
    rationale: str = "",
    supporting_claims: list[str] | None = None,
) -> str | None:
    """Create a TradeDecision node + edges from supporting Claims.

    Returns the decision node ID if created, None on error.
    """
    try:
        now = datetime.now(timezone.utc)

        decision_id = f"decision_{cycle_id}_{ticker}".lower()

        meta = {
            "action": action,
            "confidence": confidence,
            "rationale": rationale[:300],
            "ticker": ticker,
        }

        with get_db() as db:
            # Upsert decision node
            existing = db.execute(
                "SELECT id FROM ontology_nodes WHERE id = %s", [decision_id]
            ).fetchone()

            if existing:
                db.execute(
                    "UPDATE ontology_nodes SET metadata_json = %s, updated_at = %s "
                    "WHERE id = %s",
                    [json.dumps(meta), now, decision_id],
                )
            else:
                db.execute(
                    "INSERT INTO ontology_nodes "
                    "(id, node_type, label, activation, metadata_json, "
                    "source_cycle_id, created_at, updated_at) "
                    "VALUES (%s, 'TradeDecision', %s, 0.0, %s, %s, %s, %s)",
                    [
                        decision_id,
                        f"{action} {ticker} @{confidence}%",
                        json.dumps(meta),
                        cycle_id,
                        now,
                        now,
                    ],
                )

            # Edge: ticker -> decision (PREDICTED)
            _upsert_edge_safe(
                db,
                ticker,
                decision_id,
                "PREDICTED",
                weight=confidence / 100.0,
                cycle_id=cycle_id,
                now=now,
            )

            # Edges: each supporting claim -> decision (SUPPORTS)
            for claim_id in supporting_claims or []:
                _upsert_edge_safe(
                    db,
                    claim_id,
                    decision_id,
                    "SUPPORTS",
                    weight=0.7,
                    cycle_id=cycle_id,
                    now=now,
                )

        logger.info(
            "[GRAPH] Created TradeDecision %s: %s %s @%d%%",
            decision_id[:12],
            action,
            ticker,
            confidence,
        )
        return decision_id

    except Exception as e:
        logger.warning("[GRAPH] create_trade_decision failed for %s: %s", ticker, e)
        return None


def reinforce_claim(claim_id: str, outcome: str) -> None:
    """Update validated/contradicted counts + adjust edge weight.

    Args:
        claim_id: The Claim node ID
        outcome: "WIN", "LOSS", or "FLAT"
    """
    try:
        now = datetime.now(timezone.utc)

        with get_db() as db:
            if outcome == "WIN":
                db.execute(
                    "UPDATE ontology_nodes "
                    "SET validated_count = validated_count + 1, updated_at = %s "
                    "WHERE id = %s",
                    [now, claim_id],
                )
                # Strengthen all edges TO this claim
                db.execute(
                    "UPDATE ontology_edges "
                    "SET weight = LEAST(1.0, weight + 0.05), updated_at = %s "
                    "WHERE target_id = %s",
                    [now, claim_id],
                )
                logger.info("[GRAPH] Reinforced claim %s (WIN)", claim_id[:8])

            elif outcome == "LOSS":
                db.execute(
                    "UPDATE ontology_nodes "
                    "SET contradicted_count = contradicted_count + 1, updated_at = %s "
                    "WHERE id = %s",
                    [now, claim_id],
                )
                # Weaken edges faster on LOSS (asymmetric learning)
                db.execute(
                    "UPDATE ontology_edges "
                    "SET weight = GREATEST(0.0, weight - 0.10), updated_at = %s "
                    "WHERE target_id = %s",
                    [now, claim_id],
                )
                logger.info("[GRAPH] Weakened claim %s (LOSS)", claim_id[:8])

                # Auto-disprove: contradicted > 3x validated
                _check_disproven(db, claim_id)

            # FLAT: no change

    except Exception as e:
        logger.warning("[GRAPH] reinforce_claim failed for %s: %s", claim_id, e)


def mark_disproven(claim_id: str) -> None:
    """Set disproven=true on a Claim node."""
    try:
        with get_db() as db:
            now = datetime.now(timezone.utc)
            db.execute(
                "UPDATE ontology_nodes SET disproven = TRUE, updated_at = %s WHERE id = %s",
                [now, claim_id],
            )
        logger.info("[GRAPH] Disproven claim %s", claim_id[:8])
    except Exception as e:
        logger.warning("[GRAPH] mark_disproven failed for %s: %s", claim_id, e)


def get_claims_for_ticker(ticker: str) -> list[str]:
    """Get all non-disproven Claim IDs associated with a ticker."""
    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT DISTINCT e.target_id FROM ontology_edges e "
                "JOIN ontology_nodes n ON n.id = e.target_id "
                "WHERE e.source_id = %s AND n.node_type = 'Claim' "
                "AND (n.disproven IS NULL OR n.disproven = FALSE)",
                [ticker],
            ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logger.warning("[GRAPH] get_claims_for_ticker failed for %s: %s", ticker, e)
        return []


# ── Private helpers ──────────────────────────────────────────────────


def _upsert_edge_safe(
    db,
    source_id: str,
    target_id: str,
    relation: str,
    weight: float = 0.5,
    cycle_id: str = "",
    now=None,
) -> None:
    """Insert or reinforce an edge (safe, no exceptions leaked)."""
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        existing = db.execute(
            "SELECT id, evidence_count FROM ontology_edges "
            "WHERE source_id = %s AND target_id = %s AND relation = %s",
            [source_id, target_id, relation],
        ).fetchone()

        if existing:
            db.execute(
                "UPDATE ontology_edges SET evidence_count = %s, updated_at = %s "
                "WHERE id = %s",
                [existing[1] + 1, now, existing[0]],
            )
        else:
            edge_id = str(uuid.uuid4())[:12]
            db.execute(
                "INSERT INTO ontology_edges "
                "(id, source_id, target_id, relation, weight, decay, "
                "evidence_count, source_cycle_id, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, 0.85, 1, %s, %s, %s)",
                [edge_id, source_id, target_id, relation, weight, cycle_id, now, now],
            )
    except Exception as e:
        logger.debug("[GRAPH] _upsert_edge_safe: %s", e)


def _check_disproven(db, claim_id: str) -> None:
    """Mark claim as disproven if contradicted > 3x validated."""
    try:
        row = db.execute(
            "SELECT validated_count, contradicted_count FROM ontology_nodes "
            "WHERE id = %s AND node_type = 'Claim'",
            [claim_id],
        ).fetchone()
        if row:
            validated = row[0] or 0
            contradicted = row[1] or 0
            if contradicted > 3 * validated + 1:
                mark_disproven(claim_id)
    except Exception:
        pass


# ── Trading flow mutations ───────────────────────────────────────────────


def register_strategy(
    strategy_id: str,
    name: str,
    features: list[str] | None = None,
    signals: list[str] | None = None,
    risk_rules: list[str] | None = None,
    markets: list[str] | None = None,
    cycle_id: str = "",
) -> str | None:
    """Register a strategy and its connections in the trading flow graph.

    Creates the Strategy node and USES_SIGNAL / CONSUMES / GATED_BY /
    TRADES_MARKET edges.  Idempotent — safe to call on every cycle.

    Returns the strategy node ID if created, None on error.
    """
    try:
        now = datetime.now(timezone.utc)
        meta = {"name": name, "features": features or [], "signals": signals or []}

        with get_db() as db:
            existing = db.execute(
                "SELECT id FROM ontology_nodes WHERE id = %s", [strategy_id]
            ).fetchone()

            if existing:
                db.execute(
                    "UPDATE ontology_nodes SET label = %s, metadata_json = %s, updated_at = %s "
                    "WHERE id = %s",
                    [name[:80], json.dumps(meta), now, strategy_id],
                )
            else:
                db.execute(
                    "INSERT INTO ontology_nodes "
                    "(id, node_type, label, activation, metadata_json, "
                    "source_cycle_id, created_at, updated_at) "
                    "VALUES (%s, 'Strategy', %s, 0.0, %s, %s, %s, %s)",
                    [strategy_id, name[:80], json.dumps(meta), cycle_id, now, now],
                )

            # Wire up feature consumption
            for feat_id in features or []:
                _upsert_edge_safe(
                    db,
                    strategy_id,
                    feat_id,
                    "CONSUMES",
                    weight=1.0,
                    cycle_id=cycle_id,
                    now=now,
                )

            # Wire up signal usage
            for sig_id in signals or []:
                _upsert_edge_safe(
                    db,
                    strategy_id,
                    sig_id,
                    "USES_SIGNAL",
                    weight=1.0,
                    cycle_id=cycle_id,
                    now=now,
                )

            # Wire up risk rule gating
            for rule_id in risk_rules or []:
                _upsert_edge_safe(
                    db,
                    strategy_id,
                    rule_id,
                    "GATED_BY",
                    weight=1.0,
                    cycle_id=cycle_id,
                    now=now,
                )

            # Wire up market trading
            for mkt in markets or []:
                _upsert_edge_safe(
                    db,
                    strategy_id,
                    mkt,
                    "TRADES_MARKET",
                    weight=1.0,
                    cycle_id=cycle_id,
                    now=now,
                )

        logger.info(
            "[GRAPH] Registered Strategy %s (%s)",
            strategy_id,
            name,
        )
        return strategy_id

    except Exception as e:
        logger.warning("[GRAPH] register_strategy failed for %s: %s", strategy_id, e)
        return None


# ── Config impact mutations ──────────────────────────────────────────────


def register_config_dependency(
    config_key: str,
    affected_ids: list[str] | None = None,
    default_value: str = "",
    cycle_id: str = "",
) -> str | None:
    """Register a config key and the components it affects.

    Creates a ConfigKey node and AFFECTS edges to each affected component.
    Returns the config node ID.
    """
    try:
        now = datetime.now(timezone.utc)
        node_id = f"config_{uuid.uuid5(uuid.NAMESPACE_DNS, config_key).hex[:12]}"

        meta = {"key": config_key, "default_value": default_value}

        with get_db() as db:
            existing = db.execute(
                "SELECT id FROM ontology_nodes WHERE id = %s", [node_id]
            ).fetchone()

            if existing:
                db.execute(
                    "UPDATE ontology_nodes SET metadata_json = %s, updated_at = %s "
                    "WHERE id = %s",
                    [json.dumps(meta), now, node_id],
                )
            else:
                db.execute(
                    "INSERT INTO ontology_nodes "
                    "(id, node_type, label, activation, metadata_json, "
                    "source_cycle_id, created_at, updated_at) "
                    "VALUES (%s, 'ConfigKey', %s, 0.0, %s, %s, %s, %s)",
                    [node_id, config_key[:80], json.dumps(meta), cycle_id, now, now],
                )

            for target_id in affected_ids or []:
                _upsert_edge_safe(
                    db,
                    node_id,
                    target_id,
                    "AFFECTS",
                    weight=1.0,
                    cycle_id=cycle_id,
                    now=now,
                )

        logger.info(
            "[GRAPH] Registered ConfigKey %s → %d targets",
            config_key,
            len(affected_ids or []),
        )
        return node_id

    except Exception as e:
        logger.warning(
            "[GRAPH] register_config_dependency failed for %s: %s", config_key, e
        )
        return None


# ── Runtime / incident mutations ─────────────────────────────────────────


def create_incident(
    description: str,
    severity: str = "warning",
    affected_components: list[str] | None = None,
    caused_by: str | None = None,
    cycle_id: str = "",
) -> str | None:
    """Create a runtime Incident node.

    Links it to affected components via DEGRADED edges and
    optionally sets a CAUSED_BY edge to the root cause.

    Returns the incident node ID.
    """
    try:
        now = datetime.now(timezone.utc)
        incident_id = f"incident_{uuid.uuid4().hex[:12]}"

        meta = {
            "description": description[:500],
            "severity": severity,
            "timestamp": now.isoformat(),
        }

        with get_db() as db:
            db.execute(
                "INSERT INTO ontology_nodes "
                "(id, node_type, label, activation, metadata_json, "
                "source_cycle_id, created_at, updated_at) "
                "VALUES (%s, 'Incident', %s, 0.0, %s, %s, %s, %s)",
                [incident_id, description[:80], json.dumps(meta), cycle_id, now, now],
            )

            for comp_id in affected_components or []:
                _upsert_edge_safe(
                    db,
                    incident_id,
                    comp_id,
                    "DEGRADED",
                    weight=0.8,
                    cycle_id=cycle_id,
                    now=now,
                )

            if caused_by:
                _upsert_edge_safe(
                    db,
                    incident_id,
                    caused_by,
                    "CAUSED_BY",
                    weight=0.9,
                    cycle_id=cycle_id,
                    now=now,
                )

        logger.info(
            "[GRAPH] Created Incident %s: %s (severity=%s)",
            incident_id[:8],
            description[:60],
            severity,
        )
        return incident_id

    except Exception as e:
        logger.warning("[GRAPH] create_incident failed: %s", e)
        return None


def create_feature_lineage(
    feature_id: str,
    feature_name: str,
    derived_from: list[str] | None = None,
    consumed_by: list[str] | None = None,
    cycle_id: str = "",
) -> str | None:
    """Register a Feature node and its lineage edges.

    Args:
        feature_id: Unique identifier for the feature.
        feature_name: Human-readable name.
        derived_from: List of source feed / feature IDs this is derived from.
        consumed_by: List of strategy / signal IDs that consume this feature.

    Returns the feature node ID.
    """
    try:
        now = datetime.now(timezone.utc)
        meta = {"name": feature_name}

        with get_db() as db:
            existing = db.execute(
                "SELECT id FROM ontology_nodes WHERE id = %s", [feature_id]
            ).fetchone()

            if existing:
                db.execute(
                    "UPDATE ontology_nodes SET label = %s, metadata_json = %s, updated_at = %s "
                    "WHERE id = %s",
                    [feature_name[:80], json.dumps(meta), now, feature_id],
                )
            else:
                db.execute(
                    "INSERT INTO ontology_nodes "
                    "(id, node_type, label, activation, metadata_json, "
                    "source_cycle_id, created_at, updated_at) "
                    "VALUES (%s, 'Feature', %s, 0.0, %s, %s, %s, %s)",
                    [
                        feature_id,
                        feature_name[:80],
                        json.dumps(meta),
                        cycle_id,
                        now,
                        now,
                    ],
                )

            for src_id in derived_from or []:
                _upsert_edge_safe(
                    db,
                    feature_id,
                    src_id,
                    "DERIVED_FROM",
                    weight=1.0,
                    cycle_id=cycle_id,
                    now=now,
                )

            for consumer_id in consumed_by or []:
                _upsert_edge_safe(
                    db,
                    consumer_id,
                    feature_id,
                    "CONSUMES",
                    weight=1.0,
                    cycle_id=cycle_id,
                    now=now,
                )

        logger.info("[GRAPH] Registered Feature %s (%s)", feature_id, feature_name)
        return feature_id

    except Exception as e:
        logger.warning(
            "[GRAPH] create_feature_lineage failed for %s: %s", feature_id, e
        )
        return None

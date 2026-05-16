"""
Evolution Repository — DB operations for the ASI-Evolve strategy evolution system.

All reads/writes go through the shared PostgreSQL connection (app.db.connection).
"""

import json
import logging
import random
from typing import Optional

from app.db.connection import get_db
from app.schemas.evolution import (
    EvolutionNode,
    EvolutionMetrics,
    EvolutionSessionSummary,
)

logger = logging.getLogger(__name__)


def _ensure_table(db):
    """Create evolution_nodes table if it doesn't exist (idempotent)."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS evolution_nodes (
            id              VARCHAR PRIMARY KEY,
            session_id      VARCHAR NOT NULL,
            round           INTEGER NOT NULL,
            parent_id       VARCHAR,
            motivation      VARCHAR,
            code            VARCHAR,
            metrics         VARCHAR,
            score           DOUBLE,
            status          VARCHAR,
            analysis        VARCHAR,
            timestamp       VARCHAR
        )
    """)


def append_node(node: EvolutionNode) -> None:
    """Insert a new evolution node into the DB."""
    with get_db() as db:
        _ensure_table(db)
        metrics_json = node.metrics.model_dump_json() if node.metrics else None
        db.execute(
            "INSERT INTO evolution_nodes "
            "(id, session_id, round, parent_id, motivation, code, metrics, score, status, analysis, timestamp) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                node.id,
                node.session_id,
                node.round,
                node.parent_id,
                node.motivation,
                node.code,
                metrics_json,
                node.score,
                node.status,
                node.analysis,
                node.timestamp,
            ],
        )


def _row_to_node(row) -> EvolutionNode:
    """Convert a DB row tuple to an EvolutionNode."""
    metrics = None
    if row[6]:
        try:
            metrics = EvolutionMetrics(**json.loads(row[6]))
        except Exception:
            pass
    return EvolutionNode(
        id=row[0],
        session_id=row[1],
        round=row[2],
        parent_id=row[3],
        motivation=row[4] or "",
        code=row[5] or "",
        metrics=metrics,
        score=row[7],
        status=row[8] or "DISCARD",
        analysis=row[9] or "",
        timestamp=row[10] or "",
    )


def sample_nodes(
    k: int = 5, strategy: str = "score_weighted", session_id: Optional[str] = None
) -> list[EvolutionNode]:
    """Sample k nodes from the evolution DB.

    Strategies:
        score_weighted — weighted random by score (higher scores sampled more)
        epsilon_greedy — with probability epsilon pick random, else pick top-k by score
    """
    with get_db() as db:
        _ensure_table(db)

        where = "WHERE session_id = %s" if session_id else ""
        params = [session_id] if session_id else []

        rows = db.execute(
            f"SELECT id, session_id, round, parent_id, motivation, code, metrics, "
            f"score, status, analysis, timestamp FROM evolution_nodes {where} "
            f"ORDER BY timestamp DESC LIMIT 200",
            params,
        ).fetchall()

        if not rows:
            return []

        nodes = [_row_to_node(r) for r in rows]

        if strategy == "epsilon_greedy":
            # 20% random exploration, 80% exploit top scores
            epsilon = 0.2
            if random.random() < epsilon or not any(n.score for n in nodes):
                return random.sample(nodes, min(k, len(nodes)))
            scored = sorted(
                [n for n in nodes if n.score is not None],
                key=lambda n: n.score,
                reverse=True,
            )
            return scored[:k]

        # score_weighted
        scored = [n for n in nodes if n.score is not None and n.score > 0]
        if not scored:
            return random.sample(nodes, min(k, len(nodes)))

        weights = [max(n.score, 0.01) for n in scored]
        total = sum(weights)
        weights = [w / total for w in weights]
        count = min(k, len(scored))
        # Weighted sample without replacement
        selected = []
        pool = list(zip(scored, weights))
        for _ in range(count):
            if not pool:
                break
            items, ws = zip(*pool)
            ws_list = list(ws)
            total_w = sum(ws_list)
            ws_list = [w / total_w for w in ws_list]
            idx = random.choices(range(len(items)), weights=ws_list, k=1)[0]
            selected.append(items[idx])
            pool.pop(idx)
        return selected


def get_best_node(session_id: str) -> Optional[EvolutionNode]:
    """Return the highest-scoring KEEP node for a session."""
    with get_db() as db:
        _ensure_table(db)
        row = db.execute(
            "SELECT id, session_id, round, parent_id, motivation, code, metrics, "
            "score, status, analysis, timestamp FROM evolution_nodes "
            "WHERE session_id = %s AND status = 'KEEP' "
            "ORDER BY score DESC LIMIT 1",
            [session_id],
        ).fetchone()
        if not row:
            return None
        return _row_to_node(row)


def get_session_summary(session_id: str) -> dict:
    """Return aggregate stats for an evolution session."""
    with get_db() as db:
        _ensure_table(db)
        rows = db.execute(
            "SELECT status, COUNT(*), MAX(score), MIN(score), AVG(score) "
            "FROM evolution_nodes WHERE session_id = %s GROUP BY status",
            [session_id],
        ).fetchall()

        summary = EvolutionSessionSummary(session_id=session_id)
        total = 0
        best_score = None
        for status, cnt, mx, mn, avg in rows:
            total += cnt
            if status == "KEEP":
                summary.kept_count = cnt
            elif status == "DISCARD":
                summary.discarded_count = cnt
            elif status in ("SYNTAX_ERROR", "RUNTIME_ERROR"):
                summary.error_count += cnt
            elif status == "TIMEOUT":
                summary.timeout_count = cnt
            if mx is not None:
                if best_score is None or mx > best_score:
                    best_score = mx

        summary.total_rounds = total
        summary.best_score = best_score

        best = get_best_node(session_id)
        if best:
            summary.best_node_id = best.id

        return summary.model_dump()


def get_all_nodes(
    session_id: Optional[str] = None, limit: int = 200
) -> list[EvolutionNode]:
    """Return all evolution nodes, optionally filtered by session."""
    with get_db() as db:
        _ensure_table(db)
        if session_id:
            rows = db.execute(
                "SELECT id, session_id, round, parent_id, motivation, code, metrics, "
                "score, status, analysis, timestamp FROM evolution_nodes "
                "WHERE session_id = %s ORDER BY round DESC LIMIT %s",
                [session_id, limit],
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT id, session_id, round, parent_id, motivation, code, metrics, "
                "score, status, analysis, timestamp FROM evolution_nodes "
                "ORDER BY timestamp DESC LIMIT %s",
                [limit],
            ).fetchall()
        return [_row_to_node(r) for r in rows]


def get_sessions() -> list[dict]:
    """Return a list of all evolution sessions with summary stats."""
    with get_db() as db:
        _ensure_table(db)
        rows = db.execute(
            "SELECT session_id, COUNT(*) as rounds, MAX(score) as best_score, "
            "MIN(timestamp) as started, MAX(timestamp) as last_updated, "
            "SUM(CASE WHEN status='KEEP' THEN 1 ELSE 0 END) as kept, "
            "SUM(CASE WHEN status='DISCARD' THEN 1 ELSE 0 END) as discarded, "
            "SUM(CASE WHEN status IN ('SYNTAX_ERROR','RUNTIME_ERROR','TIMEOUT') THEN 1 ELSE 0 END) as errors "
            "FROM evolution_nodes GROUP BY session_id ORDER BY MAX(timestamp) DESC"
        ).fetchall()
        return [
            {
                "session_id": r[0],
                "rounds": r[1],
                "best_score": r[2],
                "started": r[3],
                "last_updated": r[4],
                "kept": r[5],
                "discarded": r[6],
                "errors": r[7],
            }
            for r in rows
        ]

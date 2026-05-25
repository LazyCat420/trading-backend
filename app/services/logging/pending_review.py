"""
Pending Review Service — Consolidates news outlier approvals and pending evolution fixes.
"""

import uuid
import logging
from datetime import datetime, timezone
from fastapi import HTTPException
from app.db.connection import get_db

logger = logging.getLogger(__name__)


# ── NEWS OUTLIERS ──

def get_pending_outliers() -> list[dict]:
    """Fetch all news articles flagged as outliers pending human review."""
    with get_db() as db:
        rows = db.execute("""
            SELECT n.id, n.ticker, n.title, n.summary, n.publisher, n.quality_reason, c.consensus, n.published_at, n.url
            FROM news_articles n
            LEFT JOIN ticker_consensus c ON n.ticker = c.ticker
            WHERE n.quality_status = 'pending_review'
            ORDER BY n.published_at DESC
            LIMIT 50
        """).fetchall()
        
    return [
        {
            "id": r[0],
            "ticker": r[1],
            "title": r[2],
            "summary": r[3],
            "publisher": r[4],
            "reason": r[5],
            "consensus": r[6] or "No consensus generated yet.",
            "published_at": r[7].isoformat() if r[7] else None,
            "url": r[8]
        }
        for r in rows
    ]


def approve_outlier(article_id: str) -> dict:
    """Approve an outlier as valid breaking news."""
    with get_db() as db:
        db.execute("UPDATE news_articles SET quality_status = 'ok' WHERE id = %s", [article_id])
        
        row = db.execute("SELECT publisher FROM news_articles WHERE id = %s", [article_id]).fetchone()
        if row and row[0]:
            db.execute("""
                UPDATE source_trust 
                SET quality_wins = quality_wins + 2
                WHERE source_type = 'publisher' AND source_name = %s
            """, [row[0]])
            
    return {"status": "approved", "article_id": article_id}


def reject_outlier(article_id: str) -> dict:
    """Reject an outlier as fake news or spam."""
    with get_db() as db:
        row = db.execute("SELECT publisher FROM news_articles WHERE id = %s", [article_id]).fetchone()
        db.execute("UPDATE news_articles SET quality_status = 'rejected' WHERE id = %s", [article_id])
        
        if row and row[0]:
            db.execute("""
                UPDATE source_trust 
                SET total_items = total_items + 5,
                    win_rate = quality_wins::FLOAT / NULLIF((total_items + 5), 0)
                WHERE source_type = 'publisher' AND source_name = %s
            """, [row[0]])
            
    return {"status": "rejected", "article_id": article_id}


def add_outlier_rule(article_id: str, rule_content: str) -> dict:
    """Add a permanent rule based on this outlier, then reject it."""
    with get_db() as db:
        row = db.execute("SELECT ticker, publisher FROM news_articles WHERE id = %s", [article_id]).fetchone()
        if not row:
            raise HTTPException(404, "Article not found")
            
        ticker, publisher = row
        fb_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        db.execute("""
            INSERT INTO user_feedback (id, ticker, feedback_type, content, created_at, is_active)
            VALUES (%s, %s, 'constraint', %s, %s, TRUE)
        """, [fb_id, ticker, rule_content, now])
        
        db.execute("UPDATE news_articles SET quality_status = 'rejected' WHERE id = %s", [article_id])
        
        if publisher:
            db.execute("""
                UPDATE source_trust 
                SET total_items = total_items + 5,
                    win_rate = quality_wins::FLOAT / NULLIF((total_items + 5), 0)
                WHERE source_type = 'publisher' AND source_name = %s
            """, [publisher])

    return {"status": "rule_added", "article_id": article_id, "rule_id": fb_id}


# ── EVOLUTION FIXES ──

def get_pending_fixes(status: str = "all", limit: int = 50) -> list[dict]:
    """List pending evolution fixes from the Debate Council."""
    with get_db() as db:
        if status == "all":
            rows = db.execute(
                "SELECT id, cycle_id, target_type, target_name, proposed_fix, "
                "motivation, proposer_model, critic_concerns, judge_score, status, "
                "created_at, resolved_at "
                "FROM pending_evolution_fixes ORDER BY created_at DESC LIMIT %s",
                [limit],
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT id, cycle_id, target_type, target_name, proposed_fix, "
                "motivation, proposer_model, critic_concerns, judge_score, status, "
                "created_at, resolved_at "
                "FROM pending_evolution_fixes WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                [status, limit],
            ).fetchall()

    cols = [
        "id", "cycle_id", "target_type", "target_name", "proposed_fix",
        "motivation", "proposer_model", "critic_concerns", "judge_score", "status",
        "created_at", "resolved_at",
    ]
    fixes = []
    for row in rows:
        d = dict(zip(cols, row))
        fixes.append(d)
    return fixes


def approve_fix(fix_id: str) -> dict:
    """Mark a pending fix as approved (ready for deployment)."""
    with get_db() as db:
        row = db.execute(
            "SELECT id, status FROM pending_evolution_fixes WHERE id = %s", [fix_id]
        ).fetchone()
        if not row:
            return {"error": "Fix not found"}
        if row[1] != "pending":
            return {"error": f"Fix is already '{row[1]}', cannot approve"}

        db.execute(
            "UPDATE pending_evolution_fixes SET status = 'approved', resolved_at = CURRENT_TIMESTAMP WHERE id = %s",
            [fix_id],
        )
    return {"status": "approved", "id": fix_id}


def reject_fix(fix_id: str) -> dict:
    """Manually reject a pending fix."""
    with get_db() as db:
        db.execute(
            "UPDATE pending_evolution_fixes SET status = 'rejected', resolved_at = CURRENT_TIMESTAMP WHERE id = %s",
            [fix_id],
        )
    return {"status": "rejected", "id": fix_id}

from fastapi import APIRouter, Depends, HTTPException
from app.log_manager import log_manager
from typing import Optional
from app.db.connection import get_db

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

@router.get("/cycles")
def list_cycles():
    """List all available cycles."""
    return {"cycles": log_manager.list_all_cycles()}

@router.get("/logs")
def get_cycle_logs(cycle_id: str):
    """Fetch logs for a specific cycle."""
    logs = log_manager.get_cycle_log(cycle_id)
    if not logs:
        raise HTTPException(status_code=404, detail="Cycle logs not found")
    return {"cycle_id": cycle_id, "logs": logs}

@router.get("/errors")
def get_cycle_errors(cycle_id: str):
    """Fetch only error logs for a specific cycle."""
    errors = log_manager.get_cycle_errors(cycle_id)
    return {"cycle_id": cycle_id, "errors": errors}

@router.get("/stats")
def get_cycle_stats(cycle_id: str):
    """Fetch aggregated stats for a cycle."""
    stats = log_manager.get_cycle_stats(cycle_id)
    if stats.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Cycle stats not found")
    return stats

@router.get("/reports")
def get_cycle_reports(cycle_id: str):
    """Fetch all ticker reports for a given cycle."""
    reports = []
    with get_db() as db:
        rows = db.execute(
            "SELECT ticker, action, confidence, result_summary, is_summary, report_markdown, created_at "
            "FROM ticker_reports WHERE cycle_id = %s ORDER BY created_at DESC", 
            [cycle_id]
        ).fetchall()
        for row in rows:
            reports.append({
                "ticker": row[0],
                "action": row[1],
                "confidence": row[2],
                "result_summary": row[3],
                "is_summary": row[4],
                "report_markdown": row[5],
                "created_at": row[6].isoformat() if row[6] else None
            })
    return {"cycle_id": cycle_id, "reports": reports}

@router.get("/autoresearch")
def get_cycle_autoresearch(cycle_id: str):
    """Fetch autoresearch audit metrics and gaps for a given cycle."""
    reports = []
    with get_db() as db:
        rows = db.execute(
            "SELECT phase, status, error, data_gaps, decision_issues, llm_issues, "
            "data_quality_score, decision_quality_score, llm_performance_score, "
            "overall_score, reflection, created_at "
            "FROM autoresearch_reports WHERE cycle_id = %s ORDER BY created_at ASC", 
            [cycle_id]
        ).fetchall()
        for row in rows:
            reports.append({
                "phase": row[0],
                "status": row[1],
                "error": row[2],
                "data_gaps": row[3],
                "decision_issues": row[4],
                "llm_issues": row[5],
                "data_quality_score": row[6],
                "decision_quality_score": row[7],
                "llm_performance_score": row[8],
                "overall_score": row[9],
                "reflection": row[10],
                "created_at": row[11].isoformat() if row[11] else None
            })
    return {"cycle_id": cycle_id, "autoresearch": reports}

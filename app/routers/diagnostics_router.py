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

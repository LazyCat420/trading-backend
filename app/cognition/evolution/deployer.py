"""
Deployment logic for evolutionary fixes.
Extracts DB resolution and file I/O to a standalone module.
Now includes pre-deploy backup for rollback safety.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db.connection import get_db
from app.cognition.evolution.target_map import resolve_target

logger = logging.getLogger(__name__)

# Backup directory relative to project root
_BACKUP_DIR = Path(__file__).resolve().parents[3] / "backups" / "evolution"

# Probation period: number of hours after deploy to monitor
_PROBATION_HOURS = 6


def deploy_fix_to_disk(fix_id: str) -> dict:
    """
    Reads an approved fix from the database, resolves the target file,
    creates a backup, writes the fix, and sets probation monitoring.
    Returns a dict with 'status' and 'message', or 'error'.
    """
    with get_db() as db:
        row = db.execute(
            "SELECT id, target_type, target_name, proposed_fix, status "
            "FROM pending_evolution_fixes WHERE id = %s",
            [fix_id],
        ).fetchone()

        if not row:
            return {"error": f"Fix {fix_id} not found"}

        status = row[4]
        if status not in ("pending", "approved"):
            return {"error": f"Fix status is '{status}', cannot deploy"}

        target_type = row[1]
        target_name = row[2]
        proposed_fix = row[3]

        try:
            target_info = resolve_target(target_type, target_name)
            file_path = target_info.get("file_path")

            if not file_path or not target_info.get("exists"):
                return {
                    "error": f"Cannot resolve target file for {target_type}/{target_name}"
                }

            file_path_obj = Path(file_path)

            # ── Create backup before overwriting ──
            backup_path = _create_backup(file_path_obj, fix_id)
            if backup_path:
                logger.info("[EVO-DEPLOY] Backup created: %s", backup_path)

            # Write to disk
            file_path_obj.write_text(proposed_fix, encoding="utf-8")
            logger.info("[EVO-DEPLOY] Deployed fix to %s", file_path)

            # Compute approach hash for dead-end dedup
            approach_hash = hashlib.sha256(proposed_fix.encode()).hexdigest()[:16]

            # Set probation window
            probation_until = datetime.now(timezone.utc) + timedelta(
                hours=_PROBATION_HOURS
            )

            db.execute(
                "UPDATE pending_evolution_fixes "
                "SET status = 'deployed', resolved_at = CURRENT_TIMESTAMP, "
                "    backup_path = %s, probation_until = %s "
                "WHERE id = %s",
                [str(backup_path) if backup_path else None, probation_until, fix_id],
            )

            return {
                "status": "deployed",
                "id": fix_id,
                "file_path": file_path,
                "backup_path": str(backup_path) if backup_path else None,
                "approach_hash": approach_hash,
                "probation_until": probation_until.isoformat(),
                "message": f"Fix deployed to {target_info.get('relative_path')}",
            }
        except Exception as e:
            logger.error("[EVO-DEPLOY] Failed to deploy fix %s: %s", fix_id, e)
            return {"error": str(e)}


def rollback_fix(fix_id: str) -> dict:
    """Restore the backup file for a deployed fix and update status."""
    with get_db() as db:
        row = db.execute(
            "SELECT target_type, target_name, backup_path, proposed_fix "
            "FROM pending_evolution_fixes WHERE id = %s",
            [fix_id],
        ).fetchone()

        if not row:
            return {"error": f"Fix {fix_id} not found"}

        backup_path = row[2]
        if not backup_path or not Path(backup_path).exists():
            return {"error": f"No backup found for fix {fix_id}"}

        try:
            # Resolve the target file path
            target_info = resolve_target(row[0], row[1])
            file_path = target_info.get("file_path")

            if not file_path:
                return {"error": f"Cannot resolve target for {row[0]}/{row[1]}"}

            # Restore from backup
            backup_content = Path(backup_path).read_text(encoding="utf-8")
            Path(file_path).write_text(backup_content, encoding="utf-8")

            # Update status
            db.execute(
                "UPDATE pending_evolution_fixes "
                "SET status = 'rolled_back', resolved_at = CURRENT_TIMESTAMP "
                "WHERE id = %s",
                [fix_id],
            )

            logger.info(
                "[EVO-DEPLOY] Rolled back fix %s, restored from %s", fix_id, backup_path
            )
            return {"status": "rolled_back", "id": fix_id, "restored_from": backup_path}

        except Exception as e:
            logger.error("[EVO-DEPLOY] Rollback failed for %s: %s", fix_id, e)
            return {"error": str(e)}


def _create_backup(file_path: Path, fix_id: str) -> Path | None:
    """Create a backup of the target file before deploying a fix."""
    try:
        if not file_path.exists():
            return None

        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_name = f"{file_path.stem}_{fix_id[:8]}{file_path.suffix}.bak"
        backup_path = _BACKUP_DIR / backup_name

        backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")
        return backup_path

    except Exception as e:
        logger.warning("[EVO-DEPLOY] Backup creation failed: %s", e)
        return None

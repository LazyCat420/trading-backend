import logging
import json
import uuid
import asyncio
from app.tools.registry import registry, PermissionLevel
from app.db.connection import get_db

logger = logging.getLogger(__name__)


@registry.register(
    name="run_local_command",
    description="Run a local shell command (e.g. bash or python). REQUIRES HUMAN APPROVAL. Use this for sandboxed data gathering scripts or dynamic environment setup.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command string to execute in the system shell.",
            },
            "reason": {
                "type": "string",
                "description": "A clear explanation of WHY this command needs to be run, to show the human approver.",
            },
        },
        "required": ["command", "reason"],
    },
    tier=2,
    source="local",
    permission=PermissionLevel.DESTRUCTIVE,
    concurrency_safe=False,
    tags=["shell", "command", "execute", "system"],
)
async def run_local_command(
    command: str, reason: str, agent_name: str = "agent"
) -> str:
    """
    Submits a command for human approval.
    """
    logger.info(
        f"[SystemTools] Agent {agent_name} requested command execution: {command}"
    )

    cmd_id = f"cmd-{uuid.uuid4().hex[:8]}"

    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO pending_approvals (id, agent_name, command, reason) VALUES (%s, %s, %s, %s)",
                (cmd_id, agent_name, command, reason),
            )
    except Exception as e:
        logger.error(f"[SystemTools] Failed to insert pending approval: {e}")
        return json.dumps(
            {"status": "error", "message": "Failed to create approval request."}
        )

    # Poll for approval
    logger.info(f"[SystemTools] Waiting for human approval for command {cmd_id}...")

    max_wait = 120  # wait 2 minutes
    poll_interval = 2

    for _ in range(int(max_wait / poll_interval)):
        await asyncio.sleep(poll_interval)
        try:
            with get_db() as db:
                row = db.execute(
                    "SELECT status, stdout, stderr FROM pending_approvals WHERE id = %s",
                    (cmd_id,),
                ).fetchone()
            if not row:
                break

            status, stdout, stderr = row[0], row[1], row[2]

            if status == "approved":
                # Re-execute query to ensure we get the output after execution
                if stdout is None and stderr is None:
                    # Still executing by the backend worker
                    continue
                return json.dumps(
                    {"status": "success", "stdout": stdout, "stderr": stderr}
                )
            elif status == "rejected":
                return json.dumps(
                    {
                        "status": "rejected",
                        "message": "The human user rejected this command.",
                    }
                )

        except Exception as e:
            logger.error(f"[SystemTools] Polling error: {e}")
            break

    # Timeout
    with get_db() as db:
        db.execute(
            "UPDATE pending_approvals SET status = 'timeout' WHERE id = %s", (cmd_id,)
        )
    return json.dumps({"status": "error", "message": "Command approval timed out."})

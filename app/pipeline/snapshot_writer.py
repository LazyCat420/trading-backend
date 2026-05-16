import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path("memory/cycle_tracking/snapshots")


def write_snapshot(cycle_id: str, ticker: str, data: dict) -> str | None:
    """
    Archive prompts, rationale, and context for a specific cycle/ticker.
    Returns the snapshot file path or None if failed.
    """
    try:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{cycle_id}_{ticker}_{timestamp}.json"
        file_path = SNAPSHOT_DIR / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved snapshot for %s to %s", ticker, filename)
        return filename
    except Exception as e:
        logger.error("Failed to save snapshot for %s: %s", ticker, e)
        return None

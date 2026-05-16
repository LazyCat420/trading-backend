import csv
from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

TSV_PATH = Path("memory/cycle_tracking/cycle_results.tsv")


def append_cycle_result(
    cycle_id: str,
    ticker: str,
    action: str,
    confidence: int,
    tokens_used: int = 0,
    elapsed_sec: float = 0.0,
    outcome: str = "",
    realized_pnl: float = 0.0,
    status: str = "ok",
    experiment_desc: str = "",
    snapshot_ref: str = "",
):
    """Append a single ticker's cycle result to the tracking TSV."""
    try:
        TSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = TSV_PATH.exists()

        timestamp = datetime.now(timezone.utc).isoformat()

        with open(TSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            if not file_exists:
                writer.writerow(
                    [
                        "cycle_id",
                        "timestamp",
                        "ticker",
                        "action",
                        "confidence",
                        "tokens_used",
                        "elapsed_sec",
                        "outcome",
                        "realized_pnl",
                        "status",
                        "experiment_desc",
                        "snapshot_ref",
                    ]
                )

            writer.writerow(
                [
                    cycle_id,
                    timestamp,
                    ticker,
                    action,
                    confidence,
                    tokens_used,
                    f"{elapsed_sec:.2f}",
                    outcome,
                    f"{realized_pnl:.4f}",
                    status,
                    experiment_desc,
                    snapshot_ref,
                ]
            )
    except Exception as e:
        logger.warning("Failed to append cycle result to TSV: %s", e)

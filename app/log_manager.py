import json
from datetime import datetime, timezone
from pathlib import Path


class LogManager:
    """Manages appending logs for V2 and A/B benchmark runs."""

    BASE_DIR = Path("logs")
    CYCLE_DIR = BASE_DIR / "cycles"
    AB_DIR = CYCLE_DIR / "ab_results"

    def __init__(self):
        try:
            self.CYCLE_DIR.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = self.CYCLE_DIR / ".write_test"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError):
            # Fallback to local logs directory if default logs dir is not writable
            self.BASE_DIR = Path("logs_local")
            self.CYCLE_DIR = self.BASE_DIR / "cycles"
            self.AB_DIR = self.CYCLE_DIR / "ab_results"
            self.CYCLE_DIR.mkdir(parents=True, exist_ok=True)

        self.AB_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_jsonl(path: Path, data: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def log_v2_cycle(self, cycle_id: str, step_name: str, payload: dict):
        """Append a step result to the v2 cycle log in logs/v2/cycle_{cycle_id}.jsonl"""
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "step": step_name,
            "payload": payload,
        }
        file_path = self.CYCLE_DIR / f"{cycle_id}.jsonl"
        self._write_jsonl(file_path, log_entry)

    def log_ab_result(
        self,
        cycle_id: str,
        ab_chosen: str,
        v1_result: dict,
        v2_result: dict,
        context: dict = None,
    ):
        """Append an A/B benchmark result to logs/ab_results/ab_{cycle_id}.jsonl"""
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "ab_chosen": ab_chosen,
            "v1_result": v1_result,
            "v2_result": v2_result,
            "context": context or {},
        }
        file_path = self.AB_DIR / f"ab_{cycle_id}.jsonl"
        self._write_jsonl(file_path, log_entry)


log_manager = LogManager()

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

    def detect_abandoned_cycles(self, max_age_hours: int = 24) -> list[dict]:
        """Scan v2 cycle logs and identify abandoned cycles.

        An abandoned cycle has v2_start but neither v2_pipeline_complete
        nor v2_error. Returns a list of dicts with cycle_id, ticker,
        start_time, and last_step for monitoring.
        """
        import glob
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        abandoned = []

        for path in sorted(self.CYCLE_DIR.glob("*.jsonl")):
            if path.name.startswith("ab_"):
                continue  # Skip A/B result files

            steps = set()
            first_entry = None
            last_entry = None
            ticker = "?"

            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        steps.add(entry.get("step", ""))
                        if first_entry is None:
                            first_entry = entry
                        last_entry = entry
                        if entry.get("payload", {}).get("ticker"):
                            ticker = entry["payload"]["ticker"]
            except Exception:
                continue

            if not first_entry:
                continue

            # Skip if cycle completed or has error logged
            if "v2_pipeline_complete" in steps or "v2_error" in steps:
                continue

            # Only flag if cycle is old enough (not still running)
            ts_str = first_entry.get("timestamp", "")
            try:
                cycle_start = datetime.fromisoformat(ts_str)
                if cycle_start > cutoff:
                    continue  # Too recent, might still be running
            except Exception:
                continue

            abandoned.append({
                "cycle_id": first_entry.get("cycle_id", path.stem),
                "ticker": ticker,
                "start_time": ts_str,
                "last_step": last_entry.get("step", "?") if last_entry else "?",
                "file": str(path),
            })

        return abandoned


log_manager = LogManager()

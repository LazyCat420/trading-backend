"""
Worker configuration model.

Loaded from worker_config.json or environment variables.
Each box gets its own config with settings from the concurrency benchmark.
"""

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkerConfig(BaseModel):
    """Configuration for a single distributed worker instance.

    Cross-platform: runs on Windows, WSL2, and bare Linux.
    Redis must be reachable at redis_url — install via:
      Windows:  winget install Redis.Redis  OR  Docker
      Linux:    apt install redis-server
      WSL2:     same as Linux (service redis-server start)
    """

    worker_id: str = Field(default="worker-01")
    redis_url: str = Field(default="redis://localhost:6379")
    max_tier: int = Field(default=1, ge=0, le=2)
    max_parallel_requests: int = Field(default=8, ge=1, le=256)
    vllm_endpoint: str = Field(default="http://localhost:8000")
    hermes_endpoint: str = Field(default="")
    heartbeat_interval_s: int = Field(default=30, ge=5)
    drain_timeout_s: int = Field(default=60, ge=10)


def load_config(config_path: str | None = None) -> WorkerConfig:
    """Load worker config from JSON file, env vars, or defaults."""
    path = config_path or os.environ.get("WORKER_CONFIG_PATH", "")
    if not path:
        cwd_path = Path.cwd() / "worker_config.json"
        if cwd_path.exists():
            path = str(cwd_path)

    config_data: dict = {}
    if path and Path(path).exists():
        try:
            with open(path, encoding="utf-8") as f:
                config_data = json.load(f)
            logger.info("Loaded worker config from %s", path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    env_map = {
        "WORKER_ID": "worker_id",
        "REDIS_URL": "redis_url",
        "WORKER_MAX_TIER": "max_tier",
        "WORKER_MAX_PARALLEL": "max_parallel_requests",
        "WORKER_VLLM_ENDPOINT": "vllm_endpoint",
        "HERMES_ENDPOINT": "hermes_endpoint",
    }
    for env_key, field_name in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            config_data[field_name] = val

    return WorkerConfig(**config_data)

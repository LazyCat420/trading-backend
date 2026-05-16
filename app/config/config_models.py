import json
import os
import logging

logger = logging.getLogger(__name__)

LIMITS_FILE = os.path.join("data", "model_limits.json")

# Static Data - Not overridable via environment variables (sensible defaults based on extensive DGX benchmarks)
DEFAULT_LIMITS = {
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": 1,
    "google/gemma-4-26B-A4B-it": 128,
    "Kbenkhaled/Qwen3.5-35B-A3B-quantized.w4a16": 24,
}


def _ensure_limits_file():
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
    if not os.path.exists(LIMITS_FILE):
        with open(LIMITS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_LIMITS, f, indent=2)


def get_model_limits() -> dict[str, int]:
    """Retrieve the dictionary of explicitly set model concurrency limits."""
    _ensure_limits_file()
    try:
        with open(LIMITS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Merge missing defaults
            changed = False
            for k, v in DEFAULT_LIMITS.items():
                if k not in data:
                    data[k] = v
                    changed = True

            if changed:
                with open(LIMITS_FILE, "w", encoding="utf-8") as fw:
                    json.dump(data, fw, indent=2)

            return data
    except Exception as e:
        logger.error(f"Failed to read model limits: {e}")
        return DEFAULT_LIMITS.copy()


def get_model_limit(model_id: str, default: int = 8) -> int:
    """Get the max concurrent limit for a single model."""
    limits = get_model_limits()
    return limits.get(model_id, default)


def set_model_limit(model_id: str, limit: int):
    """Write a new limit bound to this specific model."""
    limits = get_model_limits()
    limits[model_id] = limit
    try:
        with open(LIMITS_FILE, "w", encoding="utf-8") as f:
            json.dump(limits, f, indent=2)
        logger.info(f"Saved custom limit {limit} for model {model_id}")
    except Exception as e:
        logger.error(f"Failed to save model limit: {e}")

"""
Lightweight TTL response cache for FastAPI endpoints.

Caches function results in-memory with configurable TTL.
Supports cache invalidation by group key for write endpoints.

Usage:
    from app.cache import timed_cache, invalidate_cache

    @timed_cache(ttl_seconds=300, group="sectors")
    async def get_heatmap():
        ...

    # Invalidate when data changes:
    invalidate_cache("sectors")
"""

import time
import logging
import hashlib
import json
from functools import wraps
from typing import Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ── In-memory cache store (Bounded LRU) ──
# Key: (group, func_name, args_hash) → (timestamp, result)
_cache: OrderedDict[tuple[str, str, str], tuple[float, Any]] = OrderedDict()
MAX_CACHE_SIZE = 500

# Track cache stats for debugging
_stats = {"hits": 0, "misses": 0, "invalidations": 0}


def _make_args_hash(*args: Any, **kwargs: Any) -> str:
    """Create a deterministic hash from function arguments."""
    key_data = json.dumps(
        {
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        },
        sort_keys=True,
    )
    return hashlib.md5(key_data.encode()).hexdigest()[:12]


def timed_cache(ttl_seconds: int = 300, group: str = "default"):
    """Decorator that caches async/sync function results with a TTL.

    Args:
        ttl_seconds: How long to cache results (default 5 min).
        group: Cache group name for bulk invalidation.
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            args_hash = _make_args_hash(*args, **kwargs)
            cache_key = (group, func.__name__, args_hash)
            now = time.monotonic()

            # Check cache
            cached = _cache.get(cache_key)
            if cached is not None:
                ts, result = cached
                if (now - ts) < ttl_seconds:
                    _stats["hits"] += 1
                    _cache.move_to_end(cache_key)
                    return result
                # Clean up expired on access
                del _cache[cache_key]

            # Cache miss — call function
            _stats["misses"] += 1
            result = await func(*args, **kwargs)

            if len(_cache) >= MAX_CACHE_SIZE:
                _cache.popitem(last=False)

            _cache[cache_key] = (now, result)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            args_hash = _make_args_hash(*args, **kwargs)
            cache_key = (group, func.__name__, args_hash)
            now = time.monotonic()

            cached = _cache.get(cache_key)
            if cached is not None:
                ts, result = cached
                if (now - ts) < ttl_seconds:
                    _stats["hits"] += 1
                    _cache.move_to_end(cache_key)
                    return result
                # Clean up expired on access
                del _cache[cache_key]

            _stats["misses"] += 1
            result = func(*args, **kwargs)

            if len(_cache) >= MAX_CACHE_SIZE:
                _cache.popitem(last=False)

            _cache[cache_key] = (now, result)
            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def invalidate_cache(group: str) -> int:
    """Clear all cached entries for a given group.

    Returns number of entries cleared.
    """
    keys_to_remove = [k for k in _cache if k[0] == group]
    for k in keys_to_remove:
        del _cache[k]
    count = len(keys_to_remove)
    if count > 0:
        _stats["invalidations"] += 1
        logger.info("[cache] Invalidated %d entries for group '%s'", count, group)
    return count


def get_cache_stats() -> dict:
    """Return cache hit/miss statistics."""
    total = _stats["hits"] + _stats["misses"]
    return {
        **_stats,
        "hit_rate": round(_stats["hits"] / total * 100, 1) if total > 0 else 0,
        "entries": len(_cache),
    }

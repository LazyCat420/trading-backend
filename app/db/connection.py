"""
PostgreSQL Connection — Thread-safe connection pool.

Uses psycopg3 (sync) with a connection pool. Schema auto-initialized on first use.

Public API:
    get_db()         → returns a PooledCursor that behaves like a standard DB-API cursor
    get_write_lock() → asyncio.Lock (kept for safety, Postgres handles concurrency natively)
    close_db()       → shutdown the pool
"""

import asyncio
from contextlib import contextmanager
import json as _json
import logging
import os
import threading
import time
import traceback
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)


def safe_jsonb(val):
    """Normalize a JSONB value — always returns a dict/list regardless of
    whether _stringify_jsonb turned it into a string.

    Use this when reading JSONB columns that may have been stringified
    by the PooledCursor compatibility layer.
    """
    if isinstance(val, str):
        try:
            return _json.loads(val)
        except Exception:
            return val
    return val  # already a dict/list

_lock = threading.Lock()
_pool: ConnectionPool | None = None
_async_write_lock: asyncio.Lock | None = None


def get_write_lock() -> asyncio.Lock:
    """Return the shared asyncio.Lock for serializing write operations.

    PostgreSQL handles concurrent writes natively via MVCC, but we keep
    this lock for backward compatibility with code that already uses it.

    Usage in async code:
        async with get_write_lock():
            with get_db() as db:
                db.execute("INSERT ...")
    """
    global _async_write_lock
    if _async_write_lock is None:
        _async_write_lock = asyncio.Lock()
    return _async_write_lock


# ── Postgres strict mode (No placeholder translation) ─────────────────
# The codebase must explicitly use %s for placeholders, not ?.


class PooledCursor:
    """Wrapper around a psycopg connection that mimics standard DB-API cursor behavior.

    Key compatibility features:
    - execute(sql, params) translates %s → %s placeholders
    - fetchone() / fetchall() work identically
    - .description returns column metadata
    - Auto-commits after each execute (for legacy compatibility)
    """

    def __init__(self, conn: psycopg.Connection, pool=None):
        self._conn = conn
        self._cursor = conn.cursor()
        self._pool_ref = pool

        self.description: Any = None
        self._closed = False
        self._created_at = time.monotonic()
        # Capture creation callsite for leak detection
        self._origin = "".join(traceback.format_stack(limit=4)[:-1])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if getattr(self, "_closed", True):
            return
        # Connection was never closed — this is a LEAK
        age = time.monotonic() - getattr(self, "_created_at", 0)
        origin = getattr(self, "_origin", "unknown")
        logger.warning(
            "[DB LEAK] PooledCursor garbage-collected after %.1fs without close()!\n"
            "  Created at:\n%s",
            age,
            origin,
        )
        try:
            self.close()
        except Exception:
            pass

    def execute(self, sql: str, params=None) -> "PooledCursor":
        """Execute SQL with automatic placeholder translation."""

        # Convert list params to tuple (psycopg requires tuples)
        if isinstance(params, list):
            params = tuple(params)

        try:
            self._cursor.execute(sql, params)
            self.description = self._cursor.description
        except Exception:
            # Rollback on error to avoid "in error state" blocking
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise
        else:
            # Auto-commit for write operations
            if not self._conn.autocommit:
                try:
                    self._conn.commit()
                except Exception:
                    pass
        return self

    def executemany(self, sql: str, params_seq) -> "PooledCursor":
        """Execute SQL for a sequence of parameters."""

        # Ensure parameters are sequences (tuples)
        cleaned_seq = [tuple(p) if isinstance(p, list) else p for p in params_seq]

        try:
            self._cursor.executemany(sql, cleaned_seq)
            self.description = self._cursor.description
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise
        else:
            if not self._conn.autocommit:
                try:
                    self._conn.commit()
                except Exception:
                    pass
        return self

    @contextmanager
    def transaction(self):
        """Standard psycopg transaction management."""
        with self._conn.transaction():
            yield

    def commit(self):
        """No-op: PooledCursor auto-commits after every execute().

        Kept for backward compatibility with legacy code
        which called db.commit() explicitly.
        """
        pass

    def rollback(self):
        """Rollback the current transaction."""
        try:
            self._conn.rollback()
        except Exception:
            pass

    def _stringify_jsonb(self, row):
        """Convert dicts/lists returned by psycopg back to JSON strings for legacy compatibility."""
        if row is None:
            return None
        changed = False
        new_row = []
        for item in row:
            if isinstance(item, (dict, list)):
                import orjson

                new_row.append(orjson.dumps(item).decode("utf-8"))
                changed = True
            else:
                new_row.append(item)
        return tuple(new_row) if changed else row

    def fetchone(self):
        return self._stringify_jsonb(self._cursor.fetchone())

    def fetchall(self):
        rows = self._cursor.fetchall()
        return [self._stringify_jsonb(r) for r in rows]

    def close(self):
        """Return the connection to the pool."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            self._cursor.close()
        except Exception:
            pass
        try:
            if self._pool_ref is not None:
                self._pool_ref.putconn(self._conn)
        except Exception:
            pass


def _ensure_pool() -> ConnectionPool:
    """Ensure the connection pool exists. Thread-safe init only."""
    global _pool
    if _pool is not None:
        return _pool
    with _lock:
        if _pool is None:
            if os.getenv("TRADING_BOT_TEST_DB") == "1":
                db_url = "postgresql://trader:trading_bot_pass@10.0.0.16:5433/trading_bot_test"
            else:
                db_url = settings.DATABASE_URL
            logger.info(
                f"[DB] Connecting to PostgreSQL: {db_url.split('@')[-1] if '@' in db_url else db_url}"
            )

            # Initialize schema before pool creation so pgvector extension is ready
            _init_schema(db_url)

            def _configure_connection(conn):
                from pgvector.psycopg import register_vector

                try:
                    register_vector(conn)
                except Exception as e:
                    logger.warning(f"[DB] Failed to register pgvector: {e}")

            import sys
            is_tool = (
                os.getenv("IS_TOOL_PROCESS") == "true"
                or any("execute_tool.py" in arg for arg in sys.argv)
            )
            min_sz = 1 if is_tool else 10
            max_sz = 2 if is_tool else 50
            if is_tool:
                logger.info(f"[DB] Tool process detected. Scaling down ConnectionPool to size min={min_sz}, max={max_sz}")

            _pool = ConnectionPool(
                conninfo=db_url,
                min_size=min_sz,
                max_size=max_sz,
                kwargs={"autocommit": True},
                configure=_configure_connection,
            )
            # Wait for the pool to be ready
            _pool.wait()

            # Seed bot and run migrations using the pool
            _seed_and_migrate()
            logger.info("[DB] PostgreSQL connection pool initialized")
        return _pool


@contextmanager
def get_db():
    """Get a cursor-like object from the PostgreSQL connection pool.

    Returns a context manager yielding a PooledCursor that is API-compatible
    with standard DB-API cursors. The underlying connection is taken from the pool
    and will be returned when the block exits.
    """
    pool = _ensure_pool()
    try:
        conn = pool.getconn(timeout=5.0)
    except Exception as e:
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            import gc

            logger.warning(
                "[DB] Pool timeout! Forcing GC collection to reclaim leaked connections..."
            )
            gc.collect()
            conn = pool.getconn(timeout=5.0)
        else:
            raise

    cursor = PooledCursor(conn, pool=pool)
    try:
        yield cursor
    finally:
        cursor.close()


def _init_schema(db_url: str):
    """Run schema_pg.sql to create all tables."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema_pg.sql")
    if not os.path.exists(schema_path):
        logger.warning(f"[DB] Schema file not found: {schema_path}")
        return

    with open(schema_path, encoding="utf-8") as f:
        sql = f.read()

    try:
        with psycopg.connect(db_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                # Execute the full schema as one block
                cur.execute(sql)
            logger.info("[DB] Schema initialized from schema_pg.sql")
    except Exception as e:
        logger.error(f"[DB] Schema init error: {e}")
        raise


def _seed_and_migrate():
    """Seed default bot and run migrations after pool is initialized."""
    # ── Seed default bot if bots table is empty ──
    try:
        with get_db() as db:
            db.execute("SELECT COUNT(*) FROM bots")
            row = db.fetchone()
            if row and row[0] == 0:
                from app.config import settings as _s

                db.execute(
                    "INSERT INTO bots (bot_id, display_name, model_name, status, "
                    "cash_balance, starting_cash, total_pnl, win_rate, total_trades, "
                    "is_active, created_at) "
                    "VALUES (%s, 'Lazy Trader V4', %s, 'idle', %s, %s, 0.0, 0.0, 0, "
                    "TRUE, CURRENT_TIMESTAMP)",
                    (_s.BOT_ID, _s.ACTIVE_MODEL, _s.STARTING_CASH, _s.STARTING_CASH),
                )
                logger.info(f"[DB] Seeded default bot: {_s.BOT_ID}")
    except Exception as e:
        logger.info(f"[DB] Bot seed skipped: {e}")

    # ── Auto-migrations for existing databases ──
    try:
        from app.db.migrations import run_migrations
        # Use a raw connection from the pool for migrations as it might require it
        conn = _pool.getconn()
        try:
            run_migrations(conn)
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            logger.warning(f"[DB] Migration warning: {e}")
        finally:
            _pool.putconn(conn)
    except Exception as e:
        logger.warning(f"[DB] Migration setup warning: {e}")


def close_db():
    """Close the connection pool (for cleanup)."""
    global _pool
    with _lock:
        if _pool is not None:
            try:
                _pool.close()
                logger.info("[DB] PostgreSQL connection pool closed")
            except Exception as e:
                logger.warning(f"[DB] Pool close error: {e}")
            _pool = None

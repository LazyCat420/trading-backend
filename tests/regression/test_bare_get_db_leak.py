"""
Regression test: Bare get_db() connection leak detection.

Original bug: 303+ bare `db = get_db()` calls across the codebase would leak
connections because PooledCursor.__del__ is unreliable in async contexts. The
connection pool (max_size=50) would exhaust under sustained load, causing the
entire backend to stall.

This test scans the codebase for the bare pattern and fails if any new
unprotected calls are introduced.
"""

import re
from pathlib import Path

import pytest


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pytest.ini."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pytest.ini").exists():
            return p
        p = p.parent
    # Fallback: assume tests/ is one level below project root
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()


# Pattern: `db = get_db()` NOT inside a `with` statement
BARE_GET_DB_RE = re.compile(r"^\s+db\s*=\s*get_db\(\)", re.MULTILINE)
WITH_GET_DB_RE = re.compile(r"^\s+with\s+get_db\(\)\s+as\s+db:", re.MULTILINE)


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_number, line) for bare get_db() calls."""
    text = path.read_text(encoding="utf-8")
    violations = []
    for i, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if stripped == "db = get_db()" or stripped == "db  = get_db()":
            violations.append((i, line.strip()))
    return violations


# These files are KNOWN to still have bare patterns and are being migrated.
# Remove entries from this list as they are fixed.
ALLOWLIST = set()


@pytest.mark.unit
def test_no_new_bare_get_db_calls():
    """Ensure no NEW bare get_db() calls are introduced outside the allowlist.

    As files are migrated to `with get_db() as db:`, remove them from ALLOWLIST.
    The test will catch any regressions.
    """
    app_dir = PROJECT_ROOT / "app"
    violations = []

    for py_file in sorted(app_dir.rglob("*.py")):
        rel_path = str(py_file.relative_to(app_dir.parent)).replace("\\", "/")
        if rel_path in ALLOWLIST:
            continue

        file_violations = _scan_file(py_file)
        if file_violations:
            for line_no, line_text in file_violations:
                violations.append(f"  {rel_path}:{line_no}: {line_text}")

    if violations:
        msg = (
            f"Found {len(violations)} NEW bare get_db() calls outside the allowlist!\n"
            "Use `with get_db() as db:` instead.\n" + "\n".join(violations)
        )
        pytest.fail(msg)


@pytest.mark.unit
def test_main_py_has_no_bare_get_db():
    """Critical: main.py/cycle_main.py lifespan must never have bare get_db() calls.

    The lifespan function previously held a bare connection for the entire
    server lifetime, consuming 1 of 50 pool slots permanently.
    """
    main_path = PROJECT_ROOT / "app" / "main.py"
    if not main_path.exists():
        main_path = PROJECT_ROOT / "cycle_main.py"
    
    if not main_path.exists():
        pytest.skip("Neither app/main.py nor cycle_main.py found")

    violations = _scan_file(main_path)
    if violations:
        lines = [f"  L{ln}: {txt}" for ln, txt in violations]
        pytest.fail(
            f"{main_path.name} has bare get_db() calls (CRITICAL leak):\n" + "\n".join(lines)
        )

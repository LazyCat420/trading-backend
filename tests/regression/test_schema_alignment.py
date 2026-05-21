"""
Test: Schema Alignment Verification.

Scans all Python source files for SQL queries and cross-references
table/column names against the actual schema_pg.sql definitions.
Prevents regressions like the 'technical_indicators' vs 'technicals' bug.
"""

import re
import os
import pytest

# ── Paths ──
APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "app")
SCHEMA_PATH = os.path.join(APP_DIR, "db", "schema_pg.sql")


def _parse_schema_tables() -> dict[str, set[str]]:
    """Parse schema_pg.sql and return {table_name: {column_names}}."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        sql = f.read()

    tables = {}
    # Match CREATE TABLE blocks
    pattern = re.compile(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*\((.*?)\);",
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(sql):
        table_name = match.group(1).lower()
        body = match.group(2)
        columns = set()
        for line in body.split("\n"):
            line = line.strip().rstrip(",")
            if not line or line.upper().startswith(("PRIMARY", "UNIQUE", "CONSTRAINT", "CHECK", "FOREIGN", "--")):
                continue
            # Extract column name (first word, handling quoted identifiers)
            col_match = re.match(r'"?(\w+)"?\s+', line)
            if col_match:
                columns.add(col_match.group(1).lower())
        tables[table_name] = columns
    return tables


def _find_sql_table_references() -> list[dict]:
    """Scan all .py files for SQL FROM/JOIN/INTO clauses inside string literals."""
    refs = []
    for root, _, files in os.walk(APP_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            # Extract string literals (single, double, triple-quoted)
            # This ensures we only scan SQL queries, not Python imports
            string_pattern = re.compile(
                r'"""(.*?)"""|\'\'\'(.*?)\'\'\'|"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'' ,
                re.DOTALL,
            )
            for sm in string_pattern.finditer(content):
                string_content = sm.group(1) or sm.group(2) or sm.group(3) or sm.group(4) or ""
                # Only consider strings that look like SQL
                if not re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE)\b", string_content, re.IGNORECASE):
                    continue
                for m in re.finditer(
                    r"\b(?:FROM|JOIN|INTO|UPDATE)\s+(\w+)",
                    string_content,
                    re.IGNORECASE,
                ):
                    table = m.group(1).lower()
                    if table in ("select", "where", "set", "values", "as", "on", "and", "or", "not", "null", "true", "false"):
                        continue
                    refs.append({
                        "file": os.path.relpath(filepath, APP_DIR),
                        "table": table,
                        "match": m.group(0),
                    })
    return refs


class TestSchemaAlignment:
    """Verify all SQL table references in Python code match the schema."""

    @pytest.fixture(scope="class")
    def schema_tables(self):
        return _parse_schema_tables()

    @pytest.fixture(scope="class")
    def sql_refs(self):
        return _find_sql_table_references()

    def test_schema_file_exists(self):
        """schema_pg.sql must exist."""
        assert os.path.isfile(SCHEMA_PATH), f"Schema file not found: {SCHEMA_PATH}"

    def test_schema_has_tables(self, schema_tables):
        """Schema must define at least 10 tables."""
        assert len(schema_tables) >= 10, f"Only found {len(schema_tables)} tables in schema"

    def test_no_technical_indicators_reference(self):
        """REGRESSION: 'technical_indicators' table does not exist — must use 'technicals'."""
        refs = _find_sql_table_references()
        bad_refs = [r for r in refs if r["table"] == "technical_indicators"]
        assert not bad_refs, (
            f"Found {len(bad_refs)} references to non-existent 'technical_indicators' table "
            f"(should be 'technicals'):\n"
            + "\n".join(f"  {r['file']}: {r['match']}" for r in bad_refs)
        )

    def test_tool_sql_references_valid(self, schema_tables):
        """SQL table references in app/tools/ must exist in schema_pg.sql.

        This is the most critical directory — tool SQL errors cause silent
        agent failures (like the technical_indicators bug).
        """
        tools_dir = os.path.join(APP_DIR, "tools")
        ALIASES = {"t", "p", "e", "s", "r", "a", "b", "c", "d", "f", "m", "w"}

        missing = []
        for fname in os.listdir(tools_dir):
            if not fname.endswith(".py"):
                continue
            filepath = os.path.join(tools_dir, fname)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()

            # Find SQL strings containing SELECT/INSERT/UPDATE/DELETE
            for sm in re.finditer(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', content, re.DOTALL):
                sql_str = sm.group(1) or sm.group(2) or ""
                if not re.search(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b", sql_str, re.IGNORECASE):
                    continue
                for m in re.finditer(r"\bFROM\s+(\w+)", sql_str, re.IGNORECASE):
                    table = m.group(1).lower()
                    if table in ALIASES or len(table) <= 1:
                        continue
                    if table not in schema_tables:
                        missing.append(f"{fname}: FROM {table}")

        assert not missing, (
            f"Tool files reference tables NOT in schema_pg.sql:\n"
            + "\n".join(f"  {m}" for m in missing)
        )


class TestColumnAlignment:
    """Spot-check critical column references match the schema."""

    @pytest.fixture(scope="class")
    def schema_tables(self):
        return _parse_schema_tables()

    def test_technicals_has_expected_columns(self, schema_tables):
        """The 'technicals' table must have the columns used by quant_tools."""
        assert "technicals" in schema_tables
        expected = {"rsi_14", "macd", "macd_signal", "sma_20", "sma_50", "sma_200", "ticker", "date"}
        actual = schema_tables["technicals"]
        missing = expected - actual
        assert not missing, f"'technicals' table missing columns: {missing}"

    def test_fundamentals_has_expected_columns(self, schema_tables):
        """The 'fundamentals' table must have the columns used by check_hallucination."""
        assert "fundamentals" in schema_tables
        expected = {"pe_ratio", "market_cap", "forward_pe", "snapshot_date", "ticker"}
        actual = schema_tables["fundamentals"]
        missing = expected - actual
        assert not missing, f"'fundamentals' table missing columns: {missing}"

    def test_fundamentals_no_dividend_yield(self, schema_tables):
        """REGRESSION: 'fundamentals' does NOT have 'dividend_yield' — code should not reference it."""
        assert "fundamentals" in schema_tables
        assert "dividend_yield" not in schema_tables["fundamentals"], (
            "'dividend_yield' column found in fundamentals schema — "
            "if this was added intentionally, update the check_hallucination tool to use it"
        )

    def test_fundamentals_no_updated_at(self, schema_tables):
        """REGRESSION: 'fundamentals' uses 'snapshot_date', NOT 'updated_at'."""
        assert "fundamentals" in schema_tables
        assert "updated_at" not in schema_tables["fundamentals"], (
            "'updated_at' found in fundamentals — queries should use 'snapshot_date'"
        )

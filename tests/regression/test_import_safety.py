"""
Test: Import Safety Verification.

Uses AST analysis to detect variables that are used but never imported
in critical pipeline modules. Prevents regressions like the missing
`datetime` import in data_perticker_collection.py.
"""

import ast
import os
import builtins
import pytest

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "app")

# Standard library modules that are commonly used without import (builtins)
BUILTIN_NAMES = set(dir(builtins)) | {
    # Common module-level names that aren't technically builtins but are always available
    "__name__", "__file__", "__doc__", "__all__", "__builtins__",
    "__spec__", "__loader__", "__package__", "__cached__",
}


def _get_imports(tree: ast.AST) -> set[str]:
    """Extract all imported names from an AST."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name)
    return names


def _get_top_level_names(tree: ast.AST) -> set[str]:
    """Extract all names defined at module or function level (assignments, defs, etc.)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            # Add argument names
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                names.add(arg.arg)
            if node.args.vararg:
                names.add(node.args.vararg.arg)
            if node.args.kwarg:
                names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Global):
            names.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            names.update(node.names)
        # Comprehension variables
        elif isinstance(node, ast.comprehension):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        # Exception handler alias
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        # For loop target
        elif isinstance(node, ast.For) or isinstance(node, ast.AsyncFor):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        names.add(elt.id)
        # With statement
        elif isinstance(node, ast.With) or isinstance(node, ast.AsyncWith):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    names.add(item.optional_vars.id)
    return names


def _find_unresolved_module_attrs(filepath: str) -> list[str]:
    """Find `module.attr` references where `module` is not imported."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = _get_imports(tree)
    defined = _get_top_level_names(tree)
    all_names = imports | defined | BUILTIN_NAMES

    unresolved = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # Check for `module.something` patterns where module is a bare Name
            if isinstance(node.value, ast.Name):
                name = node.value.id
                if name not in all_names:
                    unresolved.append(f"{name}.{node.attr}")
    return unresolved


class TestImportSafety:
    """Verify no missing imports in critical pipeline modules."""

    def test_data_perticker_collection_no_missing_imports(self):
        """REGRESSION: data_perticker_collection.py must import datetime."""
        filepath = os.path.join(
            APP_DIR, "pipeline", "data", "data_perticker_collection.py"
        )
        assert os.path.isfile(filepath), f"File not found: {filepath}"

        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        imports = _get_imports(tree)

        assert "datetime" in imports, (
            "CRITICAL: 'datetime' is not imported in data_perticker_collection.py. "
            "This causes YouTube collection to silently fail for ALL tickers."
        )

    def test_quant_tools_no_unresolved_names(self):
        """quant_tools.py must not reference undefined modules."""
        filepath = os.path.join(APP_DIR, "tools", "quant_tools.py")
        unresolved = _find_unresolved_module_attrs(filepath)
        assert not unresolved, (
            f"quant_tools.py has unresolved module references: {unresolved}"
        )

    def test_pipeline_tools_no_unresolved_names(self):
        """pipeline_tools.py must not reference undefined modules."""
        filepath = os.path.join(APP_DIR, "tools", "pipeline_tools.py")
        unresolved = _find_unresolved_module_attrs(filepath)
        assert not unresolved, (
            f"pipeline_tools.py has unresolved module references: {unresolved}"
        )

    def test_critical_pipeline_files_parseable(self):
        """All critical pipeline files must parse without SyntaxError."""
        critical_files = [
            os.path.join("pipeline", "data", "data_perticker_collection.py"),
            os.path.join("tools", "quant_tools.py"),
            os.path.join("tools", "pipeline_tools.py"),
            os.path.join("tools", "finance_tools.py"),
            os.path.join("pipeline", "analysis", "curation_pass.py"),
            os.path.join("pipeline", "analysis", "purge_pass.py"),
        ]
        errors = []
        for relpath in critical_files:
            filepath = os.path.join(APP_DIR, relpath)
            if not os.path.isfile(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"  {relpath}: {e}")

        assert not errors, f"Syntax errors in critical files:\n" + "\n".join(errors)

    def test_collector_datetime_imports(self):
        """All collector files that use datetime must import it."""
        collectors_dir = os.path.join(APP_DIR, "collectors")
        issues = []
        for fname in os.listdir(collectors_dir):
            if not fname.endswith(".py") or fname.startswith("__"):
                continue
            filepath = os.path.join(collectors_dir, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            imports = _get_imports(tree)
            # Check if datetime is used as a module attribute (datetime.datetime, etc.)
            uses_datetime_module = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id == "datetime" and "datetime" not in imports:
                        uses_datetime_module = True
                        break

            if uses_datetime_module:
                issues.append(fname)

        assert not issues, (
            f"Collectors use 'datetime.X' but don't import datetime: {issues}"
        )

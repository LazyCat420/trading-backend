"""
Regression test — ensures `held` and `position_context` are defined
before any reference in execute_v2_pipeline().

Root cause: A `log_manager.log_v2_cycle()` call referenced `held` and
`position_context` before they were assigned (lines 68-71 in the original
code). This caused an `UnboundLocalError` that crashed ALL 30 tickers
during the analysis phase.

This test uses AST analysis to guarantee that all assignments to critical
variables (held, position_context) appear BEFORE any read references,
so this class of bug can never be reintroduced.
"""

import ast
import inspect
import textwrap


def _first_assign_line(source: str, var_name: str) -> int | None:
    """Return the first line number where `var_name` is assigned."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # Regular assignment: var = ...
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return node.lineno
                # Annotated target in regular assign
        # Annotated assignment: var: type = ...
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == var_name:
                return node.lineno
    return None


def _first_read_line(source: str, var_name: str, skip_line: int | None = None) -> int | None:
    """Return the first line where `var_name` is READ (not assigned).
    
    Skips the assignment line itself to avoid false positives from
    patterns like `var = var.get(...)`.
    """
    tree = ast.parse(source)
    assign_lines: set[int] = set()

    # Collect all assignment lines for this variable
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    assign_lines.add(node.lineno)
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == var_name:
                assign_lines.add(node.lineno)

    # Find first Name Load that isn't on an assignment line
    first_read = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == var_name
            and isinstance(node.ctx, ast.Load)
            and node.lineno not in assign_lines
        ):
            if skip_line and node.lineno == skip_line:
                continue
            if first_read is None or node.lineno < first_read:
                first_read = node.lineno
    return first_read


def test_held_defined_before_first_read():
    """'held' must be assigned before it is ever referenced in execute_v2_pipeline."""
    from app.cognition.orchestration.runner import execute_v2_pipeline

    source = textwrap.dedent(inspect.getsource(execute_v2_pipeline))

    assign_line = _first_assign_line(source, "held")
    assert assign_line is not None, "'held' is never assigned in execute_v2_pipeline"

    read_line = _first_read_line(source, "held")
    assert read_line is not None, "'held' is never read in execute_v2_pipeline"

    assert assign_line < read_line, (
        f"REGRESSION: 'held' is first READ at line {read_line} but first "
        f"ASSIGNED at line {assign_line}. This will cause "
        f"UnboundLocalError at runtime."
    )


def test_position_context_defined_before_first_read():
    """'position_context' must be assigned before it is ever referenced."""
    from app.cognition.orchestration.runner import execute_v2_pipeline

    source = textwrap.dedent(inspect.getsource(execute_v2_pipeline))

    assign_line = _first_assign_line(source, "position_context")
    assert assign_line is not None, "'position_context' is never assigned"

    read_line = _first_read_line(source, "position_context")
    assert read_line is not None, "'position_context' is never read"

    assert assign_line < read_line, (
        f"REGRESSION: 'position_context' is first READ at line {read_line} "
        f"but first ASSIGNED at line {assign_line}. This will cause "
        f"UnboundLocalError at runtime."
    )

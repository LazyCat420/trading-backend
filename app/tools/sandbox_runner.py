"""
Sandbox Runner — Isolated subprocess for executing agent-written Python code.

Phase 5: Custom Agentic Tool Creation (Python Sandbox)

This runs in a subprocess spawned by script_sandbox.py. It uses
RestrictedPython to enforce guardrails:
  - No filesystem access (no open(), no os, no pathlib)
  - No network access (no socket, no urllib, no requests, no httpx)
  - No subprocess/exec/eval escape hatches
  - Only math, json, statistics, and pre-injected data are available

The agent writes code that operates on a pre-injected `DATA` dict,
and the result is captured from stdout (print statements).
"""

import sys
import json


# ── Allowed Modules (safe for quant computation) ──────────────────────
SAFE_MODULES = {
    "math": __import__("math"),
    "json": __import__("json"),
    "statistics": __import__("statistics"),
}


# ── Blocked Import Names (defense in depth) ───────────────────────────
BLOCKED_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "urllib", "http", "requests", "httpx",
    "importlib", "ctypes", "signal", "threading", "multiprocessing",
    "pickle", "shelve", "tempfile", "glob", "fnmatch",
    "webbrowser", "smtplib", "ftplib", "telnetlib",
    "builtins", "__builtins__",
})


def run_isolated(code: str, data_json: str = "{}"):
    """Execute agent code in a restricted environment.

    Args:
        code: Python source code to execute.
        data_json: JSON-serialized data dict injected as the `DATA` variable.
    """
    try:
        from RestrictedPython import compile_restricted
        from RestrictedPython.Guards import safe_builtins, full_write_guard
        from RestrictedPython.Eval import (
            default_guarded_getitem,
            default_guarded_getiter,
        )
        import RestrictedPython
    except ImportError as e:
        print(json.dumps({"error": f"RestrictedPython not installed: {e}"}))
        return

    # Parse injected data
    try:
        injected_data = json.loads(data_json)
    except Exception:
        injected_data = {}

    # Build the restricted global namespace
    my_globals = {
        "__builtins__": safe_builtins,
        "_print_": RestrictedPython.PrintCollector,
        "_getattr_": getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_write_": full_write_guard,
        # Safe modules for quant work
        "math": SAFE_MODULES["math"],
        "json": SAFE_MODULES["json"],
        "statistics": SAFE_MODULES["statistics"],
        # Injected data from the pipeline
        "DATA": injected_data,
    }

    # Compile with RestrictedPython
    try:
        byte_code = compile_restricted(code, "<agent_script>", "exec")
    except Exception as e:
        print(json.dumps({
            "error": f"Compilation Error (Safety Violation): {e}",
            "hint": "Your code uses a disallowed construct. Only math, json, statistics, and DATA are available.",
        }))
        return

    # Execute
    loc = {}
    try:
        exec(byte_code, my_globals, loc)  # noqa: S102
        if "_print" in loc and loc["_print"]:
            output = loc["_print"]()
            print(output)
        elif "result" in loc:
            # Allow agents to set a `result` variable as an alternative to print
            print(json.dumps({"result": loc["result"]}))
        else:
            print(json.dumps({"status": "ok", "note": "Code executed with no output."}))
    except Exception as e:
        print(json.dumps({"error": f"Runtime Error: {e}"}))


if __name__ == "__main__":
    # Read code from stdin, optional data from argv
    code = sys.stdin.read()
    data_json = sys.argv[1] if len(sys.argv) > 1 else "{}"
    run_isolated(code, data_json)

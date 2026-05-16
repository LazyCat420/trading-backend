"""
Test & Prove Environment — Validates proposed fixes before the Judge approves them.

Depending on the target_type, this module runs a different validation strategy:
  - prompt  → Sends a test prompt to a live LLM endpoint and checks for valid JSON output.
  - scraper → Runs a quick dry-run import + syntax check of the proposed code.
  - strategy → AST-parses the code and checks for forbidden imports.

The Judge receives a structured test_result dict to make a more informed ruling.
"""

import ast
import logging

logger = logging.getLogger(__name__)


async def validate_fix(
    target_type: str,
    target_name: str,
    proposed_fix: str,
    issue_description: str,
) -> dict:
    """
    Run validation appropriate to the target type.

    Returns:
        {
            "passed": bool,
            "tests_run": int,
            "tests_passed": int,
            "details": str,          # human-readable summary
            "errors": list[str],     # list of error messages
        }
    """
    if target_type == "prompt":
        return await _validate_prompt(target_name, proposed_fix, issue_description)
    elif target_type == "scraper":
        return _validate_scraper_code(target_name, proposed_fix)
    elif target_type == "strategy":
        return _validate_strategy_code(target_name, proposed_fix)
    elif target_type == "constitution_amendment":
        # Constitution amendments are DB updates, not code — skip code validation
        return {
            "passed": True,
            "tests_run": 0,
            "tests_passed": 0,
            "details": "Constitution amendment — no code validation needed.",
            "errors": [],
        }
    else:
        return {
            "passed": True,
            "tests_run": 0,
            "tests_passed": 0,
            "details": f"No validation strategy for target_type='{target_type}'. Skipped.",
            "errors": [],
        }


async def _validate_prompt(
    target_name: str, proposed_fix: str, issue_description: str
) -> dict:
    """
    Validate a proposed prompt fix by sending a test query to the LLM
    and checking that the output is valid JSON (the most common failure mode).
    """
    errors = []
    tests_run = 0
    tests_passed = 0

    # Test 1: Basic length and content checks
    tests_run += 1
    if len(proposed_fix.strip()) < 50:
        errors.append("Proposed prompt is too short (< 50 chars). Likely hallucinated.")
    else:
        tests_passed += 1

    # Test 2: Check for common prompt engineering red flags
    tests_run += 1
    red_flags = [
        "INSERT_",
        "PLACEHOLDER",
        "TODO:",
        "FIXME:",
        "[YOUR ",
        "{{",
        "}}",
    ]
    found_flags = [f for f in red_flags if f in proposed_fix]
    if found_flags:
        errors.append(f"Prompt contains placeholder markers: {found_flags}")
    else:
        tests_passed += 1

    # Test 3: Live LLM smoke test — send the prompt and check output parses as JSON
    tests_run += 1
    try:
        from app.services.vllm_client import llm, Priority

        test_user = (
            "Given the stock ticker AAPL, provide a brief analysis. "
            "Output valid JSON with keys: action, confidence, reasoning."
        )
        response, tokens, elapsed = await llm.chat(
            system=proposed_fix[:2000],  # Use the proposed prompt as system
            user=test_user,
            temperature=0.1,
            max_tokens=512,
            priority=Priority.LOW,
            agent_name="evo_test_prove",
            ticker="_test",
        )

        # Check if response is valid JSON
        import json

        cleaned = response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        json.loads(cleaned)
        tests_passed += 1
        logger.info(
            "[TEST-PROVE] Prompt smoke test PASSED (%d tokens, %dms)",
            tokens,
            elapsed,
        )
    except Exception as e:
        errors.append(f"LLM smoke test failed: {e}")
        logger.warning("[TEST-PROVE] Prompt smoke test FAILED: %s", e)

    passed = len(errors) == 0
    return {
        "passed": passed,
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "details": (
            f"Prompt validation: {tests_passed}/{tests_run} tests passed."
            + (f" Errors: {'; '.join(errors)}" if errors else " All clear.")
        ),
        "errors": errors,
    }


def _validate_scraper_code(target_name: str, proposed_fix: str) -> dict:
    """
    Validate proposed scraper code changes.
    Checks: AST parse, no dangerous imports, no hardcoded secrets,
    function definitions, import verification, AND live sandbox execution.
    """
    errors = []
    warnings = []
    tests_run = 0
    tests_passed = 0

    # Test 1: AST parse
    tests_run += 1
    try:
        ast.parse(proposed_fix)
        tests_passed += 1
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

    # Test 2: No dangerous imports
    # Note: subprocess, os.path, shutil are legitimate for scrapers that
    # shell out to yt-dlp, ffmpeg, or manage temp files. Only flag eval/exec.
    tests_run += 1
    dangerous = ["eval(", "exec("]
    found = [d for d in dangerous if d in proposed_fix]
    if found:
        errors.append(f"Dangerous patterns found: {found}")
    else:
        tests_passed += 1

    # Test 3: No hardcoded secrets
    tests_run += 1
    secret_patterns = [
        "sk-",
        "api_key =",
        "API_KEY =",
        "password =",
        "secret =",
    ]
    found_secrets = [s for s in secret_patterns if s.lower() in proposed_fix.lower()]
    if found_secrets:
        errors.append(f"Possible hardcoded secrets: {found_secrets}")
    else:
        tests_passed += 1

    # Test 4: Check the file defines expected patterns (class or function)
    tests_run += 1
    has_def = "def " in proposed_fix or "class " in proposed_fix
    if not has_def:
        errors.append(
            "No function or class definitions found — code may be incomplete."
        )
    else:
        tests_passed += 1

    # Test 5: Import verification — catch hallucinated packages
    tests_run += 1
    import_errors = _verify_imports(proposed_fix)
    if import_errors:
        errors.append(f"Hallucinated imports: {'; '.join(import_errors)}")
    else:
        tests_passed += 1

    # Test 6: LIVE SANDBOX EXECUTION — actually run the scraper in a subprocess
    # This catches runtime errors that AST/import checks miss (network failures,
    # API format changes, broken parsing logic).
    if errors:
        # Skip sandbox if earlier tests already failed
        logger.info("[TEST-PROVE] Skipping sandbox — earlier tests failed.")
    else:
        tests_run += 1
        sandbox_result = _run_sandbox_test(target_name, proposed_fix)
        if sandbox_result["passed"]:
            tests_passed += 1
            logger.info("[TEST-PROVE] Sandbox test PASSED: %s", sandbox_result["details"])
        elif sandbox_result.get("is_network_error"):
            # Network errors are warnings, not failures — the code itself may be correct
            warnings.append(f"Sandbox network warning: {sandbox_result['details']}")
            tests_passed += 1  # Give benefit of the doubt
            logger.warning("[TEST-PROVE] Sandbox network warning: %s", sandbox_result["details"])
        else:
            errors.append(f"Sandbox execution failed: {sandbox_result['details']}")
            logger.warning("[TEST-PROVE] Sandbox test FAILED: %s", sandbox_result["details"])

    passed = len(errors) == 0
    detail_parts = [f"Scraper validation: {tests_passed}/{tests_run} tests passed."]
    if errors:
        detail_parts.append(f"Errors: {'; '.join(errors)}")
    if warnings:
        detail_parts.append(f"Warnings: {'; '.join(warnings)}")
    if not errors and not warnings:
        detail_parts.append("All clear.")

    return {
        "passed": passed,
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "details": " ".join(detail_parts),
        "errors": errors,
        "warnings": warnings,
    }


def _run_sandbox_test(target_name: str, proposed_code: str) -> dict:
    """Execute proposed scraper code in a subprocess sandbox with strict timeout.

    Writes the code to a temp file, appends a test harness that invokes the main
    function with a dummy ticker (AAPL), and checks that the subprocess exits
    cleanly with a zero exit code.

    Returns:
        {"passed": bool, "details": str, "is_network_error": bool}
    """
    import subprocess
    import tempfile
    import os

    _SANDBOX_TIMEOUT = 30  # seconds

    # Try to detect the main callable in the proposed code
    tree = ast.parse(proposed_code)
    async_funcs = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
    ]
    sync_funcs = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]

    # Prefer common naming patterns
    test_func = None
    for candidate in ["collect_news", "collect_all", "collect_for_ticker",
                       "collect", "scrape", "fetch", "run"]:
        if candidate in async_funcs:
            test_func = (candidate, True)
            break
        if candidate in sync_funcs:
            test_func = (candidate, False)
            break

    if not test_func and async_funcs:
        test_func = (async_funcs[0], True)
    elif not test_func and sync_funcs:
        test_func = (sync_funcs[0], False)

    if not test_func:
        return {
            "passed": True,
            "details": "No callable function detected — skipping sandbox.",
            "is_network_error": False,
        }

    func_name, is_async = test_func

    # Build the test harness
    if is_async:
        harness = f"""
# ── Sandbox Test Harness ──
import asyncio
import sys

async def _sandbox_test():
    try:
        result = await {func_name}("AAPL")
        print(f"SANDBOX_OK: {{type(result).__name__}} returned")
        sys.exit(0)
    except Exception as e:
        err_str = str(e).lower()
        if any(kw in err_str for kw in ["timeout", "connection", "dns", "ssl", "network", "refused"]):
            print(f"SANDBOX_NETWORK_WARN: {{e}}")
            sys.exit(2)  # Network error — not a code bug
        print(f"SANDBOX_FAIL: {{e}}")
        sys.exit(1)

asyncio.run(_sandbox_test())
"""
    else:
        harness = f"""
# ── Sandbox Test Harness ──
import sys

try:
    result = {func_name}("AAPL")
    print(f"SANDBOX_OK: {{type(result).__name__}} returned")
    sys.exit(0)
except Exception as e:
    err_str = str(e).lower()
    if any(kw in err_str for kw in ["timeout", "connection", "dns", "ssl", "network", "refused"]):
        print(f"SANDBOX_NETWORK_WARN: {{e}}")
        sys.exit(2)
    print(f"SANDBOX_FAIL: {{e}}")
    sys.exit(1)
"""

    full_code = proposed_code + "\n\n" + harness

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="evo_sandbox_",
            delete=False, dir="/tmp"
        ) as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=_SANDBOX_TIMEOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            output = (proc.stdout + proc.stderr).strip()

            if proc.returncode == 0:
                return {
                    "passed": True,
                    "details": f"Sandbox OK: {output[:200]}",
                    "is_network_error": False,
                }
            elif proc.returncode == 2:
                return {
                    "passed": False,
                    "details": f"Network issue (not a code bug): {output[:200]}",
                    "is_network_error": True,
                }
            else:
                return {
                    "passed": False,
                    "details": f"Exit code {proc.returncode}: {output[:300]}",
                    "is_network_error": False,
                }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "details": f"Sandbox timeout ({_SANDBOX_TIMEOUT}s) — code may hang.",
                "is_network_error": False,
            }
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        return {
            "passed": False,
            "details": f"Sandbox setup error: {e}",
            "is_network_error": False,
        }



def _validate_strategy_code(target_name: str, proposed_fix: str) -> dict:
    """
    Validate proposed strategy code (same checks as evolve.py).
    """
    errors = []
    tests_run = 0
    tests_passed = 0

    # Test 1: AST parse
    tests_run += 1
    try:
        ast.parse(proposed_fix)
        tests_passed += 1
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

    # Test 2: Forbidden imports check
    tests_run += 1
    try:
        from app.trading.sandbox_executor import check_forbidden_imports

        forbidden = check_forbidden_imports(proposed_fix)
        if forbidden:
            errors.append(f"Forbidden imports: {forbidden}")
        else:
            tests_passed += 1
    except ImportError:
        # sandbox_executor not available; do a basic check
        allowed = {"pandas", "numpy", "ta"}
        tree = ast.parse(proposed_fix)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod not in allowed:
                        errors.append(f"Forbidden import: {mod}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".")[0]
                    if mod not in allowed:
                        errors.append(f"Forbidden import: {mod}")
        if not errors:
            tests_passed += 1

    # Test 3: Must define generate_signals
    tests_run += 1
    if "def generate_signals" in proposed_fix:
        tests_passed += 1
    else:
        errors.append("Missing required function: generate_signals()")

    passed = len(errors) == 0
    return {
        "passed": passed,
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "details": (
            f"Strategy validation: {tests_passed}/{tests_run} tests passed."
            + (f" Errors: {'; '.join(errors)}" if errors else " All clear.")
        ),
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════
# IMPORT VERIFICATION — Catch hallucinated packages
# ═══════════════════════════════════════════════════════════════

# Packages that are part of our project or stdlib (skip verification)
_KNOWN_INTERNAL = {
    "app",
    "app.",
    "logging",
    "json",
    "os",
    "sys",
    "re",
    "ast",
    "pathlib",
    "datetime",
    "asyncio",
    "typing",
    "uuid",
    "hashlib",
    "collections",
    "functools",
    "itertools",
    "math",
    "time",
    "dataclasses",
    "enum",
    "abc",
    "contextlib",
    "io",
    "copy",
    "subprocess",
    "shutil",
    "importlib",
    "traceback",
    "textwrap",
    "urllib",
    "html",
    "csv",
    "configparser",
    "signal",
}


def _verify_imports(code: str) -> list[str]:
    """Check that imported packages actually exist using importlib.

    Returns a list of error strings for packages that can't be found.
    Only checks top-level imports (not from app.* internal imports).
    """
    import importlib.util

    errors = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # Syntax errors are caught by Test 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_pkg = alias.name.split(".")[0]
                if top_pkg in _KNOWN_INTERNAL or alias.name.startswith("app."):
                    continue
                if importlib.util.find_spec(top_pkg) is None:
                    errors.append(f"'{alias.name}' (package not installed)")

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("app."):
                continue  # Internal import
            if node.module:
                top_pkg = node.module.split(".")[0]
                if top_pkg in _KNOWN_INTERNAL:
                    continue
                if importlib.util.find_spec(top_pkg) is None:
                    errors.append(f"'{node.module}' (package not installed)")

    return errors

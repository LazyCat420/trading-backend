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
    Checks: AST parse, no dangerous imports, no hardcoded secrets.
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

    passed = len(errors) == 0
    return {
        "passed": passed,
        "tests_run": tests_run,
        "tests_passed": tests_passed,
        "details": (
            f"Scraper validation: {tests_passed}/{tests_run} tests passed."
            + (f" Errors: {'; '.join(errors)}" if errors else " All clear.")
        ),
        "errors": errors,
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

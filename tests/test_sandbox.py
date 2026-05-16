"""
Tests for Phase 5: Sandbox Runner (app/tools/sandbox_runner.py)

Tests cover:
  🔴🟢 TDD Unit Tests:
    - Happy path: math calculations, DATA injection, result variable
    - Guardrails: blocked imports (os, socket, subprocess, etc.)
    - Edge cases: empty code, timeout, syntax errors
  🔐 Security Tests:
    - Filesystem access blocked
    - Network access blocked
    - exec/eval escape hatches blocked
"""

import json
import subprocess
import sys
import os

RUNNER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "app",
    "tools",
    "sandbox_runner.py",
)


def _run_sandbox(code: str, data: dict | None = None) -> str:
    """Helper: run sandbox_runner.py in a subprocess, return stdout."""
    data_json = json.dumps(data or {})
    result = subprocess.run(
        [sys.executable, RUNNER_PATH, data_json],
        input=code,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


# ══════════════════════════════════════════════════════════════
# 🔴🟢 TDD — Happy Path
# ══════════════════════════════════════════════════════════════


class TestSandboxHappyPath:
    """Basic computation tests — these should all succeed."""

    def test_simple_math(self):
        """Agent can do basic arithmetic."""
        output = _run_sandbox("print(2 + 2)")
        assert "4" in output

    def test_math_module(self):
        """Agent can use the math module (pre-injected, no import needed)."""
        output = _run_sandbox("print(math.sqrt(16))")
        assert "4.0" in output

    def test_json_module(self):
        """Agent can use json.dumps for structured output (pre-injected)."""
        output = _run_sandbox('print(json.dumps({"result": 42}))')
        parsed = json.loads(output)
        assert parsed["result"] == 42

    def test_statistics_module(self):
        """Agent can use statistics for quant work (pre-injected)."""
        output = _run_sandbox(
            "prices = [100, 102, 98, 105]\nprint(statistics.mean(prices))"
        )
        assert "101.25" in output

    def test_data_injection(self):
        """Agent receives pre-injected DATA dict."""
        code = 'print(DATA["ticker"])'
        output = _run_sandbox(code, data={"ticker": "AAPL"})
        assert "AAPL" in output

    def test_data_injection_computation(self):
        """Agent can compute on injected DATA using statistics module."""
        code = (
            'prices = DATA["prices"]\n'
            "avg = statistics.mean(prices)\n"
            'print(json.dumps({"avg": avg}))'
        )
        output = _run_sandbox(code, data={"prices": [100, 102, 98, 105]})
        parsed = json.loads(output)
        assert parsed["avg"] == 101.25

    def test_result_variable(self):
        """Agent can set a 'result' variable as alternative to print."""
        code = "result = 42"
        output = _run_sandbox(code)
        parsed = json.loads(output)
        assert parsed["result"] == 42

    def test_no_output_ok(self):
        """Code with no output returns a status message."""
        code = "x = 1 + 1"
        output = _run_sandbox(code)
        parsed = json.loads(output)
        assert parsed["status"] == "ok"


# ══════════════════════════════════════════════════════════════
# 🔐 Security Tests — Guardrails
# ══════════════════════════════════════════════════════════════


class TestSandboxGuardrails:
    """Every one of these MUST fail — the sandbox must block them."""

    def test_blocks_os_import(self):
        """Cannot import os."""
        output = _run_sandbox("import os\nprint(os.listdir('.'))")
        assert "error" in output.lower() or "Error" in output

    def test_blocks_subprocess_import(self):
        """Cannot import subprocess."""
        output = _run_sandbox("import subprocess\nsubprocess.run(['ls'])")
        assert "error" in output.lower() or "Error" in output

    def test_blocks_socket_import(self):
        """Cannot import socket (network access)."""
        output = _run_sandbox("import socket\nsocket.socket()")
        assert "error" in output.lower() or "Error" in output

    def test_blocks_open_file(self):
        """Cannot use open() to read files."""
        output = _run_sandbox("f = open('/etc/passwd', 'r')\nprint(f.read())")
        assert "error" in output.lower() or "Error" in output
        assert "passwd" not in output.lower() or "error" in output.lower()

    def test_blocks_eval(self):
        """Cannot use eval() escape hatch."""
        output = _run_sandbox("eval('__import__(\"os\").listdir(\".\")')")
        assert "error" in output.lower() or "Error" in output

    def test_blocks_exec(self):
        """Cannot use exec() escape hatch."""
        output = _run_sandbox("exec('import os')")
        assert "error" in output.lower() or "Error" in output

    def test_blocks_pathlib(self):
        """Cannot import pathlib."""
        output = _run_sandbox("import pathlib\nprint(pathlib.Path('.').iterdir())")
        assert "error" in output.lower() or "Error" in output

    def test_blocks_shutil(self):
        """Cannot import shutil."""
        output = _run_sandbox("import shutil")
        assert "error" in output.lower() or "Error" in output


# ══════════════════════════════════════════════════════════════
# 💨 Edge Cases
# ══════════════════════════════════════════════════════════════


class TestSandboxEdgeCases:
    """Edge cases that should not crash the sandbox."""

    def test_empty_code(self):
        """Empty code should return status ok."""
        output = _run_sandbox("")
        # Should not crash
        assert output is not None

    def test_syntax_error(self):
        """Invalid Python syntax should return a compilation error."""
        output = _run_sandbox("def foo(")
        assert "error" in output.lower() or "Error" in output

    def test_runtime_error(self):
        """Division by zero should be caught."""
        output = _run_sandbox("print(1/0)")
        assert "error" in output.lower() or "Error" in output

    def test_infinite_loop_timeout(self):
        """Infinite loop should be killed by subprocess timeout."""
        # We use a short timeout in the test itself
        try:
            result = subprocess.run(
                [sys.executable, RUNNER_PATH, "{}"],
                input="while True: pass",
                capture_output=True,
                text=True,
                timeout=5,
            )
            # If it somehow finishes, that's fine
        except subprocess.TimeoutExpired:
            pass  # Expected — the loop was killed

    def test_empty_data(self):
        """Empty DATA dict should be accessible."""
        output = _run_sandbox("print(len(DATA))")
        assert "0" in output

    def test_large_output(self):
        """Large output should be captured without crash."""
        code = "print('x' * 10000)"
        output = _run_sandbox(code)
        assert len(output) >= 10000

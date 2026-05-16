"""
Sandbox Executor — runs generated strategy code in an isolated subprocess.

Security:
  - All env vars stripped (no API keys leak)
  - Forbidden imports checked via AST before execution
  - Hard 60-second timeout
  - Structured JSON-only output
"""

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Imports that are NEVER allowed in generated strategy code
FORBIDDEN_IMPORTS = {
    "os",
    "subprocess",
    "sys",
    "shutil",
    "pathlib",
    "requests",
    "httpx",
    "urllib",
    "socket",
    "alpaca",
    "alpaca_trade_api",
    "sqlite3",
    "psycopg",
    "psycopg2",
    "pymongo",
}

# Modules with write-mode file access that must be rejected
FORBIDDEN_OPEN_MODES = {"w", "a", "x", "wb", "ab", "xb", "w+", "a+", "r+"}

TIMEOUT_SECONDS = 60


@dataclass
class ExecutionResult:
    """Result of a sandboxed strategy execution."""

    status: str  # SUCCESS | SYNTAX_ERROR | RUNTIME_ERROR | TIMEOUT | FORBIDDEN_IMPORT
    metrics: Optional[dict] = None
    error_message: str = ""
    traceback: str = ""


def check_forbidden_imports(code: str) -> Optional[str]:
    """Parse code with AST and check for forbidden imports/calls.

    Returns an error message if forbidden patterns found, else None.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # Check `import X`
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in FORBIDDEN_IMPORTS:
                    return f"Forbidden import: {alias.name}"

        # Check `from X import Y`
        if isinstance(node, ast.ImportFrom) and node.module:
            module_root = node.module.split(".")[0]
            if module_root in FORBIDDEN_IMPORTS:
                return f"Forbidden import: from {node.module}"

        # Check open() calls with write modes
        if isinstance(node, ast.Call):
            func = node.func
            fname = ""
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname == "open" and len(node.args) >= 2:
                mode_arg = node.args[1]
                if isinstance(mode_arg, ast.Constant) and isinstance(
                    mode_arg.value, str
                ):
                    if mode_arg.value in FORBIDDEN_OPEN_MODES:
                        return f"Forbidden: open() with write mode '{mode_arg.value}'"
            # Check keyword mode= argument
            if fname == "open":
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        if (
                            isinstance(kw.value.value, str)
                            and kw.value.value in FORBIDDEN_OPEN_MODES
                        ):
                            return (
                                f"Forbidden: open() with write mode '{kw.value.value}'"
                            )

    return None


def _build_runner_script(strategy_path: str, data_path: str) -> str:
    """Build the Python script that runs inside the subprocess."""
    return textwrap.dedent(f"""\
        import json
        import sys
        import pandas as pd
        import numpy as np

        # Load OHLCV data
        df = pd.read_parquet("{data_path}")

        # Import the strategy
        sys.path.insert(0, "{os.path.dirname(strategy_path)}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("strategy", "{strategy_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Run the strategy
        signals = mod.generate_signals(df)

        # Compute metrics
        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=df.index)

        # Simple backtester: returns from signal shifts
        returns = df["close"].pct_change().fillna(0)
        # Signal at t generates return at t+1 (no look-ahead)
        strategy_returns = signals.shift(1).fillna(0) * returns

        n_trades = int((signals.diff().fillna(0) != 0).sum())
        total_return = float((1 + strategy_returns).prod() - 1)

        # Sharpe ratio (annualized, 252 trading days)
        if strategy_returns.std() > 0:
            sharpe = float(strategy_returns.mean() / strategy_returns.std() * (252 ** 0.5))
        else:
            sharpe = 0.0

        # Max drawdown
        cum = (1 + strategy_returns).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_drawdown = float(drawdown.min())

        # Win rate
        winning = (strategy_returns > 0).sum()
        total_periods = (strategy_returns != 0).sum()
        win_rate = float(winning / total_periods) if total_periods > 0 else 0.0

        result = {{
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
            "n_trades": n_trades,
        }}
        print(json.dumps(result))
    """)


def run_sandboxed_backtest(
    strategy_code: str,
    data_path: str,
    timeout: int = TIMEOUT_SECONDS,
) -> ExecutionResult:
    """Execute strategy code in a sandboxed subprocess.

    Args:
        strategy_code: Full Python source of the strategy.
        data_path: Path to a Parquet file with OHLCV data.
        timeout: Max seconds before kill.

    Returns:
        ExecutionResult with status and metrics.
    """
    # Step 1: AST validation
    try:
        ast.parse(strategy_code)
    except SyntaxError as e:
        return ExecutionResult(
            status="SYNTAX_ERROR",
            error_message=str(e),
            traceback=f"Line {e.lineno}: {e.msg}",
        )

    # Step 2: Forbidden import check
    forbidden = check_forbidden_imports(strategy_code)
    if forbidden:
        return ExecutionResult(
            status="SYNTAX_ERROR",
            error_message=forbidden,
            traceback=forbidden,
        )

    # Step 3: Write strategy to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="evo_strategy_"
    ) as f:
        f.write(strategy_code)
        strategy_path = f.name

    # Step 4: Write runner script
    runner_script = _build_runner_script(strategy_path, data_path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="evo_runner_"
    ) as f:
        f.write(runner_script)
        runner_path = f.name

    try:
        # Step 5: Execute in clean environment (no env vars = no API keys)
        clean_env = {
            "PATH": "/usr/bin:/usr/local/bin",
            "HOME": tempfile.gettempdir(),
            "PYTHONPATH": "",
        }

        proc = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=clean_env,
            cwd=tempfile.gettempdir(),
        )

        if proc.returncode != 0:
            return ExecutionResult(
                status="RUNTIME_ERROR",
                error_message=proc.stderr[-500:]
                if proc.stderr
                else "Unknown runtime error",
                traceback=proc.stderr[-2000:] if proc.stderr else "",
            )

        # Step 6: Parse structured output
        try:
            output = proc.stdout.strip()
            # Find the last JSON line (in case of debug prints)
            for line in reversed(output.split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    metrics = json.loads(line)
                    return ExecutionResult(status="SUCCESS", metrics=metrics)
            return ExecutionResult(
                status="RUNTIME_ERROR",
                error_message="No JSON output from strategy",
                traceback=output[-500:],
            )
        except json.JSONDecodeError as e:
            return ExecutionResult(
                status="RUNTIME_ERROR",
                error_message=f"Invalid JSON output: {e}",
                traceback=proc.stdout[-500:],
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            status="TIMEOUT",
            error_message=f"Strategy exceeded {timeout}s timeout",
        )
    finally:
        # Clean up temp files
        for p in [strategy_path, runner_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

"""
Script Sandbox — Allows agents to write and execute Python code for data processing.

Phase 5: Custom Agentic Tool Creation (Python Sandbox)

This tool enables LLM agents to write custom quant equations, data
transformations, and statistical calculations. The code is executed in
a sandboxed subprocess using RestrictedPython.

Available to the agent inside the sandbox:
  - `math` module (sin, cos, log, sqrt, etc.)
  - `json` module (dumps, loads)
  - `statistics` module (mean, median, stdev, etc.)
  - `DATA` dict — pre-injected data from the pipeline context
  - `print()` — captured as output
  - `result` variable — alternative to print() for structured output

NOT available (guardrails):
  - Filesystem access (no open, os, pathlib)
  - Network access (no socket, urllib, requests, httpx)
  - Subprocess/exec/eval (no shell escape)
  - Any import beyond math/json/statistics
"""

import json
import logging
from pydantic import BaseModel, Field

from app.tools.registry import registry, PermissionLevel

logger = logging.getLogger(__name__)


class ExecuteQuantScriptInput(BaseModel):
    """Input model for the execute_quant_script tool."""

    code: str = Field(
        ...,
        description=(
            "Python code to execute. Has access to math, json, statistics modules "
            "and a DATA dict. Use print() or set a 'result' variable for output."
        ),
    )
    data: dict = Field(
        default_factory=dict,
        description=(
            "Optional data dict injected as the DATA variable in the sandbox. "
            "Example: {'prices': [100, 102, 98], 'volume': [1000, 1200, 900]}"
        ),
    )
    timeout_seconds: int = Field(
        10,
        description="Maximum execution time in seconds.",
        ge=1,
        le=30,
    )


@registry.register(
    name="execute_quant_script",
    description=(
        "Execute custom Python code for quant calculations, data transformations, "
        "or statistical analysis. The code runs in a secure sandbox with access to "
        "math, json, statistics modules and a DATA dict. "
        "Use this for risk/reward calculations, custom indicators, portfolio math, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute. Has access to math, json, statistics modules "
                    "and a DATA dict with pipeline data. Use print() for output."
                ),
            },
            "data": {
                "type": "object",
                "description": (
                    "Data dict injected as the DATA variable. "
                    "Example: {'prices': [100, 102, 98], 'volume': [1000, 1200, 900]}"
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max execution time in seconds (1-30).",
            },
        },
        "required": ["code"],
    },
    permission=PermissionLevel.WRITE,  # Elevated but not destructive — runs in sandbox
    input_model=ExecuteQuantScriptInput,
    tags=["quant", "calculation", "sandbox", "custom"],
)
async def execute_quant_script(
    code: str,
    data: dict | None = None,
    timeout_seconds: int = 10,
) -> str:
    """Execute Python code in a restricted sandbox subprocess.

    The code is compiled and executed by RestrictedPython in a child
    process to prevent any access to the host filesystem or network.
    """
    import asyncio
    import os
    import sys

    logger.info(
        "[SANDBOX] Executing quant script (%d chars, timeout=%ds)",
        len(code),
        timeout_seconds,
    )

    runner_path = os.path.join(os.path.dirname(__file__), "sandbox_runner.py")
    data_json = json.dumps(data or {})

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            runner_path,
            data_json,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=code.encode()),
                timeout=timeout_seconds,
            )
            stdout = stdout_bytes.decode().strip()
            stderr = stderr_bytes.decode().strip()

            if stderr:
                logger.warning("[SANDBOX] stderr: %s", stderr[:200])
                return json.dumps({
                    "output": stdout,
                    "stderr": stderr[:500],
                    "status": "completed_with_warnings",
                })

            return stdout or json.dumps({"status": "ok", "note": "No output produced."})

        except asyncio.TimeoutError:
            process.kill()
            logger.warning(
                "[SANDBOX] Script exceeded %ds timeout — killed", timeout_seconds
            )
            return json.dumps({
                "error": f"Execution exceeded {timeout_seconds}s timeout and was terminated.",
                "hint": "Simplify your code or increase timeout_seconds.",
            })

    except Exception as e:
        logger.error("[SANDBOX] System error: %s", e)
        return json.dumps({"error": f"Sandbox system error: {e}"})


# ── Legacy alias: keep old tool name working ───────────────────────────
# The old run_python_script is now deprecated in favor of execute_quant_script
# but we keep it registered so existing prompts don't break.
@registry.register(
    name="run_python_script",
    description="[DEPRECATED — use execute_quant_script instead] Execute Python code in a restricted sandbox.",
    permission=PermissionLevel.DESTRUCTIVE,
    input_model=ExecuteQuantScriptInput,
)
async def run_python_script(
    code: str,
    data: dict | None = None,
    timeout_seconds: int = 5,
) -> str:
    """Legacy wrapper — delegates to execute_quant_script."""
    return await execute_quant_script(code, data, timeout_seconds)


class ExecutePythonInput(BaseModel):
    """Input model for the execute_python tool."""

    code: str | None = Field(
        None,
        description="The Python code block to execute.",
    )
    python: str | None = Field(
        None,
        description="Fallback parameter name for the Python code block to execute.",
    )
    data: dict = Field(
        default_factory=dict,
        description="Optional context data dict injected as the DATA variable.",
    )
    timeout_seconds: int = Field(
        10,
        description="Maximum execution time in seconds.",
        ge=1,
        le=30,
    )


@registry.register(
    name="execute_python",
    description=(
        "Execute Python code blocks in a secure restricted sandbox. "
        "Allows standard calculations, data transformations, and math processing. "
        "Returns the output from print statements."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code block to execute.",
            },
            "python": {
                "type": "string",
                "description": "Fallback parameter name for the Python code block.",
            },
            "data": {
                "type": "object",
                "description": "Optional context data dict injected as the DATA variable.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max execution time in seconds (1-30).",
            },
        },
    },
    permission=PermissionLevel.WRITE,
    input_model=ExecutePythonInput,
    tags=["quant", "python", "sandbox", "execute"],
)
async def execute_python(
    code: str | None = None,
    python: str | None = None,
    data: dict | None = None,
    timeout_seconds: int = 10,
) -> str:
    """Execute Python code using the RestrictedPython sandbox."""
    actual_code = code or python
    if not actual_code:
        return json.dumps({"error": "Missing required parameter: 'code' or 'python'."})
    return await execute_quant_script(actual_code, data, timeout_seconds)


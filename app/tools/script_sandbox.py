"""
Script Sandbox — Allows agents to write and execute Python code for data processing.

WARNING: This executes code on the local host environment.
For production, this MUST be moved to a Dockerized environment or Firecracker microVM.
"""

import logging
from pydantic import BaseModel, Field

from app.tools.registry import registry, PermissionLevel

logger = logging.getLogger(__name__)


class ScriptSandboxInput(BaseModel):
    code: str = Field(
        ..., description="The Python code to execute. Must be self-contained."
    )
    timeout_seconds: int = Field(5, description="Maximum execution time in seconds.")


@registry.register(
    name="run_python_script",
    description="Execute Python code to process data or perform calculations. Output is returned from stdout (print statements).",
    permission=PermissionLevel.DESTRUCTIVE,  # Requires explicit approval due to security risks
    input_model=ScriptSandboxInput,
)
async def run_python_script(code: str, timeout_seconds: int = 5) -> str:
    """Execute Python code in a restricted local environment.

    Captures stdout and returns it.
    """
    import asyncio
    import os
    import sys

    logger.info("[SANDBOX] Executing code in isolated subprocess with RestrictedPython")

    runner_path = os.path.join(os.path.dirname(__file__), "sandbox_runner.py")

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            runner_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=code.encode()), timeout=timeout_seconds
            )
            stdout = stdout_bytes.decode().strip()
            stderr = stderr_bytes.decode().strip()

            if stderr:
                return f"Execution Stderr:\n{stderr}\nStdout:\n{stdout}"
            return stdout or "Code executed successfully with no output."

        except asyncio.TimeoutError:
            process.kill()
            return f"Execution Error: Code exceeded {timeout_seconds} seconds timeout and was terminated."

    except Exception as e:
        return f"System Error executing sandbox: {e}"

"""
Native Playwright Browser Tools for the trading bot agents.

Provides a managed browser session that agents can control via tool calls.
Also exposes an MCP-compatible server interface for external clients.

Tools:
  - browser_navigate: Go to a URL, return page content as markdown
  - browser_click: Click an element by CSS selector
  - browser_type: Type text into an input field
  - browser_screenshot: Take a screenshot, save to disk, return path
  - browser_evaluate: Run arbitrary JS in the page and return result
  - run_playwright_script: Execute a full custom Python Playwright script
"""

import asyncio
import json
import logging
import sys
import time

from pathlib import Path

from app.tools.registry import registry

logger = logging.getLogger(__name__)

# ── Singleton browser session ───────────────────────────────────────

_browser = None
_context = None
_page = None
_pw = None
_lock = asyncio.Lock()

SCREENSHOTS_DIR = Path("d:/Github/vllm-trading-bot/scratch/screenshots")
SCRIPTS_DIR = Path("d:/Github/vllm-trading-bot/scratch/browser_scripts")


async def _ensure_browser():
    """Lazily launch a shared Chromium browser instance."""
    global _browser, _context, _page, _pw

    async with _lock:
        if _browser and _browser.is_connected():
            if _page and not _page.is_closed():
                return _page
            # Page was closed, make a new one
            _page = await _context.new_page()
            return _page

        # Fresh launch
        try:
            from playwright.async_api import async_playwright

            _pw = await async_playwright().start()
            _browser = await _pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            )
            _context = await _browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            )
            _page = await _context.new_page()
            logger.info("[Browser] Launched headless Chromium")
            return _page
        except ImportError:
            logger.error("[Browser] Playwright is not installed. Browser tools disabled.")
            raise RuntimeError("Playwright is not installed. Cannot launch browser.")


async def _close_browser():
    """Shut down the browser cleanly."""
    global _browser, _context, _page, _pw
    try:
        if _browser:
            await _browser.close()
        if _pw:
            await _pw.stop()
    except Exception as e:
        logger.warning(f"[Browser] Error closing: {e}")
    finally:
        _browser = _context = _page = _pw = None


def _html_to_markdown(html: str, max_chars: int = 8000) -> str:
    """Convert HTML to a minimal markdown representation for LLM consumption."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer, header
    for tag in soup.find_all(
        ["script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"]
    ):
        tag.decompose()

    lines = []
    for el in soup.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
            "td",
            "th",
            "a",
            "span",
            "div",
            "pre",
            "code",
        ]
    ):
        text = el.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            continue

        if el.name == "h1":
            lines.append(f"# {text}")
        elif el.name == "h2":
            lines.append(f"## {text}")
        elif el.name == "h3":
            lines.append(f"### {text}")
        elif el.name == "h4":
            lines.append(f"#### {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        elif el.name == "a":
            href = el.get("href", "")
            if href and not href.startswith("#") and not href.startswith("javascript:"):
                lines.append(f"[{text}]({href})")
        elif el.name in ("pre", "code"):
            lines.append(f"```\n{text}\n```")
        else:
            lines.append(text)

    # Deduplicate consecutive identical lines
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    result = "\n".join(deduped)
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n\n... [truncated, {len(result)} total chars]"
    return result


# ══════════════════════════════════════════════════════════════════
# TOOL: browser_navigate
# ══════════════════════════════════════════════════════════════════


@registry.register(
    name="browser_navigate",
    description="Navigate to a URL in a headless browser and return the page content as markdown text. Use this for JavaScript-rendered pages that simple HTTP fetching can't handle.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to."},
            "wait_for": {
                "type": "string",
                "description": "Optional CSS selector to wait for before extracting content.",
            },
        },
        "required": ["url"],
    },
    tier=0,
    source="playwright",
)
async def browser_navigate(url: str, wait_for: str = None) -> str:
    """Navigate to a URL and return page content as markdown."""
    try:
        page = await _ensure_browser()

        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        if wait_for:
            try:
                await page.wait_for_selector(wait_for, timeout=10000)
            except Exception:
                pass  # Continue even if selector not found

        # Small delay for JS rendering
        await asyncio.sleep(1)

        html = await page.content()
        title = await page.title()
        current_url = page.url

        markdown = _html_to_markdown(html)

        return json.dumps(
            {
                "status": "success",
                "title": title,
                "url": current_url,
                "content": markdown,
            }
        )

    except Exception as e:
        logger.error(f"[Browser] Navigate failed: {e}")
        return json.dumps({"status": "error", "error": str(e)})


# ══════════════════════════════════════════════════════════════════
# TOOL: run_playwright_script
# ══════════════════════════════════════════════════════════════════


@registry.register(
    name="run_playwright_script",
    description=(
        "Write and execute a custom Python Playwright script. "
        "The script runs in a fresh subprocess with its own browser instance. "
        "Use this for complex multi-step browser automations like: "
        "scraping paginated data, filling multi-step forms, extracting tables, "
        "or any task that requires custom logic beyond simple navigate/click/type. "
        "The script MUST print its output to stdout (use print()). "
        "Playwright is already installed — import from playwright.sync_api."
    ),
    parameters={
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": (
                    "A complete Python script using playwright.sync_api. "
                    "Must be self-contained. Example:\n"
                    "from playwright.sync_api import sync_playwright\n"
                    "with sync_playwright() as p:\n"
                    "    browser = p.chromium.launch(headless=True)\n"
                    "    page = browser.new_page()\n"
                    "    page.goto('https://example.com')\n"
                    "    print(page.title())\n"
                    "    browser.close()"
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max execution time in seconds (default: 60).",
            },
        },
        "required": ["script"],
    },
    tier=0,
    source="playwright",
)
async def run_playwright_script(script: str, timeout_seconds: int = 60) -> str:
    """Execute a custom Playwright script in a sandboxed subprocess."""
    try:
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        script_path = SCRIPTS_DIR / f"agent_script_{int(time.time())}.py"

        # Write script to disk for auditability
        script_path.write_text(script, encoding="utf-8")
        logger.info(f"[Browser] Executing custom script: {script_path}")

        python_exe = str(Path(sys.executable))

        process = await asyncio.create_subprocess_exec(
            python_exe,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(SCRIPTS_DIR),
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Script timed out after {timeout_seconds}s",
                    "script_path": str(script_path),
                }
            )

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        # Truncate huge outputs
        if len(stdout_text) > 10000:
            stdout_text = stdout_text[:10000] + "\n... [truncated]"
        if len(stderr_text) > 3000:
            stderr_text = stderr_text[:3000] + "\n... [truncated]"

        return json.dumps(
            {
                "status": "success" if process.returncode == 0 else "error",
                "exit_code": process.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "script_path": str(script_path),
            }
        )

    except Exception as e:
        logger.error(f"[Browser] Script execution failed: {e}")
        return json.dumps({"status": "error", "error": str(e)})

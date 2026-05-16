"""
MCP Server wrapper for the Playwright browser tools.

This exposes the same browser tools as an MCP-compatible server over stdio.
External MCP clients (Claude Desktop, etc.) can connect to this.

Usage:
    python -m app.tools.browser_mcp_server

Or via the system API:
    POST /api/v1/system/mcp/connect
    {"name": "browser", "command": "python -m app.tools.browser_mcp_server"}
"""

import asyncio
import json
import sys
import logging

logger = logging.getLogger(__name__)

# ── Tool definitions matching the registry tools ──

TOOLS = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in a headless browser and return the page content as markdown text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to."},
                "wait_for": {
                    "type": "string",
                    "description": "Optional CSS selector to wait for.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "browser_click",
        "description": "Click an element on the current page by CSS selector.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to click.",
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "browser_type",
        "description": "Type text into a form field identified by CSS selector.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input field.",
                },
                "text": {"type": "string", "description": "Text to type."},
                "press_enter": {
                    "type": "boolean",
                    "description": "Whether to press Enter after typing.",
                },
            },
            "required": ["selector", "text"],
        },
    },
    {
        "name": "browser_screenshot",
        "description": "Take a screenshot of the current browser page.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "A short name for the screenshot file.",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture the full scrollable page.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "browser_evaluate",
        "description": "Execute JavaScript in the current browser page and return the result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "run_playwright_script",
        "description": "Write and execute a custom Python Playwright script in a subprocess.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "A complete Python script using playwright.sync_api.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Max execution time in seconds.",
                },
            },
            "required": ["script"],
        },
    },
    {
        "name": "browser_close",
        "description": "Close the shared browser session to free resources.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]

# Map tool names to their handler functions (lazy import)
_handlers = None


def _get_handlers():
    global _handlers
    if _handlers is None:
        from app.tools.browser_tools import (
            browser_navigate,
            browser_click,
            browser_type,
            browser_screenshot,
            browser_evaluate,
            run_playwright_script,
            browser_close,
        )

        _handlers = {
            "browser_navigate": browser_navigate,
            "browser_click": browser_click,
            "browser_type": browser_type,
            "browser_screenshot": browser_screenshot,
            "browser_evaluate": browser_evaluate,
            "run_playwright_script": run_playwright_script,
            "browser_close": browser_close,
        }
    return _handlers


def _make_response(req_id, result):
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n"


def _make_error(req_id, code, message):
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": code, "message": message},
            }
        )
        + "\n"
    )


async def handle_request(method, params, req_id):
    """Handle a single JSON-RPC request."""

    if method == "initialize":
        return _make_response(
            req_id,
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "vllm-browser-tools",
                    "version": "1.0.0",
                },
            },
        )

    elif method == "notifications/initialized":
        return None  # No response for notifications

    elif method == "tools/list":
        return _make_response(req_id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        handlers = _get_handlers()
        if tool_name not in handlers:
            return _make_error(req_id, -32601, f"Tool '{tool_name}' not found")

        try:
            result_str = await handlers[tool_name](**arguments)
            return _make_response(
                req_id,
                {
                    "content": [{"type": "text", "text": result_str}],
                },
            )
        except Exception as e:
            return _make_response(
                req_id,
                {
                    "content": [
                        {"type": "text", "text": json.dumps({"error": str(e)})}
                    ],
                    "isError": True,
                },
            )

    else:
        return _make_error(req_id, -32601, f"Method '{method}' not found")


async def main():
    """Run the MCP server over stdio."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    # Use raw stdout for writing
    (
        writer_transport,
        writer_protocol,
    ) = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(
        writer_transport, writer_protocol, reader, asyncio.get_event_loop()
    )

    while True:
        line = await reader.readline()
        if not line:
            break

        line_str = line.decode().strip()
        if not line_str:
            continue

        try:
            msg = json.loads(line_str)
        except json.JSONDecodeError:
            continue

        req_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        response = await handle_request(method, params, req_id)

        if response:
            writer.write(response.encode())
            await writer.drain()


if __name__ == "__main__":
    # Set PYTHONPATH so the imports work
    if "d:\\Github\\vllm-trading-bot" not in sys.path:
        sys.path.insert(0, "d:\\Github\\vllm-trading-bot")

    asyncio.run(main())

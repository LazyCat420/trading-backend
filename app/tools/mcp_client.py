"""
MCP Client Adapter — connects to external MCP tool servers.

Supports stdio transport: launches the MCP server as a subprocess and
communicates via JSON-RPC 2.0 over stdin/stdout.

Usage:
    adapter = MCPClientAdapter("npx -y @example/mcp-server")
    await adapter.connect()
    # Tools are now in registry.schemas automatically
    await adapter.disconnect()
"""

import asyncio
import json
import logging
import uuid

from app.tools.registry import registry

logger = logging.getLogger(__name__)


# JSON-RPC 2.0 helpers
def _jsonrpc_request(
    method: str, params: dict | None = None, req_id: str | None = None
) -> str:
    """Build a JSON-RPC 2.0 request string (newline-delimited)."""
    msg = {
        "jsonrpc": "2.0",
        "method": method,
        "id": req_id or str(uuid.uuid4().hex[:8]),
    }
    if params is not None:
        msg["params"] = params
    return json.dumps(msg) + "\n"


class MCPClientAdapter:
    """Adapter that connects to a single MCP server over stdio transport.

    Lifecycle:
        1. connect()  — launches the subprocess, sends initialize, lists tools
        2. call_tool() — sends tools/call for a specific tool
        3. disconnect() — shuts down the subprocess cleanly
    """

    def __init__(self, command: str, env: dict | None = None, timeout: float = 30.0):
        """
        Args:
            command: Shell command to launch the MCP server (e.g. "npx -y @example/server")
            env: Optional environment variables for the subprocess
            timeout: Max seconds to wait for a response from the MCP server
        """
        self.command = command
        self.env = env
        self.timeout = timeout

        self._process: asyncio.subprocess.Process | None = None
        self._tools: dict[str, dict] = {}  # name -> MCP tool schema
        self._pending: dict[str, asyncio.Future] = {}  # id -> Future
        self._reader_task: asyncio.Task | None = None
        self._connected = False
        self._server_info: dict = {}

    async def connect(self) -> list[str]:
        """Launch the MCP server, initialize, and discover tools.

        Returns:
            List of tool names that were registered.
        """
        logger.info(f"[MCP] Launching server: {self.command}")

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )

        # Start background reader for stdout
        self._reader_task = asyncio.create_task(self._read_stdout())

        # Step 1: Initialize
        init_result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "vllm-trading-bot", "version": "1.0.0"},
            },
        )

        self._server_info = init_result.get("serverInfo", {})
        logger.info(
            f"[MCP] Server initialized: {self._server_info.get('name', 'unknown')} "
            f"v{self._server_info.get('version', '?')}"
        )

        # Step 2: Send initialized notification (no response expected)
        await self._send_notification("notifications/initialized", {})

        # Step 3: List tools
        tools_result = await self._send_request("tools/list", {})
        raw_tools = tools_result.get("tools", [])

        # Step 4: Register each MCP tool in our registry
        registered_names = []
        for tool in raw_tools:
            name = tool.get("name", "")
            if not name:
                continue

            # Prefix MCP tools to avoid name collisions with native tools
            prefixed_name = f"mcp_{name}"
            self._tools[prefixed_name] = tool

            # Build the async handler that delegates to the MCP server
            handler = self._make_tool_handler(prefixed_name, name)

            # Register in our tool registry
            registry.register(
                func=handler,
                name=prefixed_name,
                description=f"[MCP] {tool.get('description', 'No description')}",
                parameters=tool.get(
                    "inputSchema", {"type": "object", "properties": {}}
                ),
                tier=0,
                source=f"mcp:{self._server_info.get('name', 'unknown')}",
            )

            registered_names.append(prefixed_name)
            logger.info(f"[MCP] Registered tool: {prefixed_name}")

        self._connected = True
        logger.info(
            f"[MCP] Connected with {len(registered_names)} tools: {registered_names}"
        )
        return registered_names

    def _make_tool_handler(self, prefixed_name: str, original_name: str):
        """Create an async function that calls the MCP server's tool."""
        adapter = self  # capture reference

        async def _handler(**kwargs) -> str:
            return await adapter.call_tool(original_name, kwargs)

        _handler.__name__ = prefixed_name
        _handler.__doc__ = f"MCP tool: {original_name}"
        return _handler

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server and return the result as a string."""
        if not self._connected:
            return json.dumps({"error": "MCP server not connected"})

        try:
            result = await self._send_request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments,
                },
            )

            # MCP tools return content as an array of content blocks
            content_blocks = result.get("content", [])
            texts = []
            for block in content_blocks:
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    texts.append(f"[image: {block.get('mimeType', 'unknown')}]")
                else:
                    texts.append(json.dumps(block))

            return "\n".join(texts) if texts else json.dumps(result)

        except asyncio.TimeoutError:
            logger.error(f"[MCP] Tool call timed out: {tool_name}")
            return json.dumps({"error": f"MCP tool '{tool_name}' timed out"})
        except Exception as e:
            logger.error(f"[MCP] Tool call failed: {tool_name}: {e}")
            return json.dumps({"error": str(e)})

    async def disconnect(self):
        """Shut down the MCP server subprocess."""
        self._connected = False

        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()

        if self._process and self._process.returncode is None:
            try:
                self._process.stdin.close()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                self._process.kill()
                await self._process.wait()

        # Remove MCP tools from registry
        for name in list(self._tools.keys()):
            if name in registry.tools:
                del registry.tools[name]
            registry.schemas = [
                s for s in registry.schemas if s["function"]["name"] != name
            ]

        self._tools.clear()
        logger.info("[MCP] Disconnected and cleaned up tools")

    async def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        req_id = uuid.uuid4().hex[:8]
        msg = _jsonrpc_request(method, params, req_id)

        future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        self._process.stdin.write(msg.encode())
        await self._process.stdin.drain()

        try:
            result = await asyncio.wait_for(future, timeout=self.timeout)
            return result
        finally:
            self._pending.pop(req_id, None)

    async def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        msg = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            msg["params"] = params

        line = json.dumps(msg) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()

    async def _read_stdout(self):
        """Background task: read newline-delimited JSON-RPC messages from stdout."""
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[MCP] Non-JSON from server stdout: {line_str[:100]}"
                    )
                    continue

                # Check if this is a response (has "id" and "result" or "error")
                msg_id = msg.get("id")
                if msg_id and msg_id in self._pending:
                    future = self._pending[msg_id]
                    if "error" in msg:
                        err = msg["error"]
                        future.set_exception(
                            RuntimeError(
                                f"MCP error {err.get('code')}: {err.get('message')}"
                            )
                        )
                    else:
                        future.set_result(msg.get("result", {}))
                elif "method" in msg:
                    # Server-initiated request or notification — log for now
                    logger.debug(f"[MCP] Server notification: {msg.get('method')}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[MCP] stdout reader error: {e}")


# ── Global MCP adapter registry ──
_mcp_adapters: dict[str, MCPClientAdapter] = {}


async def connect_mcp_server(
    name: str, command: str, env: dict | None = None
) -> list[str]:
    """Connect to an MCP server and register its tools.

    Args:
        name: A unique identifier for this MCP server connection
        command: The shell command to launch the MCP server
        env: Optional environment variables

    Returns:
        List of registered tool names
    """
    if name in _mcp_adapters:
        logger.warning(
            f"[MCP] Server '{name}' already connected, disconnecting first..."
        )
        await disconnect_mcp_server(name)

    adapter = MCPClientAdapter(command, env=env)
    tools = await adapter.connect()
    _mcp_adapters[name] = adapter
    return tools


async def disconnect_mcp_server(name: str):
    """Disconnect an MCP server and remove its tools."""
    adapter = _mcp_adapters.pop(name, None)
    if adapter:
        await adapter.disconnect()


async def disconnect_all_mcp():
    """Disconnect all MCP servers."""
    for name in list(_mcp_adapters.keys()):
        await disconnect_mcp_server(name)

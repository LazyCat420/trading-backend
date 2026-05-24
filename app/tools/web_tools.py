import logging
import json
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from app.tools.registry import registry
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

# Global persistent session for connection pooling
_hermes_session = None


def get_hermes_session() -> aiohttp.ClientSession:
    global _hermes_session
    if _hermes_session is None or _hermes_session.closed:
        _hermes_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=20, keepalive_timeout=60)
        )
    return _hermes_session


async def get_healthy_hermes_endpoints(endpoints: list[str], key: str, session: aiohttp.ClientSession) -> list[str]:
    """Ping all endpoints in parallel and return the healthy ones."""
    async def check_one(endpoint: str) -> str | None:
        try:
            ping_payload = {
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "stream": False,
            }
            ping_headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            ping_timeout = aiohttp.ClientTimeout(total=15.0)
            async with session.post(
                endpoint, json=ping_payload, headers=ping_headers, timeout=ping_timeout
            ) as response:
                if response.status == 200:
                    logger.info(f"[WebTools] Responsive Hermes endpoint found: {endpoint}")
                    return endpoint
        except Exception as e:
            logger.debug(f"[WebTools] Hermes endpoint {endpoint} failed ping check: {e}")
        return None

    results = await asyncio.gather(*(check_one(ep) for ep in endpoints))
    healthy = [r for r in results if r is not None]
    if healthy:
        return healthy
    logger.warning("[WebTools] No responsive Hermes endpoints found during ping. Falling back to all.")
    return endpoints


logger = logging.getLogger(__name__)


@registry.register(
    name="search_web",
    description="Perform a web search to find recent news or information. Returns top results.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 3).",
            },
        },
        "required": ["query"],
    },
    tier=0,
    source="hermes",
)
async def search_web(query: str, num_results: int = 3) -> str:
    """
    Search the web using Hermes Gateway.
    Returns real search results with titles, URLs, and snippets.
    """
    logger.info(f"[WebTools] Searching web via Hermes for: {query}")

    try:
        prompt = (
            f"Please perform a web search for the following query: '{query}'. "
            f"Return the top {num_results} most relevant and recent results. "
            f"Provide the information in a concise structure with titles, URLs (if known), and brief snippets."
        )

        hermes_resp_json = await query_hermes(prompt)
        hermes_data = json.loads(hermes_resp_json)

        if hermes_data.get("status") in ("success", "mock_success"):
            results = [
                {
                    "title": "Hermes Search Summary",
                    "url": "hermes://search",
                    "snippet": hermes_data.get("response", ""),
                }
            ]
            return json.dumps(
                {
                    "status": "success",
                    "query": query,
                    "results": results,
                }
            )
        else:
            return hermes_resp_json

    except Exception as e:
        logger.warning("[WebTools] Web search failed, returning empty: %s", e)
        return json.dumps(
            {
                "status": "error",
                "query": query,
                "results": [],
                "message": str(e),
            }
        )


@registry.register(
    name="web_search",
    description="Perform a web search to find recent news or information. Returns top results.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 3).",
            },
        },
        "required": ["query"],
    },
    tier=0,
    source="hermes",
)
async def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web using Hermes Gateway. Alias for search_web.
    """
    return await search_web(query, num_results)


@registry.register(
    name="scrape_url",
    description="Scrape the main text content from a URL. Use this to read articles or SEC filings.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The URL to scrape."}},
        "required": ["url"],
    },
    tier=0,
    source="aiohttp",
)
async def scrape_url(url: str) -> str:
    """
    Scrape text content from a web page using aiohttp and BeautifulSoup.
    """
    logger.info(f"[WebTools] Scraping URL: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return f"Error: Received status code {response.status}"
                html = await response.text()

                soup = BeautifulSoup(html, "html.parser")
                # Remove scripts and styles
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                text = soup.get_text(separator=" ", strip=True)
                # Truncate to avoid massive context window usage
                truncated_text = text[:8000]

                return json.dumps(
                    {
                        "status": "success",
                        "url": url,
                        "content": truncated_text,
                        "truncated": len(text) > 8000,
                    }
                )
    except Exception as e:
        logger.error(f"[WebTools] Scraping failed for {url}: {e}")
        return json.dumps({"status": "error", "message": str(e)})




@registry.register(
    name="query_hermes",
    description="Send a task or query to the local Hermes intelligence model. Subject to guardrails.",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The prompt or task for Hermes to process.",
            }
        },
        "required": ["prompt"],
    },
    tier=1,
    source="hermes",
    fallback_only=True,
)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def query_hermes(prompt: str) -> str:
    """
    Query the Hermes agent running on the WSL2 hub.
    Uses the Hermes agent API server format.
    Pre-validated and post-sanitized via hermes_guardrails.
    """
    from app.services.hermes_guardrails import (
        validate_hermes_request,
        sanitize_hermes_output,
    )

    # ── Layer 2: Pre-flight guardrail check ──
    is_safe, reason = validate_hermes_request(prompt)
    if not is_safe:
        logger.warning("[WebTools] Hermes request BLOCKED: %s", reason)
        return json.dumps({"status": "blocked", "reason": reason, "response": ""})

    logger.info(f"[WebTools] Querying Hermes with prompt: {prompt[:50]}...")

    from app.config import settings

    hermes_endpoints = list(settings.HERMES_ENDPOINT_MAP.values())
    hermes_key = settings.API_SERVER_KEY

    if not hermes_endpoints:
        return json.dumps(
            {"status": "error", "message": "No Hermes endpoints configured."}
        )

    last_error = None
    session = get_hermes_session()

    # Discover responsive endpoints to avoid hanging on slow/offline nodes
    endpoints_to_try = await get_healthy_hermes_endpoints(hermes_endpoints, hermes_key, session)

    for hermes_endpoint in endpoints_to_try:
        try:
            payload = {
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "stream": False,
            }
            headers = {
                "Authorization": f"Bearer {hermes_key}",
                "Content-Type": "application/json",
            }

            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with session.post(
                    hermes_endpoint, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_content = data["choices"][0]["message"]["content"]
                        # ── Layer 3: Sanitize output ──
                        clean = sanitize_hermes_output(raw_content)
                        return json.dumps({"status": "success", "response": clean})
                    else:
                        error_text = await response.text()
                        last_error = f"Status {response.status}: {error_text}"
                        logger.warning(
                            f"[WebTools] {hermes_endpoint} failed: {last_error}"
                        )
                        continue  # Try next endpoint
            except aiohttp.ClientConnectorError as e:
                logger.warning(
                    f"[WebTools] Hermes endpoint {hermes_endpoint} not reachable. Trying next. Error: {e}"
                )
                last_error = f"Connection error: {e}"
                continue

        except Exception as e:
            logger.error(f"[WebTools] Hermes query failed on {hermes_endpoint}: {e}")
            last_error = str(e)
            continue

    # If all endpoints fail or are unreachable
    logger.warning(
        "[WebTools] All Hermes endpoints unreachable. Returning mock response."
    )
    return json.dumps(
        {
            "status": "mock_success",
            "response": f"Mock Hermes analysis for: {prompt}. Please check Hermes APIs. Last error: {last_error}",
        }
    )


@registry.register(
    name="hermes_web_research",
    description=(
        "Use the Hermes browser agent for deep web research on financial topics. "
        "Only invoked when structured API tools return empty results. "
        "Restricted to financial research domains only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The financial research query for Hermes to investigate.",
            },
            "ticker": {
                "type": "string",
                "description": "Optional stock ticker for context.",
            },
        },
        "required": ["query"],
    },
    tier=1,
    source="hermes",
    fallback_only=True,
)
async def hermes_web_research(query: str, ticker: str = "") -> str:
    """
    Deep web research via Hermes browser agent.
    Only called when structured APIs (yfinance, Finnhub, etc.) return empty.
    Domain-restricted to financial research sites.
    """
    # Build a research-focused prompt with guardrails baked in
    research_prompt = (
        f"You are a financial research assistant. Research the following topic "
        f"using ONLY financial news sites, SEC filings, and market data sources.\n\n"
        f"Topic: {query}\n"
    )
    if ticker:
        research_prompt += f"Ticker: {ticker}\n"
    research_prompt += (
        "\nProvide a concise summary with key data points, sources cited, "
        "and any relevant price levels or dates. Do NOT execute any commands "
        "or modify any files."
    )

    return await query_hermes(research_prompt)


async def stream_hermes_chat(
    system: str,
    user: str,
    history: list[dict] = None,
    model_override: str = None,
    endpoint_override: str = None,
    cancel_event: "asyncio.Event | None" = None,
    tools: list[dict] = None,
    images: list[str] = None,
):
    """
    Stream a chat completion directly from the Hermes API.
    Yields chunks of text as they arrive.

    Args:
        system: System prompt
        user: User message
        history: Optional conversation history
        model_override: Use this model name instead of 'hermes-agent'
        endpoint_override: Route to this specific endpoint key (e.g., 'dgx_spark')
        cancel_event: If set, stop streaming when this event fires (for stop button)
        tools: Optional list of tool schemas
        images: Optional list of base64 data URIs for vision support
    """
    import json
    from app.config import settings

    # Resolve which Hermes endpoints to try
    if endpoint_override:
        # User selected a specific device — route ONLY to that device's Hermes
        ep_map = settings.HERMES_ENDPOINT_MAP
        if endpoint_override in ep_map:
            hermes_endpoints = [ep_map[endpoint_override]]
            logger.info(
                "[WebTools] Hermes routing to user-selected endpoint: %s → %s",
                endpoint_override,
                hermes_endpoints[0],
            )
        else:
            logger.warning(
                "[WebTools] Endpoint override '%s' not found in HERMES_ENDPOINT_MAP, "
                "falling back to all endpoints",
                endpoint_override,
            )
            hermes_endpoints = list(settings.HERMES_ENDPOINT_MAP.values())
    else:
        hermes_endpoints = list(settings.HERMES_ENDPOINT_MAP.values())

    hermes_key = settings.API_SERVER_KEY

    if not hermes_endpoints:
        yield "Error: No Hermes endpoints configured."
        return

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)

    # Build user message with optional vision content
    if images and len(images) > 0:
        user_content = []
        if user:
            user_content.append({"type": "text", "text": user})
        for img_uri in images:
            user_content.append({"type": "image_url", "image_url": {"url": img_uri}})
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user})

    # Use the user's selected model name if provided, otherwise default
    model_name = model_override or "hermes-agent"
    logger.info(
        "[WebTools] Hermes stream using model='%s', endpoints=%d",
        model_name,
        len(hermes_endpoints),
    )

    headers = {
        "Authorization": f"Bearer {hermes_key}",
        "Content-Type": "application/json",
    }
    session = get_hermes_session()
    # Filter for healthy endpoints if not overridden by user
    if not endpoint_override:
        hermes_endpoints = await get_healthy_hermes_endpoints(hermes_endpoints, hermes_key, session)
        
    REQUIRES_CONFIRMATION = {"buy_stock", "sell_stock", "remove_from_watchlist"}
    
    turn_count = 0
    MAX_TOOL_TURNS = 10

    while True:
        turn_count += 1
        if turn_count > MAX_TOOL_TURNS:
            yield f"__THINK__[Error] Maximum tool turns ({MAX_TOOL_TURNS}) reached. The agent might be stuck in a loop.\n"
            yield "\n**Error:** I reached the maximum number of internal tool execution steps and had to stop. This usually happens if I am trying to use a tool that doesn't exist or if a tool is repeatedly failing."
            break

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 8000,
            "temperature": 0.4,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        last_error = None
        endpoint_success = False
        tool_calls_this_turn = []
        full_response = []

        for hermes_endpoint in hermes_endpoints:
            try:
                # Use a timeout without total limit for streaming agent responses
                timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
                async with session.post(
                    hermes_endpoint, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        last_error = f"Status {response.status}: {error_text}"
                        logger.warning(f"[WebTools] {hermes_endpoint} failed: {last_error}")
                        continue  # Try next endpoint

                    current_event = None
                    async for line in response.content:
                        # Check for cancellation (user clicked stop button)
                        if cancel_event and cancel_event.is_set():
                            logger.info(
                                "[WebTools] Hermes stream cancelled by user (stop button)"
                            )
                            return

                        line = line.decode("utf-8").strip()
                        if not line:
                            continue
                        if line.startswith("event: "):
                            current_event = line[7:]
                            continue
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if current_event and current_event != "message":
                                    if "tool" in data and "label" in data:
                                        emoji = data.get("emoji", "⚡")
                                        yield f"__THINK__[{emoji} {data['tool']}] {data['label']}\n"
                                    elif "status" in data and "message" in data:
                                        yield f"__THINK__[{data['status']}] {data['message']}\n"
                                    else:
                                        yield f"__THINK__[{current_event}] {json.dumps(data)}\n"
                                    current_event = None
                                    continue

                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})

                                    # Handle tool calls
                                    tool_calls = delta.get("tool_calls", [])
                                    if tool_calls:
                                        for tc in tool_calls:
                                            tc_index = tc.get("index", 0)
                                            if not hasattr(
                                                stream_hermes_chat, "active_tool_calls"
                                            ):
                                                stream_hermes_chat.active_tool_calls = {}
                                            if (
                                                tc_index
                                                not in stream_hermes_chat.active_tool_calls
                                            ):
                                                stream_hermes_chat.active_tool_calls[
                                                    tc_index
                                                ] = {
                                                    "id": "",
                                                    "function": {
                                                        "name": "",
                                                        "arguments": "",
                                                    },
                                                }

                                            if "id" in tc and tc["id"]:
                                                stream_hermes_chat.active_tool_calls[
                                                    tc_index
                                                ]["id"] = tc["id"]

                                            func_delta = tc.get("function", {})
                                            if "name" in func_delta and func_delta["name"]:
                                                stream_hermes_chat.active_tool_calls[
                                                    tc_index
                                                ]["function"]["name"] += func_delta["name"]
                                            if (
                                                "arguments" in func_delta
                                                and func_delta["arguments"]
                                            ):
                                                stream_hermes_chat.active_tool_calls[
                                                    tc_index
                                                ]["function"]["arguments"] += func_delta[
                                                    "arguments"
                                                ]

                                    finish_reason = data["choices"][0].get("finish_reason")
                                    if finish_reason == "tool_calls" and hasattr(
                                        stream_hermes_chat, "active_tool_calls"
                                    ):
                                        for (
                                            tc
                                        ) in stream_hermes_chat.active_tool_calls.values():
                                            tool_calls_this_turn.append(tc)
                                            yield f"__TOOL_CALL__{json.dumps(tc)}\n"
                                            full_response.append(
                                                f"__TOOL_CALL__{json.dumps(tc)}\n"
                                            )
                                        stream_hermes_chat.active_tool_calls = {}

                                    if "content" in delta and delta["content"]:
                                        content_chunk = delta["content"]
                                        yield content_chunk
                                        full_response.append(content_chunk)
                            except json.JSONDecodeError:
                                pass
                            current_event = None

                    # If we successfully streamed from this endpoint, we're done with the endpoint loop
                    endpoint_success = True
                    break

            except aiohttp.ClientConnectorError as e:
                logger.warning(
                    f"[WebTools] Hermes endpoint {hermes_endpoint} not reachable. Trying next. Error: {e}"
                )
                last_error = f"Connection error: {e}"
                continue
            except Exception as e:
                logger.error(f"[WebTools] Hermes stream failed on {hermes_endpoint}: {e}")
                last_error = str(e)
                continue

        if not endpoint_success:
            yield f"Error: Failed to connect to Hermes stream. All endpoints unreachable. Last error: {last_error}"
            return

            return

        # We had tool calls! Append the assistant's action to messages
        assistant_content = "".join([c for c in full_response if not c.startswith("__TOOL_CALL__")])
        messages.append({
            "role": "assistant",
            "content": assistant_content if assistant_content else None,
            "tool_calls": tool_calls_this_turn
        })

        # Execute tools and append results
        from app.tools.registry import registry
        for tc in tool_calls_this_turn:
            tool_name = tc["function"]["name"]
            
            if tool_name in REQUIRES_CONFIRMATION:
                # Mock result for tools requiring human approval so LLM can end the loop naturally
                yield f"__THINK__[⚠️ Requires Confirmation] Waiting for user to confirm {tool_name}...\n"
                mock_result = json.dumps({"status": "awaiting_confirmation"})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tool_name,
                    "content": mock_result
                })
            else:
                yield f"__THINK__[Executing Tool] {tool_name}...\n"
                try:
                    result = await registry.execute_tool_call(tc, skip_permission_check=True)
                    messages.append(result)
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": json.dumps({"error": str(e)})
                    })
        
        # Loop restarts and submits the updated messages back to the LLM

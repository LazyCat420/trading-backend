"""
Hermes Guardrails — Defense-in-depth for browser agent safety.

4-layer protection:
  Layer 1: Hermes config.yaml (disable terminal/file tools at source)
  Layer 2: Domain allowlist (network-level, enforced HERE before requests)
  Layer 3: Output sanitization (strip dangerous patterns from responses)
  Layer 4: SOUL.md system prompt (constraint on Hermes behavior)

This module implements Layers 2 and 3. Layers 1 and 4 are config files
deployed to the Hermes instance on WSL2.
"""

import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Layer 2: Domain Allowlist ──────────────────────────────────────────
# Only these domains can be navigated to / scraped by Hermes.
# Add new domains here as needed. Subdomains are matched automatically.

ALLOWED_DOMAINS: set[str] = {
    # Financial data & research
    "finance.yahoo.com",
    "tradingview.com",
    "stockanalysis.com",
    "finviz.com",
    "capitalflowsresearch.com",
    "macrotrends.net",
    "simplywall.st",
    "gurufocus.com",
    "wisesheets.io",
    "tipranks.com",
    "zacks.com",
    # SEC / regulatory
    "sec.gov",
    "edgar.sec.gov",
    "efts.sec.gov",
    # News
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "wsj.com",
    "marketwatch.com",
    "seekingalpha.com",
    "investing.com",
    "barrons.com",
    "ft.com",
    "thestreet.com",
    # Social sentiment
    "reddit.com",
    "old.reddit.com",
    "twitter.com",
    "x.com",
    # Search engines (Hermes may use these to find articles)
    "google.com",
    "duckduckgo.com",
    "bing.com",
    # Macro / economic
    "fred.stlouisfed.org",
    "tradingeconomics.com",
    "bls.gov",
    "bea.gov",
    "imf.org",
}

# ── Blocked action patterns ───────────────────────────────────────────
# If any of these appear in a Hermes prompt/output, we block or strip it.

BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    # Destructive shell commands
    re.compile(r"\b(?:rm|del|rmdir|format|mkfs|fdisk)\s+", re.IGNORECASE),
    re.compile(r"\bsudo\s+", re.IGNORECASE),
    re.compile(r"\b(?:chmod|chown|chgrp)\s+", re.IGNORECASE),
    # Database destructive ops
    re.compile(r"\b(?:DROP\s+TABLE|DELETE\s+FROM|TRUNCATE)\b", re.IGNORECASE),
    # HTTP destructive methods
    re.compile(r"\bcurl\s+.*-X\s*DELETE\b", re.IGNORECASE),
    re.compile(r"\bwget\s+.*--delete\b", re.IGNORECASE),
    # Package/system modification
    re.compile(
        r"\b(?:pip|npm|apt|yum|brew)\s+(?:install|uninstall|remove)\b", re.IGNORECASE
    ),
    # Process control
    re.compile(r"\b(?:kill|killall|pkill)\s+", re.IGNORECASE),
]


def is_domain_allowed(url: str) -> bool:
    """Check if a URL's domain is in the financial research allowlist.

    Handles subdomains: 'news.google.com' matches 'google.com'.
    Returns True for relative URLs or empty strings (safe by default).
    """
    if not url or not url.startswith(("http://", "https://")):
        return True  # Relative URL or empty — safe

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
    except Exception:
        logger.warning("[Guardrails] Failed to parse URL: %s", url[:100])
        return False

    # Check exact match or subdomain match
    hostname_lower = hostname.lower()
    for domain in ALLOWED_DOMAINS:
        if hostname_lower == domain or hostname_lower.endswith(f".{domain}"):
            return True

    logger.warning("[Guardrails] BLOCKED domain: %s (from %s)", hostname, url[:100])
    return False


def validate_hermes_request(prompt: str) -> tuple[bool, str]:
    """Pre-flight check on a prompt before sending to Hermes.

    Returns:
        (is_safe, reason) — if not safe, reason explains why.
    """
    if not prompt or not prompt.strip():
        return False, "Empty prompt"

    # Check for blocked action patterns
    for pattern in BLOCKED_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return False, f"Blocked pattern detected: '{match.group()}'"

    # Check for any URLs in the prompt that aren't allowed
    url_pattern = re.compile(r"https?://\S+")
    urls = url_pattern.findall(prompt)
    for url in urls:
        if not is_domain_allowed(url):
            return False, f"URL not in allowlist: {url}"

    return True, "OK"


def sanitize_hermes_output(raw: str, max_chars: int = 8000) -> str:
    """Clean Hermes output before injecting into pipeline context.

    Strips:
        - Code blocks (```...```)
        - Command-like patterns (sudo, rm, chmod, etc.)
        - URLs not in the domain allowlist
        - Excessive length
    """
    if not raw:
        return ""

    text = raw

    # 1. Truncate to prevent context overflow
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
        logger.info(
            "[Guardrails] Truncated Hermes output from %d to %d chars",
            len(raw),
            max_chars,
        )

    # 2. Strip embedded code blocks
    text = re.sub(r"```[\s\S]*?```", "[code block removed]", text)

    # 3. Strip blocked patterns
    for pattern in BLOCKED_PATTERNS:
        text = pattern.sub("[command removed]", text)

    # 4. Replace URLs not in allowlist with placeholder
    def _filter_url(match: re.Match[str]) -> str:
        url = match.group()
        if is_domain_allowed(url):
            return url
        return "[blocked-url]"

    text = re.sub(r"https?://\S+", _filter_url, text)

    return text.strip()


def filter_urls_in_task(task_data: dict) -> dict:
    """Filter any URLs in a task payload to only allowed domains.

    Used by the worker before handing tasks to Hermes.
    """
    filtered = dict(task_data)

    # Check common URL fields
    for key in ("url", "target_url", "scrape_url", "source_url"):
        if key in filtered and filtered[key]:
            if not is_domain_allowed(filtered[key]):
                logger.warning(
                    "[Guardrails] Removed blocked URL from task.%s: %s",
                    key,
                    filtered[key][:100],
                )
                filtered[key] = ""

    # Check lists of URLs
    for key in ("urls", "sources"):
        if key in filtered and isinstance(filtered[key], list):
            filtered[key] = [u for u in filtered[key] if is_domain_allowed(u)]

    return filtered

"""
Playwright-based YouTube transcript scraper.

Adapted from LazyCat420/Youtube-News-Extracter patterns:
- Primary: Network interception (catch /get_transcript & /timedtext responses)
- Fallback: DOM scraping (transcript-segment-view-model elements)
- Stealth: Automation flag removal, mouse jitter, scroll priming

CSS selectors derived from YouTube DOM (March 2026):
  - Transcript segments: transcript-segment-view-model
  - Text spans: span.yt-core-attributed-string[role='text']
"""

import logging

logger = logging.getLogger(__name__)


import asyncio
import json
import re
import time
import random


def _find_segments_recursive(obj: dict | list) -> list[dict]:
    """Recursively search JSON for transcriptSegmentRenderer objects.

    YouTube's internal API returns deeply nested JSON with varying schemas.
    Instead of hardcoded paths, recursively scan for the data we need.
    """
    results = []
    if not obj or not isinstance(obj, (dict, list)):
        return results

    if isinstance(obj, dict):
        if "transcriptSegmentRenderer" in obj:
            results.append(obj["transcriptSegmentRenderer"])
        for key in obj:
            results.extend(_find_segments_recursive(obj[key]))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_find_segments_recursive(item))

    return results


def _segments_to_text(segments: list[dict]) -> str:
    """Extract text from transcriptSegmentRenderer segments."""
    parts = []
    for seg in segments:
        snippet = seg.get("snippet", {})
        runs = snippet.get("runs", [])
        text = "".join(r.get("text", "") for r in runs)
        if text.strip():
            parts.append(text.strip())
    return " ".join(parts)


def scrape_transcript_sync(
    video_id: str, headless: bool = True, timeout_ms: int = 25000
) -> str | None:
    """Scrape YouTube transcript using Playwright with network interception.

    Strategy (from Youtube-News-Extracter):
    1. Set up network listener for /get_transcript and /timedtext responses
    2. Click "Show transcript" to trigger the API call
    3. Parse transcript from intercepted network response (most reliable)
    4. Fallback: scrape transcript segments from DOM

    Returns plain text transcript or None on failure.
    """
    try:
        import sys
        import asyncio

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.info("[youtube-pw] playwright not installed")
        return None

    url = f"https://www.youtube.com/watch?v={video_id}"
    transcript_text = None
    network_transcript = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                viewport={
                    "width": 1280 + random.randint(0, 100),
                    "height": 900 + random.randint(0, 50),
                },
                locale="en-US",
            )
            page = context.new_page()

            # ── Network Interception ──
            # Capture transcript data from YouTube's internal API responses
            captured_transcripts = []

            def on_response(response):
                nonlocal captured_transcripts
                resp_url = response.url
                if "/get_transcript" in resp_url or "/timedtext" in resp_url:
                    try:
                        body = response.text()
                        # Try JSON parse (get_transcript endpoint)
                        try:
                            data = json.loads(body)
                            segments = _find_segments_recursive(data)
                            if segments:
                                text = _segments_to_text(segments)
                                if len(text) > 100:
                                    captured_transcripts.append(text)
                                    logger.info(
                                        f"[youtube-pw] Network intercept: {len(segments)} segments, {len(text)} chars"
                                    )
                        except json.JSONDecodeError:
                            # timedtext endpoint returns XML/other formats
                            # Extract text between XML tags if present
                            texts = re.findall(r">([^<]+)<", body)
                            text = " ".join(
                                t.strip()
                                for t in texts
                                if t.strip() and len(t.strip()) > 1
                            )
                            if len(text) > 100:
                                captured_transcripts.append(text)
                                logger.info(
                                    f"[youtube-pw] Timedtext intercept: {len(text)} chars"
                                )
                    except Exception:
                        pass

            page.on("response", on_response)

            # ── Navigate ──
            logger.info(f"[youtube-pw] Loading {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            # Dismiss cookie consent
            try:
                consent = page.locator("button:has-text('Accept all')")
                if consent.count() > 0:
                    consent.first.click(timeout=3000)
                    page.wait_for_timeout(1000)
            except Exception:
                pass

            # ── Stealth: Human-like behavior ──
            # Mouse jitter
            page.mouse.move(random.randint(200, 800), random.randint(200, 600))
            page.wait_for_timeout(500 + random.randint(0, 500))
            # Scroll to trigger lazy loading
            page.evaluate("window.scrollBy(0, 300)")
            page.wait_for_timeout(1000 + random.randint(0, 1000))

            # ── Find and click "Show transcript" ──
            transcript_opened = False

            # Step 1: Expand description ("...more" button)
            expand_selectors = [
                "#expand",
                "tp-yt-paper-button#expand",
                "[aria-label='Show more']",
            ]
            for sel in expand_selectors:
                try:
                    btn = page.locator(sel)
                    if btn.count() > 0:
                        btn.first.click(timeout=3000)
                        page.wait_for_timeout(1500)
                        break
                except Exception:
                    continue

            # Verify expansion
            try:
                is_expanded = page.evaluate("""
                    () => {
                        const expander = document.querySelector('#description-inline-expander');
                        return expander && expander.hasAttribute('is-expanded');
                    }
                """)
            except Exception:
                is_expanded = False

            # Step 2: Click "Show transcript" button
            transcript_selectors = [
                "button:has-text('Show transcript')",
                "[aria-label='Show transcript']",
            ]
            for sel in transcript_selectors:
                try:
                    btn = page.locator(sel)
                    if btn.count() > 0:
                        # Hover first (anti-bot: simulate human)
                        btn.first.hover(timeout=2000)
                        page.wait_for_timeout(500 + random.randint(0, 500))
                        btn.first.click(timeout=3000)
                        transcript_opened = True
                        break
                except Exception:
                    continue

            # Step 3: If description expand didn't work, try three-dot menu
            if not transcript_opened:
                try:
                    menu_btn = page.locator("button[aria-label='More actions']").first
                    menu_btn.click(timeout=3000)
                    page.wait_for_timeout(1000)
                    transcript_menu = page.locator(
                        "ytd-menu-service-item-renderer:has-text('Show transcript')"
                    )
                    if transcript_menu.count() > 0:
                        transcript_menu.first.click(timeout=3000)
                        transcript_opened = True
                except Exception:
                    pass

            if not transcript_opened:
                logger.info(
                    f"[youtube-pw] Could not find 'Show transcript' for {video_id}"
                )
                browser.close()
                return None

            # ── Wait for network interception or DOM rendering ──
            # Wait up to 10s for transcript data to arrive via network
            wait_start = time.time()
            while time.time() - wait_start < 10:
                if captured_transcripts:
                    break
                page.wait_for_timeout(500)

            # ── Extract results ──
            # Priority 1: Network-intercepted transcript (most reliable)
            if captured_transcripts:
                # Use longest transcript (filter out ad transcripts)
                network_transcript = max(captured_transcripts, key=len)
                logger.info(
                    f"[youtube-pw] Network transcript OK: {len(network_transcript)} chars"
                )
                transcript_text = network_transcript
            else:
                # Priority 2: DOM scraping (fallback)
                logger.info(
                    "[youtube-pw] No network data, falling back to DOM scraping"
                )
                try:
                    page.wait_for_selector(
                        "transcript-segment-view-model, ytd-transcript-segment-renderer",
                        timeout=8000,
                    )

                    # Modern YouTube (2024-2026)
                    segments = page.locator(
                        "transcript-segment-view-model span.yt-core-attributed-string[role='text']"
                    )
                    count = segments.count()

                    if count == 0:
                        # Older YouTube DOM
                        segments = page.locator(
                            "ytd-transcript-segment-renderer .segment-text"
                        )
                        count = segments.count()

                    if count > 0:
                        parts = []
                        for i in range(count):
                            text = segments.nth(i).inner_text().strip()
                            if text:
                                parts.append(text)
                        transcript_text = " ".join(parts)
                        logger.info(
                            f"[youtube-pw] DOM scrape OK: {count} segments, {len(transcript_text)} chars"
                        )
                except Exception as e:
                    logger.info(f"[youtube-pw] DOM scrape failed: {e}")

            browser.close()

    except Exception as e:
        logger.info(f"[youtube-pw] Error scraping {video_id}: {e}")
        return None

    if transcript_text and len(transcript_text) > 50:
        return transcript_text
    return None


async def scrape_transcript_async(
    video_id: str, headless: bool = True, timeout_ms: int = 25000
) -> str | None:
    """Async wrapper around sync Playwright scraper."""
    return await asyncio.to_thread(
        scrape_transcript_sync, video_id, headless, timeout_ms
    )

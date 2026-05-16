"""
Vision Scraper — Playwright overlay removal + Qwen3.5-35B screenshot OCR.

Last resort for article extraction when text scraping fails.
Uses multimodal LLM to read article content from screenshots.

Strategy:
  1. Load page with Playwright stealth mode
  2. Remove paywall overlays / newsletter modals via JS
  3. Scroll full page, capture viewport screenshots
  4. Send screenshots to Qwen3.5-35B for OCR
  5. Return extracted text

Works for:
  ✅ Soft paywalls (overlay-based, content IS in DOM)
  ✅ Anti-bot pages (after challenge passes)
  ❌ Hard paywalls (content not sent by server — nothing to reveal)
"""

import logging

logger = logging.getLogger(__name__)


import base64
import os

# Common paywall/overlay selectors to remove
OVERLAY_SELECTORS = [
    # Generic paywalls
    '[class*="paywall"]',
    '[class*="Paywall"]',
    '[class*="subscribe-wall"]',
    '[class*="gate"]',
    '[id*="paywall"]',
    '[id*="Paywall"]',
    "[data-paywall]",
    '[data-testid*="paywall"]',
    # Newsletter/signup modals
    '[class*="modal-overlay"]',
    '[class*="Modal"]',
    '[class*="newsletter"]',
    '[class*="signup"]',
    '[class*="popup"]',
    '[class*="Popup"]',
    # Cookie/consent banners
    '[class*="consent"]',
    '[class*="cookie-banner"]',
    '[class*="CookieBanner"]',
    # Sticky elements that cover content
    '[class*="sticky-header"]',
    '[class*="StickyHeader"]',
]

# CSS to inject after overlay removal
CLEANUP_CSS = """
    body { overflow: auto !important; position: static !important; }
    html { overflow: auto !important; }
    .paywall, .subscribe-wall, .modal-overlay { display: none !important; }
"""

# JS to remove overlays and restore scrolling
CLEANUP_JS = """
() => {
    // Remove known overlay elements
    const selectors = %SELECTORS%;
    for (const sel of selectors) {
        document.querySelectorAll(sel).forEach(el => el.remove());
    }

    // Remove all position:fixed elements (likely overlays/banners)
    const allFixed = document.querySelectorAll('*');
    for (const el of allFixed) {
        const style = window.getComputedStyle(el);
        if (style.position === 'fixed' && el.offsetHeight > 100) {
            // Only remove if it covers significant screen area
            if (el.offsetWidth > window.innerWidth * 0.5) {
                el.remove();
            }
        }
    }

    // Restore scrolling
    document.body.style.overflow = 'auto';
    document.body.style.position = 'static';
    document.documentElement.style.overflow = 'auto';

    // Remove blur/filter from article content
    document.querySelectorAll('[style*="blur"]').forEach(el => {
        el.style.filter = 'none';
    });
    document.querySelectorAll('[style*="opacity: 0"]').forEach(el => {
        el.style.opacity = '1';
    });

    return document.body.scrollHeight;
}
""".replace("%SELECTORS%", str(OVERLAY_SELECTORS))


async def extract_article_vision(
    url: str,
    max_screenshots: int = 5,
    viewport_width: int = 1280,
    viewport_height: int = 900,
    save_dir: str | None = None,
) -> str | None:
    """Extract article text by screenshotting and sending to Qwen3.5-35B.

    Returns extracted text or None.
    """
    from playwright.async_api import async_playwright

    screenshots = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )
            context = await browser.new_context(
                viewport={
                    "width": viewport_width,
                    "height": viewport_height,
                },
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            )
            page = await context.new_page()

            # Load page — use domcontentloaded, NOT networkidle
            # (ad-heavy sites like Yahoo/Investing never go network-idle)
            logger.info(f"[vision] Loading {url[:80]}...")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            except Exception as nav_err:
                # Even if navigation times out, page may have partial content
                logger.info(f"[vision] Navigation warning: {nav_err}")
            await page.wait_for_timeout(5000)  # Let JS render content

            # Remove overlays and get page height
            page_height = await page.evaluate(CLEANUP_JS)
            await page.wait_for_timeout(1000)

            # Inject cleanup CSS
            await page.add_style_tag(content=CLEANUP_CSS)

            # Calculate scroll positions
            num_screenshots = min(
                max_screenshots,
                max(1, (page_height // viewport_height) + 1),
            )
            logger.info(
                f"[vision] Page height: {page_height}px, "
                f"taking {num_screenshots} screenshots"
            )

            # Capture screenshots at each scroll position
            for i in range(num_screenshots):
                scroll_y = i * viewport_height
                await page.evaluate(f"window.scrollTo(0, {scroll_y})")
                await page.wait_for_timeout(500)

                screenshot_bytes = await page.screenshot(
                    type="png",
                    full_page=False,  # Just the viewport
                )
                screenshots.append(screenshot_bytes)

                # Save to disk if requested
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    fpath = os.path.join(save_dir, f"vision_{i + 1}.png")
                    with open(fpath, "wb") as fh:
                        fh.write(screenshot_bytes)
                    logger.info(
                        f"[vision] Screenshot {i + 1}/{num_screenshots} "
                        f"saved: {fpath} ({len(screenshot_bytes)} bytes)"
                    )
                else:
                    logger.info(
                        f"[vision] Screenshot {i + 1}/{num_screenshots} "
                        f"(scroll={scroll_y}px, {len(screenshot_bytes)} bytes)"
                    )

            await browser.close()

    except Exception as e:
        logger.info(f"[vision] Playwright error: {e}")
        return None

    if not screenshots:
        logger.info("[vision] No screenshots captured")
        return None

    # Send screenshots to Qwen3.5-35B
    return await _ocr_screenshots(screenshots, url)


async def _ocr_screenshots(
    screenshots: list[bytes],
    url: str,
) -> str | None:
    """Send screenshots to Qwen3.5-35B for OCR."""
    import httpx
    from app.config import settings

    # Build multimodal content array
    content = []
    for i, img_bytes in enumerate(screenshots):
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                "These are screenshots of a news article. "
                "Read ALL text visible in the images and return the "
                "complete article text. Combine text from all images "
                "into one continuous article. Return ONLY the article "
                "text, no commentary or descriptions of the images."
            ),
        }
    )

    payload = {
        "model": settings.ACTIVE_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            total_b64 = sum(len(base64.b64encode(s)) for s in screenshots)
            logger.info(
                f"[vision] Sending {len(screenshots)} screenshots to "
                f"Qwen3.5-35B ({total_b64:,} base64 chars)..."
            )
            r = await client.post(
                f"{settings.JETSON_VLLM_URL}/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            logger.info(f"[vision] OCR complete: {len(text)} chars, {tokens} tokens")
            return text if len(text) > 100 else None

    except Exception as e:
        logger.info(f"[vision] OCR error: {e}")
        return None


async def vision_deep_read(url: str) -> str | None:
    """Public API: Extract article text via vision pipeline.

    Use as last-resort fallback in deep_read_article().
    """
    logger.info(f"[vision] Starting vision extraction for {url[:80]}...")
    text = await extract_article_vision(url)
    if text and len(text) > 100:
        logger.info(f"[vision] Success: {len(text)} chars extracted")
        return text
    logger.info("[vision] Failed or too short")
    return None

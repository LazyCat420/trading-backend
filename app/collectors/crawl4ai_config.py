"""
crawl4ai Configuration — Full-featured v0.8.5 scraper config for the trading bot.

Centralizes all crawl4ai settings so every scraper uses the same
stealth + caching + content extraction configuration.

Features enabled (v0.8.5):
  - Native stealth (enable_stealth — replaces deprecated simulate_user/magic)
  - Ad blocking (avoid_ads — cleaner article text)
  - Consent popup removal (remove_consent_popups — auto-dismiss GDPR banners)
  - Shadow DOM flattening (flatten_shadow_dom — modern web components)
  - Full-page scanning (infinite scroll handling)
  - Lazy load handling (wait_for_images)
  - Screenshot capture for debugging
  - Caching (avoid redundant fetches)
  - Clean markdown with PruningContentFilter (AI-friendly output)
  - Link extraction and scoring
  - Media extraction (images, srcset, picture)
  - IFrame content extraction
  - Overlay/popup removal
  - Session management
  - Dynamic JS execution
  - Auto-retry with fallback (max_retries + fallback_fetch_function)
  - Batch parallel scraping (arun_many)
  - Speed profiles (text_mode + light_mode for bulk operations)
  - Memory saving mode (recycle pages in batch ops)

Usage:
    from app.collectors.crawl4ai_config import (
        get_browser_config, get_crawl_config, scrape_url
    )

    # Quick scrape
    result = await scrape_url("https://example.com/article")

    # Fast bulk scrape (skip images/CSS)
    results = await scrape_urls_batch(["url1", "url2"], fast=True)

    # Custom config
    browser_cfg = get_browser_config(text_mode=True)
    crawl_cfg = get_crawl_config(screenshot=True, scan_full_page=True)
"""

import logging

logger = logging.getLogger(__name__)


import asyncio
import os


# ---------------------------------------------------------------------------
# Playwright fallback for crawl4ai's fallback_fetch_function
# ---------------------------------------------------------------------------


async def _playwright_fallback(url: str) -> str:
    """Fallback scraper using Playwright stealth when crawl4ai fails.

    Wired into crawl4ai's fallback_fetch_function so retries
    happen automatically.
    """
    try:
        from app.collectors.news_playwright import scrape_article_sync

        text = await asyncio.to_thread(scrape_article_sync, url, True, 20000, 5000)
        if text and len(text) > 50:
            chars = len(text)
            logger.info(f"[crawl4ai] Playwright fallback: {chars} chars")
            return text
    except ImportError:
        logger.info("[crawl4ai] Playwright not installed for fallback")
    except Exception as e:
        logger.info(f"[crawl4ai] Playwright fallback error: {e}")
    return ""


# ---------------------------------------------------------------------------
# BrowserConfig
# ---------------------------------------------------------------------------


def get_browser_config(
    headless: bool = True,
    proxy: str | None = None,
    use_persistent_context: bool = False,
    user_data_dir: str | None = None,
    viewport_width: int = 1280,
    viewport_height: int = 800,
    # v0.8.5 features
    text_mode: bool = False,
    light_mode: bool = False,
    avoid_ads: bool = True,
    memory_saving: bool = False,
    max_pages_before_recycle: int = 0,
) -> "BrowserConfig":  # noqa: F821
    """Build a BrowserConfig with stealth + anti-detection defaults.

    Features (v0.8.5):
      - enable_stealth: native stealth mode (replaces deprecated magic/simulate_user)
      - avoid_ads: remove ad content from pages
      - text_mode: skip images/CSS for 2-3x speed (use for bulk scraping)
      - light_mode: lightweight browser profile
      - memory_saving_mode: recycle pages to conserve memory in batch ops
      - Proxy support (pass "http://user:pass@proxy:8080")
      - Persistent sessions (saves cookies/localStorage across runs)
      - HTTPS error tolerance
    """
    from crawl4ai import BrowserConfig

    kwargs = dict(
        browser_type="chromium",
        headless=headless,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        ignore_https_errors=True,
        verbose=False,
        # v0.8.5 stealth + performance
        enable_stealth=True,
        avoid_ads=avoid_ads,
        text_mode=text_mode,
        light_mode=light_mode,
        memory_saving_mode=memory_saving,
    )

    # Max pages before recycling (0 = disabled)
    if max_pages_before_recycle > 0:
        kwargs["max_pages_before_recycle"] = max_pages_before_recycle

    # Proxy
    if proxy:
        kwargs["proxy_config"] = {"server": proxy}

    # Persistent sessions (keep cookies, login state)
    if use_persistent_context:
        kwargs["use_persistent_context"] = True
        kwargs["user_data_dir"] = user_data_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "crawl4ai_profiles"
        )

    return BrowserConfig(**kwargs)


def get_fast_browser_config() -> "BrowserConfig":  # noqa: F821
    """Lightweight browser config for bulk scraping. 2-3x faster.

    Skips images, CSS, and ads. Uses memory saving mode.
    Use this for RSS body scraping, bulk article collection, etc.
    """
    return get_browser_config(
        text_mode=True,
        light_mode=True,
        avoid_ads=True,
        memory_saving=True,
        max_pages_before_recycle=10,
    )


# ---------------------------------------------------------------------------
# CrawlerRunConfig
# ---------------------------------------------------------------------------


def get_crawl_config(
    # Content
    css_selector: str | None = None,
    target_elements: list[str] | None = None,
    excluded_tags: list[str] | None = None,
    excluded_selector: str | None = None,
    word_count_threshold: int = 30,
    # Stealth (v0.8.5 — keep legacy params for backward compat until fully removed)
    remove_overlay_elements: bool = True,
    remove_consent_popups: bool = True,
    # Page scanning
    scan_full_page: bool = True,
    scroll_delay: float = 0.3,
    max_scroll_steps: int | None = 10,
    # Shadow DOM
    flatten_shadow_dom: bool = True,
    # Timing
    wait_until: str = "domcontentloaded",
    page_timeout: int = 30000,
    wait_for_images: bool = True,
    delay_before_return_html: float = 0.5,
    # Media
    screenshot: bool = False,
    screenshot_wait_for: float | None = None,
    exclude_external_images: bool = False,
    image_score_threshold: int = 3,
    # Links
    score_links: bool = True,
    exclude_social_media_links: bool = True,
    exclude_domains: list[str] | None = None,
    # Caching
    cache_mode: str | None = "ENABLED",
    session_id: str | None = None,
    # IFrames
    process_iframes: bool = True,
    # JS execution
    js_code: str | list[str] | None = None,
    # Markdown
    use_pruning_filter: bool = True,
    pruning_threshold: float = 0.4,
    # Retry + Fallback (v0.8.5)
    max_retries: int = 2,
    use_fallback: bool = True,
    # Debugging
    verbose: bool = False,
    log_console: bool = False,
    capture_network_requests: bool = False,
) -> "CrawlerRunConfig":  # noqa: F821
    """Build a CrawlerRunConfig with all v0.8.5 features enabled.

    Features:
      🕵️ Stealth: enable_stealth on BrowserConfig (replaces simulate_user/magic)
      🛡️ Consent popup removal: auto-dismiss GDPR/cookie banners
      🔄 Full-page scan: auto-scroll to load all dynamic content
      🌑 Shadow DOM: flatten shadow DOM for modern web components
      🕰️ Lazy load: wait for images to fully load
      📸 Screenshots: capture page state for debugging
      💾 Caching: avoid redundant fetches (ENABLED by default)
      📝 Clean markdown: PruningContentFilter removes boilerplate
      🔗 Link scoring: quality scores for all links
      🖼️ Media: image scoring and filtering
      📡 IFrames: inline iframe content
      🛡️ Overlay removal: auto-dismiss modals/popups
      🔁 Auto-retry: max_retries with Playwright fallback
    """
    from crawl4ai import CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter

    kwargs = dict(
        # Content processing
        word_count_threshold=word_count_threshold,
        excluded_tags=excluded_tags or ["script", "style", "nav", "footer", "header"],
        only_text=False,
        # Overlay + consent popup removal
        remove_overlay_elements=remove_overlay_elements,
        remove_consent_popups=remove_consent_popups,
        # Page scanning (infinite scroll)
        scan_full_page=scan_full_page,
        scroll_delay=scroll_delay,
        # Shadow DOM
        flatten_shadow_dom=flatten_shadow_dom,
        # Page timing
        wait_until=wait_until,
        page_timeout=page_timeout,
        wait_for_images=wait_for_images,
        delay_before_return_html=delay_before_return_html,
        # Media
        screenshot=screenshot,
        image_score_threshold=image_score_threshold,
        exclude_external_images=exclude_external_images,
        # Links
        score_links=score_links,
        exclude_social_media_links=exclude_social_media_links,
        # IFrames
        process_iframes=process_iframes,
        # Retry
        max_retries=max_retries,
        # Debug
        verbose=verbose,
        log_console=log_console,
        capture_network_requests=capture_network_requests,
    )

    # Optional params
    if max_scroll_steps is not None:
        kwargs["max_scroll_steps"] = max_scroll_steps
    if css_selector:
        kwargs["css_selector"] = css_selector
    if target_elements:
        kwargs["target_elements"] = target_elements
    if excluded_selector:
        kwargs["excluded_selector"] = excluded_selector
    if screenshot_wait_for is not None:
        kwargs["screenshot_wait_for"] = screenshot_wait_for
    if exclude_domains:
        kwargs["exclude_domains"] = exclude_domains
    if session_id:
        kwargs["session_id"] = session_id
    if js_code:
        kwargs["js_code"] = js_code

    # Fallback: wire Playwright as automatic fallback
    if use_fallback:
        kwargs["fallback_fetch_function"] = _playwright_fallback

    # Caching
    if cache_mode:
        cache_map = {
            "ENABLED": CacheMode.ENABLED,
            "BYPASS": CacheMode.BYPASS,
            "DISABLED": CacheMode.DISABLED,
        }
        kwargs["cache_mode"] = cache_map.get(cache_mode, CacheMode.ENABLED)

    # Markdown generation with pruning filter
    if use_pruning_filter:
        kwargs["markdown_generator"] = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=pruning_threshold,
                threshold_type="fixed",
                min_word_threshold=word_count_threshold,
            )
        )

    return CrawlerRunConfig(**kwargs)


# ---------------------------------------------------------------------------
# scrape_url — single URL
# ---------------------------------------------------------------------------


async def scrape_url(
    url: str,
    max_chars: int = 15000,
    screenshot: bool = False,
    scan_full_page: bool = False,
    css_selector: str | None = None,
    js_code: str | None = None,
    rate_limit_delay: float = 3.0,
    fast: bool = False,
) -> dict:
    """Scrape a URL with full-featured crawl4ai v0.8.5 config.

    Returns dict with:
      - text: clean article text (markdown)
      - links: extracted links
      - media: extracted media (images)
      - screenshot: base64 screenshot (if requested)
      - metadata: page metadata
      - success: bool

    Args:
        fast: Use lightweight browser (text_mode, no images) for speed.
        rate_limit_delay: seconds to sleep after scrape.
    """
    from crawl4ai import AsyncWebCrawler

    browser_cfg = get_fast_browser_config() if fast else get_browser_config()
    crawl_cfg = get_crawl_config(
        screenshot=screenshot,
        scan_full_page=scan_full_page,
        css_selector=css_selector,
        js_code=js_code,
    )

    result = {
        "text": "",
        "links": [],
        "media": [],
        "screenshot": None,
        "metadata": {},
        "success": False,
        "url": url,
    }

    try:
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            r = await crawler.arun(url=url, config=crawl_cfg)

            if r.success:
                result = _process_crawl_result(r, url, max_chars)
            else:
                result["error"] = getattr(r, "error_message", "Unknown error")

    except Exception as e:
        result["error"] = str(e)

    # Rate limit
    await asyncio.sleep(rate_limit_delay)

    return result


# ---------------------------------------------------------------------------
# scrape_urls_batch — parallel multi-URL via arun_many
# ---------------------------------------------------------------------------


async def scrape_urls_batch(
    urls: list[str],
    max_chars: int = 15000,
    fast: bool = True,
    rate_limit_delay: float = 2.0,
) -> list[dict]:
    """Scrape multiple URLs in parallel using crawl4ai's arun_many.

    Much faster than sequential scrape_url() calls — uses internal
    browser tab pooling and concurrent fetching.

    Args:
        fast: Use lightweight browser (text_mode, no images) for speed.
        rate_limit_delay: seconds to sleep after entire batch.
    """
    from crawl4ai import AsyncWebCrawler

    browser_cfg = get_fast_browser_config() if fast else get_browser_config()
    crawl_cfg = get_crawl_config()

    results = []
    try:
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            raw_results = await crawler.arun_many(urls=urls, config=crawl_cfg)
            for i, r in enumerate(raw_results):
                url = urls[i] if i < len(urls) else "unknown"
                if r.success:
                    results.append(_process_crawl_result(r, url, max_chars))
                else:
                    results.append(
                        {
                            "text": "",
                            "links": [],
                            "media": [],
                            "screenshot": None,
                            "metadata": {},
                            "success": False,
                            "url": url,
                            "error": getattr(r, "error_message", "Unknown error"),
                        }
                    )
    except Exception as e:
        # If batch fails entirely, return error for each URL
        for url in urls:
            results.append(
                {
                    "text": "",
                    "links": [],
                    "media": [],
                    "screenshot": None,
                    "metadata": {},
                    "success": False,
                    "url": url,
                    "error": f"Batch scrape error: {e}",
                }
            )

    await asyncio.sleep(rate_limit_delay)
    return results


# ---------------------------------------------------------------------------
# Legacy sequential scrape (kept for backward compat)
# ---------------------------------------------------------------------------


async def scrape_urls(
    urls: list[str],
    max_chars: int = 15000,
    rate_limit_delay: float = 5.0,
) -> list[dict]:
    """Scrape multiple URLs sequentially with rate limiting.

    For parallel scraping, use scrape_urls_batch() instead.
    Goes slow to avoid IP bans — `rate_limit_delay` seconds between each URL.
    """
    results = []
    for url in urls:
        r = await scrape_url(
            url, max_chars=max_chars, rate_limit_delay=rate_limit_delay
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _process_crawl_result(r, url: str, max_chars: int) -> dict:
    """Normalize a CrawlResult into our standard dict format."""
    result = {
        "text": "",
        "links": [],
        "media": [],
        "screenshot": None,
        "metadata": {},
        "success": False,
        "url": url,
    }

    # Text (prefer pruned fit_markdown)
    text = ""
    if hasattr(r, "fit_markdown") and r.fit_markdown:
        text = r.fit_markdown
    elif r.markdown:
        text = r.markdown
    result["text"] = text[:max_chars] if text else ""

    # Links
    if hasattr(r, "links") and r.links:
        result["links"] = r.links

    # Media
    if hasattr(r, "media") and r.media:
        result["media"] = r.media

    # Screenshot
    if hasattr(r, "screenshot") and r.screenshot:
        result["screenshot"] = r.screenshot

    # Metadata
    if hasattr(r, "metadata") and r.metadata:
        result["metadata"] = r.metadata

    result["success"] = len(result["text"]) > 50
    return result

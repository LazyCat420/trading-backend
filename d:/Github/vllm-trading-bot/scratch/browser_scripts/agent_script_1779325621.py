from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.youtube.com/watch?v=O-8bwEiRuUo')
    # Check for caption availability in page source
    content = page.content()
    has_caption = '"hasCaption":true' in content or '"isCaption":true' in content
    print(f"hasCaption found: {has_caption}")
    # Also check for any caption-related metadata
    if 'caption' in content.lower():
        print("Caption keywords found in page")
    browser.close()
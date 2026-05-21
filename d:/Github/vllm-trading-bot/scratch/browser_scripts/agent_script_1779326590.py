from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.youtube.com/watch?v=vQcHJCmpyZM')
    # Check for caption availability in the page source
    source = page.content()
    has_caption = '"hasCaption":true' in source or '"isCaption":true' in source
    print(f"hasCaption found: {has_caption}")
    # Also check for any caption-related metadata
    if 'caption' in source.lower():
        # Find caption-related lines
        for line in source.split('\n'):
            if 'caption' in line.lower() and 'true' in line.lower():
                print(f"Caption indicator: {line.strip()[:200]}")
    browser.close()
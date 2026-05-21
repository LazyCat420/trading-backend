from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.youtube.com/watch?v=WbDnLofcdQs')
    # Check for hasCaption in the page source
    content = page.content()
    if 'hasCaption": true' in content or '"hasCaption":true' in content:
        print("CAPTIONS_AVAILABLE: Video has captions.")
    else:
        print("NO_CAPTIONS: Video has no captions available.")
    browser.close()
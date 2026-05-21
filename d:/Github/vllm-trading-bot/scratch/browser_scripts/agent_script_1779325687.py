from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.youtube.com/watch?v=vQcHJCmpyZM')
    
    # Check for caption availability in the page source
    source = page.content()
    has_caption = '"hasCaption":true' in source or '"captionTracks"' in source
    
    print(f"hasCaption found in source: {has_caption}")
    
    # Also try to find captionTracks specifically
    if '"captionTracks"' in source:
        print("Video HAS captionTracks - captions are available")
    else:
        print("Video does NOT have captionTracks - no captions available")
    
    browser.close()
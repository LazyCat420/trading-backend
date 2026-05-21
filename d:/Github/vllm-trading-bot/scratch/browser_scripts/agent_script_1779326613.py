from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.youtube.com/watch?v=vQcHJCmpyZM')
    source = page.content()
    has_caption = '"hasCaption":true' in source
    print(f"hasCaption: {has_caption}")
    browser.close()
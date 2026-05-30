import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        with patch("app.collectors.news_collector.scraper_client.collect", new_callable=AsyncMock) as mock_scrape:'
replacement = '        with patch("app.services.scraper_client.ScraperClient.collect", new_callable=AsyncMock) as mock_scrape:'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)

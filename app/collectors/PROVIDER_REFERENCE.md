# News & Data API Provider Reference

> Auto-generated from live API audit on 2026-05-02.
> All 14 API keys tested and verified working.

## Tier 1: Full-Text News APIs (Best for Deep Research)

Use these when the agent needs to READ and ANALYZE article content.

### WorldNewsAPI ⭐ BEST FULL TEXT
- **Content**: Full article text (up to 5,847 chars per article)
- **Rate Limit**: 300 req/day
- **Capabilities**: keyword search, source country, language, sentiment, full text
- **Best For**: Deep article analysis, sentiment research, when you need the actual content
- **API**: `GET https://api.worldnewsapi.com/search-news?text={query}&language=en`

### AlphaVantage ⭐ BEST SENTIMENT
- **Content**: Full text (473ch) + built-in sentiment scores per ticker
- **Rate Limit**: 25 req/day (conservative — use wisely)
- **Capabilities**: news sentiment, per-ticker sentiment scores, topic filter, ticker filter, date range
- **Best For**: Sentiment analysis, when you need numerical sentiment scores alongside articles
- **API**: `GET https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}`

### Finnhub ⭐ HIGHEST VOLUME
- **Content**: Full text (500ch) with 244 articles per ticker query
- **Rate Limit**: 60 calls/min (800/day effective)
- **Capabilities**: per-ticker news, company news, market news, sentiment score
- **Best For**: Bulk news collection, per-ticker coverage, high-frequency updates
- **API**: `GET https://finnhub.io/api/v1/company-news?symbol={ticker}&from={date}&to={date}`

### Polygon/Massive ⭐ PUBLISHER QUALITY
- **Content**: Full text (576ch) from premium publishers
- **Rate Limit**: 5 req/min (unlimited daily)
- **Capabilities**: ticker news, publisher filter, date range, full article text
- **Best For**: High-quality publisher sources, when you need reliable financial journalism
- **API**: `GET https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=10`

### MarketAux
- **Content**: Full text (329ch) with entity detection
- **Rate Limit**: 100 req/day
- **Capabilities**: entity detection, ticker filter, industry filter, full text
- **Best For**: Entity-tagged articles, industry-specific research
- **API**: `GET https://api.marketaux.com/v1/news/all?symbols={ticker}`

### StockData
- **Content**: Full text (329ch) with entity detection + sentiment
- **Rate Limit**: 100 req/day
- **Capabilities**: ticker filter, entity detection, industry filter, sentiment
- **Best For**: Ticker-specific research with sentiment overlay
- **API**: `GET https://api.stockdata.org/v1/news/all?symbols={ticker}`

## Tier 2: Snippet-Only News APIs (Best for Discovery)

Use these when scanning for headlines/topics — they return summaries, not full articles.

### CurrentsAPI ⭐ BEST VOLUME (snippets)
- **Content**: Snippets (276ch), 30 articles per call
- **Rate Limit**: 600 req/day (highest quota)
- **Best For**: Broad scanning, high-volume headline monitoring

### GNews
- **Content**: Snippets (266ch), 459 total articles available
- **Rate Limit**: 100 req/day
- **Best For**: Topic scanning, country-specific news

### NewsAPI
- **Content**: Snippets (260ch), 108 total results
- **Rate Limit**: 100 req/day
- **Best For**: Source-specific filtering, broad keyword search

### TheNewsAPI
- **Content**: Snippets (163ch), 3 per page
- **Rate Limit**: 150 req/day
- **Best For**: Entity detection, category filtering

## Tier 3: Market Data APIs (Not News)

### TwelveData
- **Data**: OHLCV prices, technical indicators, forex, crypto
- **Rate Limit**: 800 req/day, 8/min
- **Best For**: Real-time quotes, technical analysis

### FRED (Federal Reserve)
- **Data**: GDP, CPI, unemployment, interest rates, economic indicators
- **Rate Limit**: 120 req/min (practically unlimited)
- **Best For**: Macro analysis, economic indicators

### EIA (Energy Information Administration)
- **Data**: Crude oil prices, natural gas, energy production
- **Rate Limit**: Unlimited
- **Best For**: Energy sector analysis, commodity prices

### FMP (Financial Modeling Prep)
- **Data**: Stock news (full text), press releases, SEC filings, earnings
- **Rate Limit**: 250 req/day
- **Best For**: Company-specific press releases and SEC filings

## Agent Decision Matrix

| Need | Best Provider | Reason |
|------|--------------|--------|
| Deep article analysis | WorldNewsAPI | Longest full text (5.8k chars) |
| Sentiment scores | AlphaVantage | Built-in per-ticker sentiment |
| Bulk ticker news | Finnhub | 244 articles/call, 60 calls/min |
| Premium publishers | Polygon/Massive | High-quality financial journalism |
| Headline scanning | CurrentsAPI | 600 req/day, 30 articles/call |
| Macro indicators | FRED | GDP, CPI, rates — unlimited |
| Energy/oil | EIA | Direct government data |
| SEC filings | FMP | Press releases + 10-K access |

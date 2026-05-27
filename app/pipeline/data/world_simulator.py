"""
World Simulator — Generates simulated market conditions for tickers.

Used in 'simulation' mode to bypass live API calls and generate realistic stock price
movements, technical indicators, and news articles for testing and evaluation.
"""

import datetime
import random
import logging
import uuid
import math
from app.db.connection import get_db

logger = logging.getLogger(__name__)

def generate_simulated_world(tickers: list[str], trend: str = "bullish", news_sentiment: str = "positive") -> None:
    """
    Generates simulated data for the given tickers and writes them to the database.
    
    Args:
        tickers: List of ticker symbols to simulate
        trend: Price trend direction ('bullish', 'bearish', 'neutral', 'volatile')
        news_sentiment: Overall news tone ('positive', 'negative', 'neutral')
    """
    logger.info(
        "[WORLD-SIM] Generating simulated world for %s | trend=%s | news_sentiment=%s",
        tickers, trend, news_sentiment
    )
    
    for ticker in tickers:
        ticker = ticker.upper()
        
        # 1. Clear existing simulated data for this ticker to avoid mixing historical periods
        _clear_ticker_data(ticker)
        
        # 2. Generate 180 days of simulated daily prices
        dates, prices = _generate_price_history(ticker, trend)
        
        # 3. Calculate and save technical indicators
        _generate_technicals(ticker, dates, prices)
        
        # 4. Generate and save fundamentals
        _generate_fundamentals(ticker, prices[-1], trend)
        
        # 5. Generate and save news articles and Reddit posts
        _generate_articles_and_posts(ticker, news_sentiment)

def _clear_ticker_data(ticker: str) -> None:
    """Clear existing data in market tables for the ticker."""
    with get_db() as db:
        try:
            # Delete from price_history
            db.execute("DELETE FROM price_history WHERE ticker = %s", [ticker])
            # Delete from technicals
            db.execute("DELETE FROM technicals WHERE ticker = %s", [ticker])
            # Delete from fundamentals
            db.execute("DELETE FROM fundamentals WHERE ticker = %s", [ticker])
            # Delete from news_articles
            db.execute("DELETE FROM news_articles WHERE ticker = %s", [ticker])
            # Delete from reddit_posts
            db.execute("DELETE FROM reddit_posts WHERE ticker = %s", [ticker])
            # Delete from balance_sheet
            db.execute("DELETE FROM balance_sheet WHERE ticker = %s", [ticker])
            # Delete from financial_history
            db.execute("DELETE FROM financial_history WHERE ticker = %s", [ticker])
            logger.info("[WORLD-SIM] Cleared existing database records for %s", ticker)
        except Exception as e:
            logger.error("[WORLD-SIM] Failed to clear existing records for %s: %s", ticker, e)

def _generate_price_history(ticker: str, trend: str) -> tuple[list[datetime.date], list[float]]:
    """Generates 180 days of daily price data (OHLCV) and saves to price_history."""
    dates = []
    prices = []
    
    # Establish trend parameters (daily drift and volatility/standard deviation)
    # Note: Negated because price history is reversed at the end to land on the baseline today.
    if trend == "bullish":
        drift = -0.0012   # Upward drift (approx +20% over 180 days chronologically)
        vol = 0.015
    elif trend == "bearish":
        drift = 0.0015  # Downward drift (approx -25% over 180 days chronologically)
        vol = 0.015
    elif trend == "volatile":
        drift = -0.0002
        vol = 0.035      # High volatility
    else: # neutral
        drift = -0.0001
        vol = 0.010      # Low volatility
        
    # Start with a baseline price
    price = 200.0
    today = datetime.date.today()
    
    # Generate prices backwards or forwards? It's easiest to generate forward and align dates
    temp_prices = []
    current_price = price
    
    # Simple Box-Muller transform for normal distribution
    def normal_sample(mean=0, std=1):
        u1 = random.random()
        u2 = random.random()
        # Avoid log(0)
        while u1 == 0:
            u1 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z
        
    for i in range(180):
        change_pct = drift + normal_sample(0, vol)
        current_price = current_price * (1.0 + change_pct)
        if current_price < 2.0:  # Prevent price dropping to zero/penny stock unless requested
            current_price = 2.0
        temp_prices.append(current_price)
        
    # Reverse to make them chronological ending today
    temp_prices.reverse()
    
    rows = []
    for idx, p in enumerate(temp_prices):
        day = today - datetime.timedelta(days=179 - idx)
        dates.append(day)
        prices.append(p)
        
        # Simulated OHLCV
        open_p = p * (1.0 + (random.random() - 0.5) * 0.01)
        high_p = max(p, open_p) * (1.0 + random.random() * 0.015)
        low_p = min(p, open_p) * (1.0 - random.random() * 0.015)
        close_p = p
        vol_v = int(5000000 + random.random() * 10000000)
        
        rows.append((ticker, day, open_p, high_p, low_p, close_p, vol_v, "world_simulator"))
        
    with get_db() as db:
        try:
            db.executemany(
                "INSERT INTO price_history (ticker, date, open, high, low, close, volume, source) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                rows
            )
            logger.info("[WORLD-SIM] Inserted %d price history records for %s", len(rows), ticker)
        except Exception as e:
            logger.error("[WORLD-SIM] Failed to insert price history for %s: %s", ticker, e)
            
    return dates, prices

def _generate_technicals(ticker: str, dates: list[datetime.date], prices: list[float]) -> None:
    """Computes technical indicators on price series and saves to technicals table."""
    
    # 1. Simple Moving Averages (SMA)
    def compute_sma(period):
        sma = []
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(prices[i]) # Fallback for early periods
            else:
                sma.append(sum(prices[i - period + 1 : i + 1]) / period)
        return sma
        
    sma50 = compute_sma(50)
    sma200 = compute_sma(200)
    sma20 = compute_sma(20)
    
    # 2. Exponential Moving Averages (EMA)
    def compute_ema(period):
        ema = []
        multiplier = 2.0 / (period + 1)
        # Seed EMA with simple average
        ema.append(prices[0])
        for i in range(1, len(prices)):
            val = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(val)
        return ema
        
    ema12 = compute_ema(12)
    ema26 = compute_ema(26)
    
    # 3. MACD
    macd = [e12 - e26 for e12, e26 in zip(ema12, ema26)]
    
    # MACD Signal Line (EMA 9 of MACD)
    macd_signal = []
    multiplier_sig = 2.0 / (9 + 1)
    macd_signal.append(macd[0])
    for i in range(1, len(macd)):
        val = (macd[i] - macd_signal[-1]) * multiplier_sig + macd_signal[-1]
        macd_signal.append(val)
        
    macd_hist = [m - s for m, s in zip(macd, macd_signal)]
    
    # 4. Bollinger Bands
    bb_upper = []
    bb_lower = []
    for i in range(len(prices)):
        if i < 19:
            bb_upper.append(prices[i] * 1.05)
            bb_lower.append(prices[i] * 0.95)
        else:
            slice_p = prices[i-19 : i+1]
            mean = sum(slice_p) / 20.0
            variance = sum((x - mean) ** 2 for x in slice_p) / 20.0
            std_dev = math.sqrt(variance)
            bb_upper.append(mean + 2.0 * std_dev)
            bb_lower.append(mean - 2.0 * std_dev)
            
    # 5. ATR (Average True Range)
    # Simulated ATR using rolling price volatility
    atr_14 = []
    for i in range(len(prices)):
        if i == 0:
            atr_14.append(prices[0] * 0.02)
        else:
            prev_close = prices[i-1]
            tr = max(prices[i] * 1.015 - prices[i] * 0.985, 
                     abs(prices[i] * 1.015 - prev_close), 
                     abs(prices[i] * 0.985 - prev_close))
            # Smooth it
            if i < 14:
                atr_14.append(tr)
            else:
                atr_14.append((atr_14[-1] * 13 + tr) / 14.0)
                
    # 6. RSI (Relative Strength Index)
    rsi_14 = []
    for i in range(len(prices)):
        if i < 14:
            rsi_14.append(50.0) # neutral seed
        else:
            gains = 0
            losses = 0
            for j in range(i-13, i+1):
                diff = prices[j] - prices[j-1]
                if diff > 0:
                    gains += diff
                else:
                    losses += abs(diff)
            avg_gain = gains / 14.0
            avg_loss = losses / 14.0
            if avg_loss == 0:
                rs = 100.0
            else:
                rs = avg_gain / avg_loss
            rsi_14.append(100.0 - (100.0 / (1.0 + rs)))
            
    # 7. Stochastic %K and %D
    stoch_k = []
    for i in range(len(prices)):
        if i < 14:
            stoch_k.append(50.0)
        else:
            slice_p = prices[i-13 : i+1]
            lowest = min(slice_p)
            highest = max(slice_p)
            current = prices[i]
            if highest == lowest:
                stoch_k.append(50.0)
            else:
                stoch_k.append(((current - lowest) / (highest - lowest)) * 100.0)
                
    # Smooth %K to get %D (3-day SMA of %K)
    stoch_d = []
    for i in range(len(stoch_k)):
        if i < 2:
            stoch_d.append(stoch_k[i])
        else:
            stoch_d.append(sum(stoch_k[i-2 : i+1]) / 3.0)
            
    # Support & Resistance
    # Computed as minimum and maximum of the last 30 days
    support = []
    resistance = []
    for i in range(len(prices)):
        lookback = min(i, 30)
        if lookback == 0:
            support.append(prices[0] * 0.95)
            resistance.append(prices[0] * 1.05)
        else:
            slice_p = prices[i - lookback : i + 1]
            support.append(min(slice_p) * 0.98)
            resistance.append(max(slice_p) * 1.02)
            
    # Bulk insert into technicals
    rows = []
    for idx, day in enumerate(dates):
        rows.append((
            ticker, day, rsi_14[idx], macd[idx], macd_signal[idx], macd_hist[idx],
            sma20[idx], sma50[idx], sma200[idx], ema12[idx], ema26[idx],
            bb_upper[idx], sma20[idx], bb_lower[idx], atr_14[idx], 14.0,
            stoch_k[idx], stoch_d[idx], 0.0, prices[idx], support[idx], resistance[idx]
        ))
        
    with get_db() as db:
        try:
            db.executemany(
                "INSERT INTO technicals (ticker, date, rsi_14, macd, macd_signal, macd_hist, "
                "sma_20, sma_50, sma_200, ema_12, ema_26, bb_upper, bb_mid, bb_lower, "
                "atr_14, adx_14, stoch_k, stoch_d, obv, vwap, support, resistance) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                rows
            )
            logger.info("[WORLD-SIM] Inserted %d technical indicator records for %s", len(rows), ticker)
        except Exception as e:
            logger.error("[WORLD-SIM] Failed to insert technical indicators for %s: %s", ticker, e)

def _generate_fundamentals(ticker: str, current_price: float, trend: str) -> None:
    """Generates and saves simulated fundamental metrics for the ticker."""
    
    if trend == "bullish":
        pe = 35.5
        forward_pe = 28.0
        rev_growth = 0.22
        profit_margin = 0.18
        debt_to_equity = 0.85
    elif trend == "bearish":
        pe = 14.2
        forward_pe = 16.5
        rev_growth = -0.06
        profit_margin = 0.04
        debt_to_equity = 1.85
    elif trend == "volatile":
        pe = 22.0
        forward_pe = 20.0
        rev_growth = 0.08
        profit_margin = 0.10
        debt_to_equity = 1.15
    else: # neutral
        pe = 19.5
        forward_pe = 18.0
        rev_growth = 0.04
        profit_margin = 0.09
        debt_to_equity = 1.05
        
    with get_db() as db:
        try:
            db.execute(
                "INSERT INTO fundamentals (ticker, snapshot_date, source, market_cap, pe_ratio, "
                "forward_pe, peg_ratio, price_to_book, price_to_sales, ev_to_ebitda, profit_margin, "
                "roe, roa, revenue, revenue_growth, net_income, debt_to_equity, current_ratio, beta, "
                "week_52_high, week_52_low, short_float_pct) "
                "VALUES (%s, %s, 'world_simulator', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    ticker, datetime.date.today(), current_price * 5_000_000_000, pe, forward_pe,
                    1.5, 4.2, 3.8, 12.5, profit_margin, 0.15, 0.08, 25_000_000_000, rev_growth,
                    4_500_000_000, debt_to_equity, 1.6, 1.25, current_price * 1.2, current_price * 0.7, 0.03
                ]
            )
            
            # Populate balance_sheet and financial_history table for depth
            db.execute(
                "INSERT INTO balance_sheet (ticker, period_end, total_assets, total_liabilities, "
                "total_equity, cash, total_debt, working_capital) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [ticker, datetime.date.today() - datetime.timedelta(days=60), 80_000_000_000, 40_000_000_000, 40_000_000_000, 15_000_000_000, 20_000_000_000, 10_000_000_000]
            )
            
            db.execute(
                "INSERT INTO financial_history (ticker, period_type, period_end, revenue, gross_profit, "
                "operating_income, net_income, eps, free_cash_flow) "
                "VALUES (%s, 'quarterly', %s, %s, %s, %s, %s, 1.25, %s)",
                [ticker, datetime.date.today() - datetime.timedelta(days=60), 6_250_000_000, 4_100_000_000, 1_500_000_000, 1_125_000_000, 950_000_000]
            )
            
            logger.info("[WORLD-SIM] Inserted simulated fundamental records for %s", ticker)
        except Exception as e:
            logger.error("[WORLD-SIM] Failed to insert fundamentals for %s: %s", ticker, e)

def _generate_articles_and_posts(ticker: str, sentiment: str) -> None:
    """Generates simulated news and reddit posts and saves them."""
    
    # Seed details based on sentiment
    if sentiment == "positive":
        news_templates = [
            ("{ticker} Announces Unprecedented Q1 Earnings Beat, Stock Skyrockets", "The company reported massive earnings per share beat, highlighting strong global demand and expansion of its high-margin cloud services division. Revenues grew 25% year over year.", "Bloomberg", 85),
            ("{ticker} Secures Strategic Multi-Billion Dollar Contract, Analysts Bullish", "A new landmark deal with sovereign partners will deploy {ticker}'s infrastructure globally over the next 5 years, guaranteeing long-term recurring revenue stream.", "Reuters", 90),
            ("{ticker} CEO Outlines Bold AI Roadmap in Latest Interview", "Highlighting breakthrough advancements, the CEO announced the launch of new generative platforms that could double efficiency and open net new market segments.", "Wall Street Journal", 80),
        ]
        reddit_templates = [
            ("Why I am loading up on {ticker} shares before the conference", "Look at the fundamentals. P/E is super reasonable given their massive 22% growth rate. Technicals show SMA 50 is holding support. Easiest buy in tech right now.", 142, 0.85),
            ("DD: The macro tailwinds behind {ticker} are crazy", "Institutional holders have increased by 8% this quarter. Insider selling is zero. Options flows show huge call sweeps at higher strikes.", 215, 0.78),
        ]
    elif sentiment == "negative":
        news_templates = [
            ("{ticker} Stock Plummets Following Disappointing Revenue Forecast", "High inventory levels and sluggish supply chain conditions led management to lower its full-year outlook. Profit margins compressed by 300 basis points.", "Bloomberg", 20),
            ("{ticker} Under Regulator Investigation Over Data Compliance Issues", "The agency opened a probe into security procedures, risking major fines and potential service suspensions in European territories.", "Reuters", 15),
            ("Competitors Eating Into {ticker}'s Core Market Share, Report Warns", "A comprehensive study indicates customers are shifting to cheaper options, leaving {ticker} with high overhead and declining growth prospects.", "Financial Times", 30),
        ]
        reddit_templates = [
            ("Why I just sold all my {ticker} stock for a loss", "This company is completely lost. Technicals show a clear death cross as the SMA 50 broke below SMA 200. High debt leverage is going to crash them.", 84, 0.15),
            ("Short Thesis: {ticker} has massive downside risk", "Insider selling is spiking. The P/E multiple is completely disconnected from their contracting revenue growth. I bought puts at $190.", 195, 0.12),
        ]
    else: # neutral
        news_templates = [
            ("{ticker} Announces Date for Annual Shareholders Meeting", "The company will hold its annual meeting in late June, where it will present routine proxy statements and updates to its board structure.", "Yahoo Finance", 50),
            ("{ticker} Declares Standard Quarterly Dividend", "Consistent with historical policy, the company announced a regular dividend payable next month to shareholders of record.", "MarketWatch", 52),
            ("{ticker} Participates in Panel on Sector Innovation", "Company executives discussed broad industry trends at the summit, highlighting collaborative efforts and generic technology developments.", "TechCrunch", 55),
        ]
        reddit_templates = [
            ("Any thoughts on {ticker} at these levels?", "It seems to be trading in a range. SMA 50 and SMA 200 are flat, RSI is exactly 50. Is this just dead money for now?", 45, 0.50),
            ("{ticker} Weekly Discussion Thread", "What are you doing with your shares? Selling covered calls seems like the only way to make money here since volatility is so low.", 68, 0.48),
        ]
        
    now = datetime.datetime.now()
    
    with get_db() as db:
        try:
            # 1. Insert news articles
            for idx, (title, summary, publisher, score) in enumerate(news_templates):
                title = title.format(ticker=ticker)
                summary = summary.format(ticker=ticker)
                art_id = f"sim-news-{ticker}-{idx}-{uuid.uuid4().hex[:6]}"
                pub_date = now - datetime.timedelta(hours=idx * 6)
                
                db.execute(
                    "INSERT INTO news_articles (id, ticker, title, publisher, url, published_at, summary, "
                    "llm_summary, source, summarized_at, quality_status, quality_score, is_cluster_winner) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'world_simulator', %s, 'passed', %s, TRUE)",
                    [
                        art_id, ticker, title, publisher, f"https://example.com/{art_id}", pub_date,
                        summary, f"SUMMARY: {summary}", now, score
                    ]
                )
                
            # 2. Insert reddit posts
            for idx, (title, body, score, sentiment_val) in enumerate(reddit_templates):
                title = title.format(ticker=ticker)
                body = body.format(ticker=ticker)
                post_id = f"sim-reddit-{ticker}-{idx}-{uuid.uuid4().hex[:6]}"
                pub_date = now - datetime.timedelta(hours=idx * 8)
                
                db.execute(
                    "INSERT INTO reddit_posts (id, ticker, subreddit, title, body, score, upvote_ratio, "
                    "comment_count, flair, sentiment_score, summary, created_utc, summarized_at, quality_status, quality_score) "
                    "VALUES (%s, %s, 'wallstreetbets', %s, %s, %s, 0.85, 25, 'Discussion', %s, %s, %s, %s, 'passed', 80)",
                    [
                        post_id, ticker, title, body, score, sentiment_val, body[:150], pub_date, now
                    ]
                )
                
            logger.info("[WORLD-SIM] Inserted news and reddit records for %s", ticker)
        except Exception as e:
            logger.error("[WORLD-SIM] Failed to insert articles and posts for %s: %s", ticker, e)

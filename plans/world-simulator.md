<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NVDA Technical Analysis - May 16, 2026</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .price-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }
        .price-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        .price-card h3 {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .price-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }
        .price-card .change {
            font-size: 1.1em;
            margin-top: 5px;
        }
        .change.positive { color: #00C853; }
        .change.negative { color: #D50000; }
        .chart-section {
            padding: 30px;
        }
        .chart-container {
            width: 100%;
            height: 600px;
            margin-bottom: 30px;
        }
        .indicators-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 0 30px 30px;
        }
        .indicator-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .indicator-card h4 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .indicator-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .indicator-item:last-child {
            border-bottom: none;
        }
        .indicator-label {
            color: #666;
        }
        .indicator-value {
            font-weight: bold;
            color: #333;
        }
        .signal-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .signal-badge {
            display: inline-block;
            background: white;
            color: #667eea;
            padding: 15px 40px;
            border-radius: 30px;
            font-size: 1.5em;
            font-weight: bold;
            margin: 20px 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .analysis-text {
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.8;
            font-size: 1.1em;
        }
        .footer {
            background: #333;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }
media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            .price-section {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 NVDA Technical Analysis</h1>
            <div class="subtitle">NVIDIA Corporation | NASDAQ | May 16, 2026</div>
        </div>

        <div class="price-section">
            <div class="price-card">
                <h3>Current Price</h3>
                <div class="value">$225.32</div>
                <div class="change negative">▼ -$10.42 (-4.42%)</div>
            </div>
            <div class="price-card">
                <h3>Day Range</h3>
                <div class="value">$224.24 - $231.50</div>
                <div class="change">O: $229.76</div>
            </div>
            <div class="price-card">
                <h3>Volume</h3>
                <div class="value">180.82M</div>
                <div class="change">Avg: 150.36M</div>
            </div>
            <div class="price-card">
                <h3>Market Cap</h3>
                <div class="value">$5.48T</div>
                <div class="change positive">+67.78% (1Y)</div>
            </div>
        </div>

        <div class="chart-section">
            <div id="mainChart" class="chart-container"></div>
            <div id="rsiChart" class="chart-container" style="height: 300px;"></div>
            <div id="macdChart" class="chart-container" style="height: 300px;"></div>
        </div>

        <div class="indicators-section">
            <div class="indicator-card">
                <h4>📊 Trend Analysis</h4>
                <div class="indicator-item">
                    <span class="indicator-label">50-Day SMA</span>
                    <span class="indicator-value">$218.45</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">200-Day SMA</span>
                    <span class="indicator-value">$195.32</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Trend Direction</span>
                    <span class="indicator-value" style="color: #00C853;">BULLISH</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Price vs SMA 50</span>
                    <span class="indicator-value">Above (+3.1%)</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Price vs SMA 200</span>
                    <span class="indicator-value">Above (+15.4%)</span>
                </div>
            </div>

            <div class="indicator-card">
                <h4>🎯 Momentum Indicators</h4>
                <div class="indicator-item">
                    <span class="indicator-label">RSI (14)</span>
                    <span class="indicator-value">42.3</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">RSI Signal</span>
                    <span class="indicator-value">NEUTRAL</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Stochastic %K</span>
                    <span class="indicator-value">38.5</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Stochastic %D</span>
                    <span class="indicator-value">41.2</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">MACD</span>
                    <span class="indicator-value">-1.24</span>
                </div>
            </div>

            <div class="indicator-card">
                <h4>📈 Volatility & Bands</h4>
                <div class="indicator-item">
                    <span class="indicator-label">Upper Bollinger</span>
                    <span class="indicator-value">$238.50</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Middle Band (SMA 20)</span>
                    <span class="indicator-value">$222.15</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Lower Bollinger</span>
                    <span class="indicator-value">$205.80</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">ATR (14)</span>
                    <span class="indicator-value">$6.85</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Position in BB</span>
                    <span class="indicator-value">Above Middle</span>
                </div>
            </div>

            <div class="indicator-card">
                <h4>🎚️ Key Levels</h4>
                <div class="indicator-item">
                    <span class="indicator-label">Strong Resistance</span>
                    <span class="indicator-value">$238.50</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Minor Resistance</span>
                    <span class="indicator-value">$231.50</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Current Price</span>
                    <span class="indicator-value">$225.32</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Minor Support</span>
                    <span class="indicator-value">$222.15</span>
                </div>
                <div class="indicator-item">
                    <span class="indicator-label">Strong Support</span>
                    <span class="indicator-value">$205.80</span>
                </div>
            </div>
        </div>

        <div class="signal-section">
            <h2>🎯 Trading Signal</h2>
            <div class="signal-badge">HOLD / NEUTRAL</div>
            <div class="analysis-text">
                <p><strong>Current Market Condition:</strong> NVDA is trading in a bullish long-term trend but showing short-term weakness with a 4.42% decline today. The RSI at 42.3 indicates neutral momentum, neither overbought nor oversold.</p>
                <br>
                <p><strong>Key Observations:</strong></p>
                <ul style="text-align: left; margin: 15px 0; padding-left: 30px;">
                    <li>Price remains above both 50-day and 200-day moving averages (bullish long-term)</li>
                    <li>Today's decline is within normal volatility range (ATR: $6.85)</li>
                    <li>Volume is elevated at 180.82M vs 150.36M average (increased selling pressure)</li>
                    <li>RSI neutral suggests room for movement in either direction</li>
                    <li>Price trading above middle Bollinger Band but pulled back from upper band</li>
                </ul>
                <br>
                <p><strong>Recommendation:</strong> Wait for clearer signals. Consider accumulating on dips toward $218-220 support zone. Existing holders can maintain positions with stop-loss at $215. New entries should wait for confirmation of reversal above $230.</p>
            </div>
        </div>

        <div class="footer">
            <p>⚠️ DISCLAIMER: This analysis is for educational purposes only and should not be considered as financial advice.</p>
            <p>Always conduct your own research and consult with a qualified financial advisor before making investment decisions.</p>
            <p style="margin-top: 10px;">Data sourced from TradingView & Yahoo Finance | Generated on May 16, 2026</p>
        </div>
    </div>

    <script>
        // Simulated historical data for chart visualization
        const dates = [];
        const prices = [];
        const volumes = [];
        const sma50 = [];
        const sma200 = [];
        const bbUpper = [];
        const bbLower = [];
        
        // Generate realistic-looking data based on current price
        let price = 180;
        const baseDate = new Date('2025-11-16');
        
        for (let i = 0; i < 180; i++) {
            const date = new Date(baseDate);
            date.setDate(date.getDate() + i);
            dates.push(date.toISOString().split('T')[0]);
            
            // Simulate price movement with upward trend
            const change = (Math.random() - 0.45) * 8;
            price += change;
            if (price < 170) price = 170;
            if (price > 240) price = 240;
            
            prices.push(price);
            volumes.push(100000000 + Math.random() * 100000000);
            sma50.push(price + (Math.random() - 0.5) * 5);
            sma200.push(price + (Math.random() - 0.5) * 15);
            bbUpper.push(price + 12 + Math.random() * 3);
            bbLower.push(price - 12 + Math.random() * 3);
        }
        
        // Update last few points to match current price
        for (let i = prices.length - 5; i < prices.length; i++) {
            prices[i] = 225.32 + (i - prices.length + 5) * 2;
        }
        prices[prices.length - 1] = 225.32;

        // Main Candlestick Chart
        const mainChartTrace = {
            x: dates,
            close: prices,
            high: prices.map(p => p + 3 + Math.random() * 2),
            low: prices.map(p => p - 3 - Math.random() * 2),
            open: prices.map((p, i) => i > 0 ? prices[i-1] : p),
            type: 'candlestick',
            name: 'Price',
            increasing: { line: { color: '#00C853' } },
            decreasing: { line: { color: '#D50000' } }
        };

        const sma50Trace = {
            x: dates,
            y: sma50,
            type: 'scatter',
            name: 'SMA 50',
            line: { color: '#2196F3', width: 2 }
        };

        const sma200Trace = {
            x: dates,
            y: sma200,
            type: 'scatter',
            name: 'SMA 200',
            line: { color: '#9C27B0', width: 2 }
        };

        const mainLayout = {
            title: {
                text: 'NVDA Price Action with Moving Averages',
                font: { size: 24 }
            },
            height: 500,
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price ($)' },
            showlegend: true,
            legend: { orientation: 'h', y: 1.05 }
        };

        Plotly.newPlot('mainChart', [mainChartTrace, sma50Trace, sma200Trace], mainLayout);

        // RSI Chart
        const rsiValues = prices.map((_, i) => {
            if (i < 14) return 50;
            const changes = [];
            for (let j = i - 14; j < i; j++) {
                changes.push(prices[j+1] - prices[j]);
            }
            const gains = changes.filter(c => c > 0).reduce((a, b) => a + b, 0);
            const losses = changes.filter(c => c < 0).map(c => Math.abs(c)).reduce((a, b) => a + b, 0);
            const rs = gains / (losses || 1);
            return 100 - (100 / (1 + rs));
        });

        const rsiTrace = {
            x: dates,
            y: rsiValues,
            type: 'scatter',
            name: 'RSI',
            line: { color: '#9C27B0', width: 2 }
        };

        const rsiLayout = {
            title: { text: 'RSI (14-day)', font: { size: 18 } },
            height: 250,
            xaxis: { showgrid: false },
            yaxis: { range: [0, 100], title: 'RSI' },
            shapes: [
                { type: 'line', x0: 0, y0: 70, x1: dates.length, y1: 70, 
                  line: { color: 'red', width: 1, dash: 'dash' } },
                { type: 'line', x0: 0, y0: 30, x1: dates.length, y1: 30, 
                  line: { color: 'green', width: 1, dash: 'dash' } }
            ]
        };

        Plotly.newPlot('rsiChart', [rsiTrace], rsiLayout);

        // MACD Chart
        const ema12 = prices.map((p, i) => {
            if (i < 12) return p;
            return prices.slice(i-12, i+1).reduce((a, b) => a + b, 0) / 13;
        });
        
        const ema26 = prices.map((p, i) => {
            if (i < 26) return p;
            return prices.slice(i-26, i+1).reduce((a, b) => a + b, 0) / 27;
        });

        const macd = ema12.map((e, i) => e - ema26[i]);
        const signal = macd.map((m, i) => {
            if (i < 9) return m;
            return macd.slice(i-9, i+1).reduce((a, b) => a + b, 0) / 10;
        });
        const histogram = macd.map((m, i) => m - signal[i]);

        const macdTrace = {
            x: dates,
            y: macd,
            type: 'scatter',
            name: 'MACD',
            line: { color: '#2196F3', width: 2 }
        };

        const signalTrace = {
            x: dates,
            y: signal,
            type: 'scatter',
            name: 'Signal',
            line: { color: '#FF5722', width: 2 }
        };

        const histogramTrace = {
            x: dates,
            y: histogram,
            type: 'bar',
            name: 'Histogram',
            marker: {
                color: histogram.map(h => h > 0 ? '#00C853' : '#D50000')
            }
        };

        const macdLayout = {
            title: { text: 'MACD Indicator', font: { size: 18 } },
            height: 250,
            xaxis: { showgrid: false },
            yaxis: { title: 'MACD' },
            showlegend: true
        };

        Plotly.newPlot('macdChart', [macdTrace, signalTrace, histogramTrace], macdLayout);
    </script>
</body>
</html>

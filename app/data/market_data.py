import pandas as pd
import numpy as np
import datetime
import logging
from app.data.market_snapshot import MarketSnapshot
from app.collectors.yfinance_collector import (
    fetch_ohlcv_dataframe,
    fetch_fundamentals_dict,
)

logger = logging.getLogger(__name__)


async def build_snapshot(ticker: str, lookback_days: int = 60) -> MarketSnapshot:
    """
    Builds a MarketSnapshot by fetching OHLCV data and computing technicals/quant metrics.
    No LLM calls are made.
    """
    from app.data.market_data_store import get_latest_snapshot, save_snapshot

    # Check cache first
    cached = get_latest_snapshot(ticker)
    if cached:
        logger.info(f"Using cached market snapshot for {ticker}")
        return cached

    logger.info(f"Building market snapshot for {ticker}")

    # Fetch data
    df = await fetch_ohlcv_dataframe(ticker, period=f"{lookback_days}d")
    info = await fetch_fundamentals_dict(ticker) or {}

    # Start assembling fields
    kwargs = {
        "ticker": ticker,
        "fetched_at": datetime.datetime.now(datetime.UTC),
        "data_source": "yfinance",
        "candles_used": len(df) if df is not None else 0,
        # Price
        "price": None,
        "open": None,
        "high": None,
        "low": None,
        "volume": None,
        "vwap": None,
        # Technicals
        "rsi_14": None,
        "macd": None,
        "macd_signal": None,
        "macd_hist": None,
        "bb_upper": None,
        "bb_lower": None,
        "bb_pct": None,
        "sma_20": None,
        "sma_50": None,
        "sma_200": None,
        "atr_14": None,
        "adx_14": None,
        "stoch_k": None,
        "stoch_d": None,
        # Quant
        "returns_1d": None,
        "returns_5d": None,
        "returns_20d": None,
        "volatility_20d": None,
        "sharpe_20d": None,
        "max_drawdown_20d": None,
        "beta_20d": None,
        # Fundamentals
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps") or info.get("epsTrailingTwelveMonths"),
        "market_cap": info.get("marketCap"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margin": info.get("profitMargins"),
        "debt_to_equity": info.get("debtToEquity"),
    }

    if df is not None and not df.empty:
        # Latest price info
        latest = df.iloc[-1]
        kwargs["price"] = float(latest.get("Close", 0)) or None
        kwargs["open"] = float(latest.get("Open", 0)) or None
        kwargs["high"] = float(latest.get("High", 0)) or None
        kwargs["low"] = float(latest.get("Low", 0)) or None
        kwargs["volume"] = int(latest.get("Volume", 0)) if "Volume" in latest else None

        # We need a proper index for pandas_ta (DatetimeIndex)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 1. Basic Technicals via pandas-ta
        try:
            rsi = df.ta.rsi(length=14)
            if rsi is not None and not rsi.empty:
                kwargs["rsi_14"] = float(rsi.dropna().iloc[-1])
        except Exception:
            pass

        try:
            macd_df = df.ta.macd(fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                macd_df = macd_df.dropna()
                if not macd_df.empty:
                    kwargs["macd"] = float(macd_df.iloc[-1, 0])
                    kwargs["macd_hist"] = float(macd_df.iloc[-1, 1])
                    kwargs["macd_signal"] = float(macd_df.iloc[-1, 2])
        except Exception:
            pass

        try:
            bb_df = df.ta.bbands(length=20, std=2)
            if bb_df is not None and not bb_df.empty:
                bb_df = bb_df.dropna()
                if not bb_df.empty:
                    kwargs["bb_lower"] = float(bb_df.iloc[-1, 0])
                    kwargs["bb_upper"] = float(bb_df.iloc[-1, 2])
                    kwargs["bb_pct"] = (
                        float(bb_df.iloc[-1, 4]) if bb_df.shape[1] > 4 else None
                    )
        except Exception:
            pass

        try:
            sma_20 = df.ta.sma(length=20)
            if sma_20 is not None and not sma_20.empty:
                kwargs["sma_20"] = float(sma_20.dropna().iloc[-1])
        except Exception:
            pass
        try:
            sma_50 = df.ta.sma(length=50)
            if sma_50 is not None and not sma_50.empty:
                kwargs["sma_50"] = float(sma_50.dropna().iloc[-1])
        except Exception:
            pass
        try:
            sma_200 = df.ta.sma(length=200)
            if sma_200 is not None and not sma_200.empty:
                kwargs["sma_200"] = float(sma_200.dropna().iloc[-1])
        except Exception:
            pass

        try:
            atr = df.ta.atr(length=14)
            if atr is not None and not atr.empty:
                kwargs["atr_14"] = float(atr.dropna().iloc[-1])
        except Exception:
            pass

        try:
            adx_df = df.ta.adx(length=14)
            if adx_df is not None and not adx_df.empty:
                adx_df = adx_df.dropna()
                if not adx_df.empty:
                    kwargs["adx_14"] = float(adx_df.iloc[-1, 0])
        except Exception:
            pass

        try:
            stoch_df = df.ta.stoch()
            if stoch_df is not None and not stoch_df.empty:
                stoch_df = stoch_df.dropna()
                if not stoch_df.empty:
                    kwargs["stoch_k"] = float(stoch_df.iloc[-1, 0])
                    kwargs["stoch_d"] = float(stoch_df.iloc[-1, 1])
        except Exception:
            pass

        try:
            vwap = df.ta.vwap()
            if vwap is not None and not vwap.empty:
                vwap = vwap.dropna()
                if not vwap.empty:
                    kwargs["vwap"] = float(vwap.iloc[-1])
        except Exception:
            pass

        # 2. Quant Metrics
        try:
            close_prices = df["Close"]
            daily_returns = close_prices.pct_change()

            if len(close_prices) >= 2:
                kwargs["returns_1d"] = float(
                    close_prices.iloc[-1] / close_prices.iloc[-2] - 1
                )
            if len(close_prices) >= 6:
                kwargs["returns_5d"] = float(
                    close_prices.iloc[-1] / close_prices.iloc[-6] - 1
                )
            if len(close_prices) >= 21:
                kwargs["returns_20d"] = float(
                    close_prices.iloc[-1] / close_prices.iloc[-21] - 1
                )

            # Volatility (Annualized over 20 days)
            recent_20_returns = daily_returns.tail(20)
            if len(recent_20_returns.dropna()) >= 10:
                kwargs["volatility_20d"] = float(recent_20_returns.std() * np.sqrt(252))

                # Sharpe Ratio (Assume 0% risk free for simple calculation)
                mean_return = recent_20_returns.mean()
                std_return = recent_20_returns.std()
                if std_return > 0:
                    kwargs["sharpe_20d"] = float(
                        (mean_return * 252) / (std_return * np.sqrt(252))
                    )

            # Max Drawdown over last 20 days
            recent_20_prices = close_prices.tail(20)
            if len(recent_20_prices) >= 10:
                roll_max = recent_20_prices.cummax()
                drawdown = recent_20_prices / roll_max - 1.0
                kwargs["max_drawdown_20d"] = float(drawdown.min())

            # Beta
            kwargs["beta_20d"] = info.get(
                "beta"
            )  # Just use fundamental beta as fallback, accurate beta needs SPY data

        except Exception as e:
            logger.warning(f"Error computing quant metrics for {ticker}: {e}")

    # Remove nans to avoid downstream issues
    for k, v in kwargs.items():
        if isinstance(v, float) and np.isnan(v):
            kwargs[k] = None

    snapshot = MarketSnapshot(**kwargs)
    try:
        from app.data.market_data_store import save_snapshot

        save_snapshot(snapshot)
    except Exception as e:
        logger.warning(f"Error saving snapshot for {ticker}: {e}")

    return snapshot

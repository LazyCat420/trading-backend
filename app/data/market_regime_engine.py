import logging
import pandas as pd
import yfinance as yf
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def get_latest_regime():
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM market_regime ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in db.description]
    return dict(zip(cols, row))


def get_sector_breadth_data():
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM sector_breadth WHERE date = (SELECT MAX(date) FROM sector_breadth)"
        ).fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


def get_cross_correlations(period: str):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM cross_asset_correlations WHERE period = %s", (period,)
        ).fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


async def detect_anomalies():
    with get_db() as db:
        try:
            query = "SELECT * FROM global.anomalies ORDER BY detected_at DESC LIMIT 10"
            rows = db.execute(query).fetchall()
            if not rows:
                return []
            cols = [desc[0] for desc in db.description]
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []


async def compute_sector_breadth():
    logger.info("Computing sector breadth...")
    with get_db() as db:
        query = """
            SELECT p.ticker, p.date, p.close, t.sector
            FROM price_history p
            JOIN ticker_metadata t ON p.ticker = t.ticker
            WHERE t.sp500 = TRUE AND p.source = 'yfinance'
            ORDER BY p.date ASC
        """
        cursor = db.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        import pandas as pd

        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return

        df["date"] = pd.to_datetime(df["date"])

        df["sma50"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(50).mean()
        )
        df["sma200"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(200).mean()
        )
        df["high_252"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(252).max()
        )
        df["low_252"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(252).min()
        )

        df["above_50"] = df["close"] > df["sma50"]
        df["above_200"] = df["close"] > df["sma200"]
        df["is_new_high"] = df["close"] >= df["high_252"]
        df["is_new_low"] = df["close"] <= df["low_252"]

        latest_date = df["date"].max()
        latest_df = df[df["date"] == latest_date]

        sectors = latest_df["sector"].unique()
        inserts = []

        for sector in sectors:
            sdf = latest_df[latest_df["sector"] == sector]
            if sdf.empty:
                continue

            pct_above_50 = sdf["above_50"].mean() * 100
            pct_above_200 = sdf["above_200"].mean() * 100
            new_highs = sdf["is_new_high"].sum()
            new_lows = sdf["is_new_low"].sum()
            net_highs = new_highs - new_lows

            inserts.append(
                (
                    sector,
                    latest_date.strftime("%Y-%m-%d"),
                    float(pct_above_50) if pd.notna(pct_above_50) else 0.0,
                    float(pct_above_200) if pd.notna(pct_above_200) else 0.0,
                    int(new_highs),
                    int(new_lows),
                    int(net_highs),
                )
            )

        if inserts:
            query_ins = """
                INSERT INTO sector_breadth (sector, date, pct_above_sma50, pct_above_sma200, new_highs, new_lows, net_highs, computed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (sector, date) DO UPDATE SET
                    pct_above_sma50 = EXCLUDED.pct_above_sma50,
                    pct_above_sma200 = EXCLUDED.pct_above_sma200,
                    new_highs = EXCLUDED.new_highs,
                    new_lows = EXCLUDED.new_lows,
                    net_highs = EXCLUDED.net_highs,
                    computed_at = CURRENT_TIMESTAMP
            """
            for item in inserts:
                db.execute(query_ins, item)

async def compute_market_regime():
    logger.info("Computing market regime...")

    tickers = ["^VIX", "^TNX", "DX-Y.NYB", "SPY"]
    try:
        data = yf.download(tickers, period="1mo", progress=False)["Close"]
        if data.empty:
            return

        latest = data.iloc[-1]
        prev_5d = data.iloc[-5] if len(data) >= 5 else data.iloc[0]

        vix_level = float(latest.get("^VIX", 15.0))
        yield_10y = float(latest.get("^TNX", 4.0))
        dollar_index = float(latest.get("DX-Y.NYB", 100.0))
        sp500_level = float(latest.get("SPY", 5000.0))

        prev_spy = float(prev_5d.get("SPY", sp500_level))
        prev_dollar = float(prev_5d.get("DX-Y.NYB", dollar_index))

        sp500_change_5d = (
            ((sp500_level - prev_spy) / prev_spy) * 100 if prev_spy > 0 else 0.0
        )
        dollar_change_5d = (
            ((dollar_index - prev_dollar) / prev_dollar) * 100
            if prev_dollar > 0
            else 0.0
        )

        if vix_level > 25:
            regime_label = "Crisis"
        elif vix_level > 20:
            regime_label = "Risk-Off"
        elif sp500_change_5d > 1.0:
            regime_label = "Risk-On"
        else:
            regime_label = "Neutral"

        inserts = [
            (
                pd.Timestamp.now().strftime("%Y-%m-%d"),
                vix_level,
                "Elevated" if vix_level > 20 else "Normal",
                0.0,
                1.0,
                "Normal",
                yield_10y,
                yield_10y,
                0.0,
                "Normal",
                dollar_index,
                dollar_change_5d,
                sp500_level,
                sp500_change_5d,
                50.0,
                regime_label,
            )
        ]

        with get_db() as db:
            query_ins = """
                INSERT INTO market_regime (date, vix_level, vix_signal, vix_zscore, vix_term_ratio, vix_term_signal, yield_2y, yield_10y, yield_2y10y_spread, yield_signal, dollar_index, dollar_change_5d, sp500_level, sp500_change_5d, breadth_sp500, regime_label, computed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (date) DO UPDATE SET
                    vix_level = EXCLUDED.vix_level,
                    vix_signal = EXCLUDED.vix_signal,
                    dollar_index = EXCLUDED.dollar_index,
                    sp500_level = EXCLUDED.sp500_level,
                    regime_label = EXCLUDED.regime_label,
                    sp500_change_5d = EXCLUDED.sp500_change_5d,
                    computed_at = CURRENT_TIMESTAMP
            """
            for item in inserts:
                db.execute(query_ins, item)

    except Exception as e:
        logger.error(f"Error computing market regime: {e}")


async def compute_cross_asset_correlations():
    logger.info("Computing cross asset correlations...")
    # Safe fallback if not fully implemented yet
    pass

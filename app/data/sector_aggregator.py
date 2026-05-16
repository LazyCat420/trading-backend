import logging
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def get_sector_heatmap():
    with get_db() as db:
        row = db.execute("SELECT MAX(date) FROM sector_performance").fetchone()
        if not row or not row[0]:
            return []

        latest_date = row[0]
        rows = db.execute(
            "SELECT * FROM sector_performance WHERE date = %s", (latest_date,)
        ).fetchall()

        if not rows:
            return []

        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


def get_sector_stocks(sector: str):
    with get_db() as db:
        query = """
            WITH latest_date AS (
                SELECT MAX(date) as max_date FROM price_history
            ),
            latest_prices AS (
                SELECT p.ticker, p.close, p.volume
                FROM price_history p
                JOIN latest_date l ON p.date = l.max_date
            ),
            prev_prices AS (
                SELECT p.ticker, p.close as prev_close
                FROM price_history p
                JOIN latest_date l ON p.date = (l.max_date - INTERVAL '1 day')::date
            )
            SELECT 
                t.ticker, t.name, t.market_cap, t.industry,
                l.close as price, l.volume,
                CASE WHEN p.prev_close > 0 THEN ((l.close - p.prev_close) / p.prev_close) * 100 ELSE 0 END as return_1d
            FROM ticker_metadata t
            LEFT JOIN latest_prices l ON t.ticker = l.ticker
            LEFT JOIN prev_prices p ON t.ticker = p.ticker
            WHERE t.sector = %s AND t.sp500 = TRUE
            ORDER BY t.market_cap DESC NULLS LAST
        """
        rows = db.execute(query, (sector,)).fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


async def compute_sector_performance():
    logger.info("Computing sector performance...")
    with get_db() as db:
        # Load all price history and metadata into pandas for vectorized operations
        query = """
            SELECT p.ticker, p.date, p.close, p.volume, t.sector, t.market_cap
            FROM price_history p
            JOIN ticker_metadata t ON p.ticker = t.ticker
            WHERE t.sp500 = TRUE AND t.sector IS NOT NULL
            ORDER BY p.date ASC
        """
        cursor = db.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        import pandas as pd

        df = pd.DataFrame(rows, columns=cols)

        if df.empty:
            logger.warning("No price data found. Cannot compute sector performance.")
            return "No data"

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Calculate daily returns per ticker
        df["return_1d"] = df.groupby("ticker")["close"].pct_change()
        df["return_5d"] = df.groupby("ticker")["close"].pct_change(periods=5)
        df["return_30d"] = df.groupby("ticker")["close"].pct_change(periods=30)

        # For volume, maybe calculate ratio compared to 30d SMA
        df["vol_sma_30"] = df.groupby("ticker")["volume"].transform(
            lambda x: x.rolling(30).mean()
        )
        df["vol_ratio"] = df["volume"] / df["vol_sma_30"]

        # We only care about the latest date for each sector
        latest_date = df["date"].max()
        latest_df = df[df["date"] == latest_date].copy()

        if latest_df.empty:
            return "No recent data"

        sectors = latest_df["sector"].unique()
        inserts = []

        for sector in sectors:
            sdf = latest_df[latest_df["sector"] == sector]

            # Calculate market cap weights
            total_mcap = sdf["market_cap"].sum()
            if total_mcap > 0:
                weights = sdf["market_cap"] / total_mcap
                avg_1d = (sdf["return_1d"] * weights).sum() * 100
                avg_5d = (sdf["return_5d"] * weights).sum() * 100
                avg_30d = (sdf["return_30d"] * weights).sum() * 100
            else:
                avg_1d = sdf["return_1d"].mean() * 100
                avg_5d = sdf["return_5d"].mean() * 100
                avg_30d = sdf["return_30d"].mean() * 100

            # Breadth (pct of stocks positive over 1d)
            breadth_pct = (sdf["return_1d"] > 0).mean() * 100

            # Top gainer / loser (1d)
            if not sdf.empty and not sdf["return_1d"].isna().all():
                top_gainer_row = sdf.loc[sdf["return_1d"].idxmax()]
                top_loser_row = sdf.loc[sdf["return_1d"].idxmin()]
                top_gainer = top_gainer_row["ticker"]
                top_gainer_return = top_gainer_row["return_1d"] * 100
                top_loser = top_loser_row["ticker"]
                top_loser_return = top_loser_row["return_1d"] * 100
            else:
                top_gainer, top_gainer_return, top_loser, top_loser_return = (
                    None,
                    0,
                    None,
                    0,
                )

            avg_volume_ratio = sdf["vol_ratio"].mean()
            stock_count = len(sdf)

            # Momentum signal based on 5d
            momentum_signal = (
                "Bullish"
                if avg_5d > 1.0
                else ("Bearish" if avg_5d < -1.0 else "Neutral")
            )

            inserts.append(
                (
                    sector,
                    latest_date.strftime("%Y-%m-%d"),
                    float(avg_1d) if pd.notna(avg_1d) else 0.0,
                    float(avg_5d) if pd.notna(avg_5d) else 0.0,
                    float(avg_30d) if pd.notna(avg_30d) else 0.0,
                    float(breadth_pct) if pd.notna(breadth_pct) else 0.0,
                    top_gainer,
                    float(top_gainer_return) if pd.notna(top_gainer_return) else 0.0,
                    top_loser,
                    float(top_loser_return) if pd.notna(top_loser_return) else 0.0,
                    float(avg_volume_ratio) if pd.notna(avg_volume_ratio) else 1.0,
                    momentum_signal,
                    stock_count,
                )
            )

        query = """
            INSERT INTO sector_performance 
            (sector, date, avg_return_1d, avg_return_5d, avg_return_30d, breadth_pct, 
             top_gainer, top_gainer_return, top_loser, top_loser_return, 
             avg_volume_ratio, momentum_signal, stock_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (sector, date) DO UPDATE SET
                avg_return_1d = EXCLUDED.avg_return_1d,
                avg_return_5d = EXCLUDED.avg_return_5d,
                avg_return_30d = EXCLUDED.avg_return_30d,
                breadth_pct = EXCLUDED.breadth_pct,
                top_gainer = EXCLUDED.top_gainer,
                top_gainer_return = EXCLUDED.top_gainer_return,
                top_loser = EXCLUDED.top_loser,
                top_loser_return = EXCLUDED.top_loser_return,
                avg_volume_ratio = EXCLUDED.avg_volume_ratio,
                momentum_signal = EXCLUDED.momentum_signal,
                stock_count = EXCLUDED.stock_count,
                computed_at = CURRENT_TIMESTAMP
        """

        for item in inserts:
            db.execute(query, item)

    logger.info(f"Computed and saved performance for {len(inserts)} sectors.")
    return f"Processed {len(inserts)} sectors"


async def backfill_sector_performance():
    """
    Derives daily sector avg_return_1d from price_history + ticker_metadata
    and writes to sector_performance. Only calculates historical 1-day returns
    to quickly backfill empty history needed by the correlation engine.
    """
    logger.info("Backfilling sector_performance from historical price_history...")
    with get_db() as db:
        # Check if we already have sufficient history (e.g. more than 1 day)
        row = db.execute("SELECT COUNT(DISTINCT date) FROM sector_performance").fetchone()
        if row and row[0] > 1:
            logger.info("Sector performance already has history. Skipping backfill.")
            return

        query = """
            SELECT p.ticker, p.date, p.close, t.sector
            FROM price_history p
            JOIN ticker_metadata t ON p.ticker = t.ticker
            WHERE t.sp500 = TRUE
              AND p.source = 'yfinance'
              AND t.sector IS NOT NULL
            ORDER BY p.ticker, p.date ASC
        """
        cursor = db.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        
        if not rows:
            logger.warning("backfill_sector_performance: no price_history rows found, skipping.")
            return
            
        import pandas as pd
        df = pd.DataFrame(rows, columns=cols)

    if df.empty:
        logger.warning("backfill_sector_performance: no price_history rows found, skipping.")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    # Compute 1-day % return per ticker
    df["return_1d"] = df.groupby("ticker")["close"].pct_change()

    # Aggregate to daily sector average return
    sector_daily = (
        df.groupby(["sector", "date"])["return_1d"]
        .mean()
        .reset_index()
        .rename(columns={"return_1d": "avg_return_1d"})
    )
    sector_daily = sector_daily.dropna(subset=["avg_return_1d"])

    inserts = [
        (row["sector"], row["date"].strftime("%Y-%m-%d"), float(row["avg_return_1d"]))
        for _, row in sector_daily.iterrows()
    ]

    if not inserts:
        logger.warning("backfill_sector_performance: computed 0 rows, check price_history data.")
        return

    with get_db() as db:
        db.executemany(
            """
            INSERT INTO sector_performance (sector, date, avg_return_1d, computed_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (sector, date) DO UPDATE SET
                avg_return_1d = EXCLUDED.avg_return_1d,
                computed_at = CURRENT_TIMESTAMP
            """,
            inserts,
        )

    logger.info(
        "backfill_sector_performance: inserted %d sector-day rows across %d sectors.",
        len(inserts),
        sector_daily["sector"].nunique(),
    )

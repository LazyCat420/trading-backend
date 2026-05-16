import logging
import pandas as pd
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def get_sector_correlation_map(period: str):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM sector_correlations WHERE period = %s", (period,)
        ).fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


def get_inverse_sector_pairs(period: str):
    with get_db() as db:
        query = "SELECT * FROM sector_correlations WHERE period = %s AND correlation < -0.4 ORDER BY correlation ASC"
        rows = db.execute(query, (period,)).fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


def get_commodity_sector_links(commodity: str):
    with get_db() as db:
        query = "SELECT * FROM stock_commodity_correlations WHERE commodity = %s AND (correlation > 0.3 OR correlation < -0.3) ORDER BY correlation DESC"
        rows = db.execute(query, (commodity,)).fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


async def compute_all_correlations():
    logger.info("Computing all correlations...")
    inserts = []
    comm_inserts = []
    with get_db() as db:
        # 1. Sector-Pair Correlations
        query_sector = """
            SELECT sector, date, avg_return_1d
            FROM sector_performance
            ORDER BY date ASC
        """
        cursor = db.execute(query_sector)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        df_sector = pd.DataFrame(rows, columns=cols)

        if not df_sector.empty:
            df_sector["date"] = pd.to_datetime(df_sector["date"])
            pivot_sector = df_sector.pivot(
                index="date", columns="sector", values="avg_return_1d"
            )

            periods = {"30d": 30, "90d": 90}

            for period_name, days in periods.items():
                recent_data = pivot_sector.tail(days)
                if len(recent_data) < days * 0.5:
                    continue

                corr_matrix = recent_data.corr()
                sectors = corr_matrix.columns

                for i in range(len(sectors)):
                    for j in range(i + 1, len(sectors)):
                        sec_a = sectors[i]
                        sec_b = sectors[j]
                        corr = corr_matrix.loc[sec_a, sec_b]

                        if pd.isna(corr):
                            continue

                        tier = "neutral"
                        if corr > 0.8:
                            tier = "highly_correlated"
                        elif corr > 0.5:
                            tier = "correlated"
                        elif corr < -0.5:
                            tier = "inversely_correlated"
                        elif corr < -0.2:
                            tier = "weakly_inversely_correlated"

                        inserts.append(
                            (
                                sec_a,
                                sec_b,
                                float(corr),
                                tier,
                                period_name,
                                len(recent_data),
                            )
                        )

            if inserts:
                query_ins = """
                    INSERT INTO sector_correlations (sector_a, sector_b, correlation, tier, period, data_points, computed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (sector_a, sector_b, period) DO UPDATE SET
                        correlation = EXCLUDED.correlation,
                        tier = EXCLUDED.tier,
                        data_points = EXCLUDED.data_points,
                        computed_at = CURRENT_TIMESTAMP
                """
                db.executemany(query_ins, inserts)

        # 2. Stock vs Commodity Correlations
        query_stock = """
            SELECT p.ticker, p.date, p.close as stock_price, t.sector
            FROM price_history p
            JOIN ticker_metadata t ON p.ticker = t.ticker
            WHERE t.sp500 = TRUE AND p.source = 'yfinance'
        """
        cursor = db.execute(query_stock)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        df_stocks = pd.DataFrame(rows, columns=cols)

        query_comm = """
            SELECT symbol as commodity, date, close as comm_price
            FROM asset_prices
            WHERE asset_class = 'commodity'
        """
        cursor = db.execute(query_comm)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description] if cursor.description else []
        df_comms = pd.DataFrame(rows, columns=cols)

        if df_stocks.empty or df_comms.empty:
            logger.warning(
                "Missing stock or commodity data. Skipping commodity correlations."
            )
        else:
            df_stocks["date"] = pd.to_datetime(df_stocks["date"])
            df_comms["date"] = pd.to_datetime(df_comms["date"])

            pivot_stocks = df_stocks.pivot(
                index="date", columns="ticker", values="stock_price"
            ).pct_change()
            pivot_comms = df_comms.pivot(
                index="date", columns="commodity", values="comm_price"
            ).pct_change()

            joined = pivot_stocks.join(pivot_comms, how="inner")

            periods = {"30d": 30, "90d": 90}

            for period_name, days in periods.items():
                recent_data = joined.tail(days)
                if len(recent_data) < days * 0.5:
                    continue

                tickers = pivot_stocks.columns
                commodities = pivot_comms.columns

                for comm in commodities:
                    if comm not in recent_data.columns:
                        continue
                    for ticker in tickers:
                        if ticker not in recent_data.columns:
                            continue

                        valid = recent_data[[ticker, comm]].dropna()
                        if len(valid) < 10:
                            continue

                        corr = valid[ticker].corr(valid[comm])
                        if pd.isna(corr):
                            continue

                        sensitivity = "neutral"
                        if corr > 0.4:
                            sensitivity = "highly_sensitive"
                        elif corr < -0.4:
                            sensitivity = "inversely_sensitive"

                        # Store all calculated correlations
                        comm_inserts.append(
                            (
                                ticker,
                                comm,
                                float(corr),
                                sensitivity,
                                period_name,
                                len(valid),
                            )
                        )

            if comm_inserts:
                query_comm_ins = """
                    INSERT INTO stock_commodity_correlations (ticker, commodity, correlation, sensitivity, period, data_points, computed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (ticker, commodity, period) DO UPDATE SET
                        correlation = EXCLUDED.correlation,
                        sensitivity = EXCLUDED.sensitivity,
                        data_points = EXCLUDED.data_points,
                        computed_at = CURRENT_TIMESTAMP
                """
                db.executemany(query_comm_ins, comm_inserts)

    logger.info(
        f"Computed {len(inserts)} sector correlations and {len(comm_inserts)} commodity correlations."
    )
    return f"Computed {len(inserts)} sector & {len(comm_inserts)} comm correlations"

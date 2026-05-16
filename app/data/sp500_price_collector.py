import logging
import pandas as pd
import yfinance as yf
from app.db.connection import get_db

logger = logging.getLogger(__name__)


async def collect_sp500_prices(period: str = "6mo"):
    """
    Batch downloads historical prices for all S&P 500 tickers from ticker_metadata.
    """
    logger.info(f"Batch downloading S&P 500 prices for period: {period}...")

    with get_db() as db:
        rows = db.execute(
            "SELECT ticker FROM ticker_metadata WHERE sp500 = TRUE"
        ).fetchall()
        if not rows:
            logger.error(
                "No S&P 500 tickers found in ticker_metadata. Run load_sp500_universe first."
            )
            return

        tickers = [row[0] for row in rows]

        # yfinance batch download is faster but can be fragile. We'll do a robust batch.
        # We download in chunks of 100 to avoid URL too long or memory spikes.
        chunk_size = 100
        inserts = []

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            logger.info(f"Downloading prices for batch {i // chunk_size + 1}...")

            try:
                # We use group_by='ticker' to get a clean MultiIndex
                data = yf.download(
                    chunk, period=period, group_by="ticker", progress=False
                )

                # yf.download returns different structures depending on if 1 or multiple tickers are provided
                if len(chunk) == 1:
                    ticker = chunk[0]
                    for date, row in data.iterrows():
                        if pd.isna(row["Close"]):
                            continue
                        inserts.append(
                            (
                                ticker,
                                date.strftime("%Y-%m-%d"),
                                float(row["Open"])
                                if not pd.isna(row["Open"])
                                else None,
                                float(row["High"])
                                if not pd.isna(row["High"])
                                else None,
                                float(row["Low"]) if not pd.isna(row["Low"]) else None,
                                float(row["Close"])
                                if not pd.isna(row["Close"])
                                else None,
                                int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                                "yfinance",
                            )
                        )
                else:
                    for ticker in chunk:
                        # Depending on yfinance version, columns might have ticker on level 0 or level 1.
                        # Usually if group_by='ticker', level 0 is ticker.
                        if ticker not in data.columns.levels[0]:
                            continue

                        ticker_data = data[ticker]
                        for date, row in ticker_data.iterrows():
                            # Use pd.isna safely on series values
                            # yfinance latest version might return a Series for Close instead of a scalar if not grouped correctly, but this should be fine.
                            try:
                                # Using .iloc or accessing by name
                                close_val = row.get("Close")
                                if close_val is None or pd.isna(close_val):
                                    continue

                                open_val = row.get("Open")
                                high_val = row.get("High")
                                low_val = row.get("Low")
                                volume_val = row.get("Volume")

                                inserts.append(
                                    (
                                        ticker,
                                        date.strftime("%Y-%m-%d"),
                                        float(open_val) if pd.notna(open_val) else None,
                                        float(high_val) if pd.notna(high_val) else None,
                                        float(low_val) if pd.notna(low_val) else None,
                                        float(close_val)
                                        if pd.notna(close_val)
                                        else None,
                                        int(volume_val) if pd.notna(volume_val) else 0,
                                        "yfinance",
                                    )
                                )
                            except Exception as inner_e:
                                # Skip row if weird format
                                pass
            except Exception as e:
                logger.error(f"Error downloading chunk starting with {chunk[0]}: {e}")

        if inserts:
            logger.info(f"Inserting {len(inserts)} price records into the database...")
            query = """
                INSERT INTO price_history (ticker, date, open, high, low, close, volume, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, date, source) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """
            # Execute individually as our PooledCursor doesn't expose executemany directly
            count = 0
            for item in inserts:
                try:
                    db.execute(query, item)
                    count += 1
                except Exception as e:
                    pass

            logger.info(f"Successfully collected and saved {count} price records.")
        else:
            logger.warning("No price data collected.")

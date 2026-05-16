import logging
import uuid
import json
from app.db.connection import get_db

logger = logging.getLogger(__name__)


def get_active_rotations(limit: int = 10):
    with get_db() as db:
        query = (
            "SELECT * FROM sector_rotation_signals ORDER BY detected_at DESC LIMIT %s"
        )
        rows = db.execute(query, (limit,)).fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


def get_divergent_stocks(sector: str, min_divergence: float = 3.0):
    with get_db() as db:
        # Find stocks whose 5d return is min_divergence% different from the sector's 5d return
        query = """
            WITH latest_date AS (
                SELECT MAX(date) as max_date FROM price_history
            ),
            latest_prices AS (
                SELECT p.ticker, p.close
                FROM price_history p
                JOIN latest_date l ON p.date = l.max_date
            ),
            prev_prices AS (
                SELECT p.ticker, p.close as prev_close
                FROM price_history p
                JOIN latest_date l ON p.date = (l.max_date - INTERVAL '5 days')::date
            ),
            stock_returns AS (
                SELECT 
                    t.ticker, t.name, t.sector,
                    CASE WHEN p.prev_close > 0 THEN ((l.close - p.prev_close) / p.prev_close) * 100 ELSE 0 END as return_5d
                FROM ticker_metadata t
                JOIN latest_prices l ON t.ticker = l.ticker
                JOIN prev_prices p ON t.ticker = p.ticker
                WHERE t.sector = %s AND t.sp500 = TRUE
            ),
            sector_ret AS (
                SELECT avg_return_5d 
                FROM sector_performance 
                WHERE sector = %s AND date = (SELECT MAX(date) FROM sector_performance)
            )
            SELECT s.ticker, s.name, s.return_5d, r.avg_return_5d as sector_return_5d,
                   (s.return_5d - r.avg_return_5d) as divergence
            FROM stock_returns s, sector_ret r
            WHERE ABS(s.return_5d - r.avg_return_5d) >= %s
            ORDER BY ABS(s.return_5d - r.avg_return_5d) DESC
        """
        rows = db.execute(query, (sector, sector, min_divergence)).fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in db.description]
    return [dict(zip(cols, r)) for r in rows]


async def detect_rotations():
    logger.info("Detecting sector rotations...")
    with get_db() as db:
        # Check the latest 5d returns of all sectors
        query = """
            WITH latest_date AS (
                SELECT MAX(date) as max_date FROM sector_performance
            )
            SELECT s.sector, s.avg_return_5d
            FROM sector_performance s
            JOIN latest_date l ON s.date = l.max_date
        """
        rows = db.execute(query).fetchall()
        if not rows:
            return "No sector performance data"

        sector_returns = {r[0]: r[1] for r in rows if r[1] is not None}

        # Get inversely correlated pairs to check for rotation
        pairs_query = "SELECT sector_a, sector_b, correlation FROM sector_correlations WHERE period = '30d' AND correlation < -0.4"
        pairs = db.execute(pairs_query).fetchall()

        inserts = []

        for sec_a, sec_b, corr in pairs:
            ret_a = sector_returns.get(sec_a, 0)
            ret_b = sector_returns.get(sec_b, 0)

            # Check if they are moving in opposite directions significantly
            if ret_a > 2.0 and ret_b < -2.0:
                # Money flowing from B to A
                evidence = {
                    "reason": f"{sec_b} down {ret_b:.2f}%, {sec_a} up {ret_a:.2f}%",
                    "correlation": corr,
                }
                inserts.append(
                    (
                        str(uuid.uuid4()),
                        sec_b,
                        sec_a,
                        float(ret_b),
                        float(ret_a),
                        float(corr),
                        None,
                        "High",
                        json.dumps(evidence),
                    )
                )
            elif ret_b > 2.0 and ret_a < -2.0:
                # Money flowing from A to B
                evidence = {
                    "reason": f"{sec_a} down {ret_a:.2f}%, {sec_b} up {ret_b:.2f}%",
                    "correlation": corr,
                }
                inserts.append(
                    (
                        str(uuid.uuid4()),
                        sec_a,
                        sec_b,
                        float(ret_a),
                        float(ret_b),
                        float(corr),
                        None,
                        "High",
                        json.dumps(evidence),
                    )
                )

        if inserts:
            query_ins = """
                INSERT INTO sector_rotation_signals (id, from_sector, to_sector, from_return_5d, to_return_5d, correlation, commodity_trigger, confidence, evidence_json, detected_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
            for item in inserts:
                db.execute(query_ins, item)

    logger.info(f"Detected {len(inserts)} rotation signals.")
    return f"Detected {len(inserts)} rotations"

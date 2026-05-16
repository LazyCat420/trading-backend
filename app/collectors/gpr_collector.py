"""
GPR Collector — Caldara-Iacoviello Geopolitical Risk Index.

Pure data collector. No LLM calls.
Writes to: global.gpr_index
No API key required — downloads published Excel data.

The GPR Index is constructed by counting newspaper articles related to
geopolitical tensions across 10 major international newspapers.
Published by Matteo Iacoviello (Federal Reserve Board).

Two sub-indexes:
  - GPRD_THREAT: war threats, peace threats, military buildups, nuclear threats
  - GPRD_ACT: beginning of war, escalation of war, terror acts

Data format (Excel .xls):
  Columns: DAY (int YYYYMMDD), GPRD, GPRD_ACT, GPRD_THREAT, date (datetime)

Source: https://www.matteoiacoviello.com/gpr.asp
"""

import io
import logging
from datetime import datetime, timezone

import httpx
import pandas as pd

from app.db.connection import get_db

logger = logging.getLogger(__name__)

# ── GPR Data Source ──────────────────────────────────────────────
# Daily recent data (last ~40 years of daily observations)
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
# Monthly export (alternative)
GPR_MONTHLY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"


async def _download_and_parse(url: str) -> pd.DataFrame:
    """Download Excel file and parse into DataFrame."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}
        )
        resp.raise_for_status()

    df = pd.read_excel(io.BytesIO(resp.content))
    return df


async def collect_gpr_index() -> int:
    """
    Download and parse the GPR Index data.
    Uses the daily data file from Iacoviello's website.
    Returns number of rows written.
    """
    try:
        df = await _download_and_parse(GPR_DAILY_URL)
        source_type = "daily"
    except Exception as e:
        logger.warning("[gpr] Daily download failed (%s), trying monthly...", e)
        try:
            df = await _download_and_parse(GPR_MONTHLY_URL)
            source_type = "monthly"
        except Exception as e2:
            logger.error("[gpr] Both daily and monthly downloads failed: %s", e2)
            logger.info(f"  [gpr] Failed to download GPR data: {e2}")
            return 0

    if df.empty:
        logger.info("  [gpr] Downloaded file was empty")
        return 0

    # Identify columns — daily file uses GPRD, GPRD_ACT, GPRD_THREAT
    gpr_col = None
    act_col = None
    threat_col = None
    date_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower == "date":
            date_col = col
        elif col_lower == "gprd" or col_lower == "gpr":
            gpr_col = col
        elif "act" in col_lower:
            act_col = col
        elif "threat" in col_lower:
            threat_col = col

    # If no 'date' column, use the 'DAY' column (integer YYYYMMDD format)
    if date_col is None and "DAY" in df.columns:
        df["parsed_date"] = pd.to_datetime(
            df["DAY"].astype(str), format="%Y%m%d", errors="coerce"
        )
        date_col = "parsed_date"

    if date_col is None or gpr_col is None:
        logger.error(
            "[gpr] Could not identify date/gpr columns. Found: %s", df.columns.tolist()
        )
        logger.info(f"  [gpr] Column detection failed: {df.columns.tolist()}")
        return 0

    # Drop rows with missing date or GPR
    df = df.dropna(subset=[date_col, gpr_col])

    with get_db() as db:
        count = 0

        for _, row in df.iterrows():
            try:
                date_val = pd.Timestamp(row[date_col]).date()
                gpr_val = float(row[gpr_col])
                acts_val = (
                    float(row[act_col])
                    if act_col and pd.notna(row.get(act_col))
                    else None
                )
                threats_val = (
                    float(row[threat_col])
                    if threat_col and pd.notna(row.get(threat_col))
                    else None
                )
            except (ValueError, TypeError):
                continue

            now = datetime.now(timezone.utc)
            db.execute(
                """
                INSERT INTO global.gpr_index
                (date, gpr, gpr_threats, gpr_acts, source, collected_at)
                VALUES (%s, %s, %s, %s, 'policyuncertainty', %s)
                ON CONFLICT (date) DO UPDATE SET
                    gpr = EXCLUDED.gpr,
                    gpr_threats = EXCLUDED.gpr_threats,
                    gpr_acts = EXCLUDED.gpr_acts,
                    collected_at = EXCLUDED.collected_at
            """,
                [date_val, gpr_val, threats_val, acts_val, now],
            )
            count += 1

        # Log results
        latest_idx = df[date_col].idxmax()
        latest_date = pd.Timestamp(df.loc[latest_idx, date_col]).date()
        latest_gpr = df.loc[latest_idx, gpr_col]

        logger.info(f"  [gpr] {count} GPR index values written ({source_type})")
        logger.info(f"    Date range: {df[date_col].min().date()} → {latest_date}")
        logger.info(f"    Latest GPR: {latest_gpr:.1f}")

        return count


async def collect_all() -> dict:
    """Main entry point."""
    count = await collect_gpr_index()
    return {"gpr_index": count}

"""
Dynamic Tool Router for Cognition V2.
Resolves missing fields in Evidence Packets by routing them to configured data collectors.
"""

import logging
import asyncio
from typing import List

from app.collectors.yfinance_collector import (
    collect_price_history,
    collect_fundamentals,
)

# Import news_collector dynamically if available or assume standard api
from app.collectors.yfinance_collector import collect_news as yf_collect_news

logger = logging.getLogger(__name__)


async def resolve_missing_data(ticker: str, missing_fields: List[str]) -> bool:
    """
    Given a list of missing fields from an EvidencePacket, invokes the appropriate
    data collectors in parallel to backfill the database.

    Returns True if any fetch attempts were made.
    """
    if not missing_fields:
        return False

    tasks = []
    logger.info(f"[ROUTER] Triggered for {ticker} by missing fields: {missing_fields}")

    if "price" in missing_fields:
        logger.info(
            "[ROUTER] -> Routing 'price' to yfinance_collector.collect_price_history"
        )
        tasks.append(collect_price_history(ticker, period="6mo"))

    if "pe_ratio" in missing_fields or "fundamentals" in missing_fields:
        logger.info(
            "[ROUTER] -> Routing 'pe_ratio'/fundamentals to yfinance_collector.collect_fundamentals"
        )
        tasks.append(collect_fundamentals(ticker))

    if "news" in missing_fields:
        logger.info("[ROUTER] -> Routing 'news' to yfinance_collector.collect_news")
        tasks.append(yf_collect_news(ticker))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"[ROUTER] Task {i} failed: {res}")
        return True

    return False

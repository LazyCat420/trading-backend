import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketSnapshot:
    """
    Single source of truth for all numerical values used in prompts.
    """

    ticker: str
    fetched_at: datetime.datetime
    data_source: str
    candles_used: int

    # Price
    price: Optional[float]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    volume: Optional[int]
    vwap: Optional[float]

    # Technicals
    rsi_14: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    bb_pct: Optional[float]
    sma_20: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    atr_14: Optional[float]
    adx_14: Optional[float]
    stoch_k: Optional[float]
    stoch_d: Optional[float]

    # Quant
    returns_1d: Optional[float]
    returns_5d: Optional[float]
    returns_20d: Optional[float]
    volatility_20d: Optional[float]
    sharpe_20d: Optional[float]
    max_drawdown_20d: Optional[float]
    beta_20d: Optional[float]

    # Fundamentals
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    eps: Optional[float]
    market_cap: Optional[float]
    revenue_growth: Optional[float]
    profit_margin: Optional[float]
    debt_to_equity: Optional[float]

    def to_prompt_block(self) -> str:
        """Serializes all fields into a labeled text block for LLM contexts."""
        lines = [f"=== MARKET SNAPSHOT: {self.ticker} ==="]
        lines.append(
            f"Fetched: {self.fetched_at.isoformat()} | Source: {self.data_source} | Candles: {self.candles_used}"
        )

        lines.append("\n[PRICE]")
        for k in ["price", "open", "high", "low", "volume", "vwap"]:
            v = getattr(self, k)
            lines.append(f"{k.upper()}={v if v is not None else 'N/A'}")

        lines.append("\n[TECHNICALS]")
        for k in [
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_lower",
            "bb_pct",
            "sma_20",
            "sma_50",
            "sma_200",
            "atr_14",
            "adx_14",
            "stoch_k",
            "stoch_d",
        ]:
            v = getattr(self, k)
            # Round technicals and quant numbers to 4 decimals if they are floats
            if v is not None and isinstance(v, float):
                v = round(v, 4)
            lines.append(f"{k.upper()}={v if v is not None else 'N/A'}")

        lines.append("\n[QUANT]")
        for k in [
            "returns_1d",
            "returns_5d",
            "returns_20d",
            "volatility_20d",
            "sharpe_20d",
            "max_drawdown_20d",
            "beta_20d",
        ]:
            v = getattr(self, k)
            if v is not None and isinstance(v, float):
                v = round(v, 4)
            lines.append(f"{k.upper()}={v if v is not None else 'N/A'}")

        lines.append("\n[FUNDAMENTALS]")
        for k in [
            "pe_ratio",
            "forward_pe",
            "eps",
            "market_cap",
            "revenue_growth",
            "profit_margin",
            "debt_to_equity",
        ]:
            v = getattr(self, k)
            if v is not None and isinstance(v, float):
                v = round(v, 4)
            lines.append(f"{k.upper()}={v if v is not None else 'N/A'}")

        return "\n".join(lines)

    def assert_price_valid(self):
        """Raises ValueError if core price data is missing."""
        missing = []
        for k in ["price", "open", "high", "low", "volume"]:
            if getattr(self, k) is None:
                missing.append(k)
        if missing:
            raise ValueError(
                f"Missing essential price data for {self.ticker}: {', '.join(missing)}"
            )

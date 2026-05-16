"""
Peer Comparison Processor — Pure math, no LLM.

Computes cross-stock comparative metrics for sector peers and
uncorrelated outperformers. All calculations done in Python so the
LLM never wastes tokens on arithmetic.

Key functions:
    find_sector_peer(ticker)                    → closest sector competitor
    find_uncorrelated_outperformer(ticker, wl)  → best different-sector stock
    compute_exhaustion_signals(ticker)           → trending vs already rallied
    compare_pair(ticker_a, ticker_b)             → head-to-head quant table
    get_momentum_regime(ticker)                  → trend lifecycle stage
    build_comparison_context(ticker, watchlist)   → formatted text for LLM
"""

import logging
import numpy as np
import pandas as pd
from app.db.connection import get_db


def _cursor_to_df(cursor) -> pd.DataFrame:
    """Convert a PostgreSQL cursor result to a pandas DataFrame."""
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    cols = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=cols)


from app.processors.quant_processor import (
    get_zscore,
    get_sharpe,
    get_sortino,
    get_drawdown,
    get_risk_reward,
    get_relative_strength,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# HARDCODED PEER MAP — fallback when ticker_metadata is sparse
# ═══════════════════════════════════════════════════════════
_PEER_MAP = {
    # Payments
    "MA": "V",
    "V": "MA",
    # Big Tech
    "AAPL": "MSFT",
    "MSFT": "AAPL",
    "GOOGL": "META",
    "META": "GOOGL",
    "GOOG": "META",
    # Semis
    "NVDA": "AMD",
    "AMD": "NVDA",
    "INTC": "AMD",
    "TSM": "ASML",
    "ASML": "TSM",
    "MU": "MRVL",
    "MRVL": "MU",
    # E-commerce / Cloud
    "AMZN": "MSFT",
    "SHOP": "AMZN",
    # EV / Auto
    "TSLA": "RIVN",
    "RIVN": "TSLA",
    # Banks
    "JPM": "GS",
    "GS": "JPM",
    "BAC": "WFC",
    "WFC": "BAC",
    # Energy
    "XOM": "CVX",
    "CVX": "XOM",
    "SLB": "HAL",
    "HAL": "SLB",
    # Streaming
    "NFLX": "DIS",
    "DIS": "NFLX",
    # Social
    "SNAP": "PINS",
    "PINS": "SNAP",
    # Fintech
    "SOFI": "HOOD",
    "HOOD": "SOFI",
    # Retail
    "WMT": "COST",
    "COST": "WMT",
    "TGT": "WMT",
    # Healthcare
    "UNH": "CVS",
    "CVS": "UNH",
    "JNJ": "PFE",
    "PFE": "JNJ",
    # Airlines
    "DAL": "UAL",
    "UAL": "DAL",
    # Telecom
    "T": "VZ",
    "VZ": "T",
    # Crypto-adjacent
    "COIN": "HOOD",
    "MSTR": "COIN",
    # China tech
    "BABA": "JD",
    "JD": "BABA",
    "PDD": "BABA",
}

# ═══════════════════════════════════════════════════════════
# SAME-COMPANY EXCLUSIONS — dual-class shares, etc.
# These should NEVER be used as peers for each other.
# ═══════════════════════════════════════════════════════════
_SAME_COMPANY: dict[str, set[str]] = {
    "GOOGL": {"GOOG"},
    "GOOG": {"GOOGL"},
    "BRK.A": {"BRK.B"},
    "BRK.B": {"BRK.A"},
    "BRK-A": {"BRK-B"},
    "BRK-B": {"BRK-A"},
    "DISCA": {"DISCK", "DISCB"},
    "DISCK": {"DISCA", "DISCB"},
    "DISCB": {"DISCA", "DISCK"},
    "FOX": {"FOXA"},
    "FOXA": {"FOX"},
    "LBRDA": {"LBRDK"},
    "LBRDK": {"LBRDA"},
    "NWSA": {"NWS"},
    "NWS": {"NWSA"},
}


# ═══════════════════════════════════════════════════════════
# PEER DISCOVERY
# ═══════════════════════════════════════════════════════════


def find_sector_peer(ticker: str) -> dict | None:
    """Find the best sector peer for a given ticker.

    Priority:
      1. ticker_metadata (same sector + industry, closest market cap)
      2. ticker_correlations (r > 0.7 over 90d, same sector)
      3. Hardcoded fallback map

    Returns dict with {ticker, name, sector, industry, source} or None.
    """
    t = ticker.upper()
    with get_db() as db:
        excluded = _SAME_COMPANY.get(t, set())  # dual-class / same-company tickers

        # 1. Query ticker_metadata for same sector/industry
        try:
            meta = db.execute(
                "SELECT sector, industry, market_cap FROM ticker_metadata WHERE ticker = %s",
                [t],
            ).fetchone()

            if meta and meta[0]:
                sector, industry, mcap = meta
                # Same industry first — exclude same-company tickers
                peers = db.execute(
                    "SELECT ticker, name, sector, industry, market_cap "
                    "FROM ticker_metadata "
                    "WHERE industry = %s AND ticker != %s "
                    "ORDER BY ABS(market_cap - %s) ASC LIMIT 5",
                    [industry, t, mcap or 0],
                ).fetchall()
                for peer in peers:
                    if peer[0] not in excluded:
                        return {
                            "ticker": peer[0],
                            "name": peer[1],
                            "sector": peer[2],
                            "industry": peer[3],
                            "source": "ticker_metadata_industry",
                        }

                # Fallback: same sector — exclude same-company tickers
                peers = db.execute(
                    "SELECT ticker, name, sector, industry, market_cap "
                    "FROM ticker_metadata "
                    "WHERE sector = %s AND ticker != %s "
                    "ORDER BY ABS(market_cap - %s) ASC LIMIT 5",
                    [sector, t, mcap or 0],
                ).fetchall()
                for peer in peers:
                    if peer[0] not in excluded:
                        return {
                            "ticker": peer[0],
                            "name": peer[1],
                            "sector": peer[2],
                            "industry": peer[3],
                            "source": "ticker_metadata_sector",
                        }
        except Exception as e:
            logger.debug("peer discovery DB query failed: %s", e)

        # 2. Correlation-based (r > 0.7)
        try:
            corr = db.execute(
                "SELECT ticker_a, ticker_b, correlation "
                "FROM ticker_correlations "
                "WHERE (ticker_a = %s OR ticker_b = %s) AND period = '90d' "
                "AND correlation > 0.7 "
                "ORDER BY correlation DESC LIMIT 1",
                [t, t],
            ).fetchone()
            if corr:
                peer_ticker = corr[1] if corr[0] == t else corr[0]
                if peer_ticker not in excluded:
                    return {
                        "ticker": peer_ticker,
                        "name": peer_ticker,
                        "sector": "unknown",
                        "industry": "unknown",
                        "source": f"correlation_90d_r={corr[2]:.2f}",
                    }
        except Exception as e:
            logger.debug("peer correlation query failed: %s", e)

        # 3. Hardcoded fallback
        if t in _PEER_MAP:
            peer_t = _PEER_MAP[t]
            return {
                "ticker": peer_t,
                "name": peer_t,
                "sector": "unknown",
                "industry": "unknown",
                "source": "hardcoded_peer_map",
            }

        return None


def find_uncorrelated_outperformer(
    ticker: str,
    watchlist: list[str],
    days: int = 63,  # ~3 months
    max_corr: float = 0.3,
) -> dict | None:
    """Find the best-performing watchlist stock from a DIFFERENT sector
    with low correlation to the target.

    Returns dict with {ticker, return_3mo, correlation, sector} or None.
    """
    t = ticker.upper()
    with get_db() as db:
        # Get target's sector
        target_sector = None
        try:
            meta = db.execute(
                "SELECT sector FROM ticker_metadata WHERE ticker = %s", [t]
            ).fetchone()
            if meta:
                target_sector = meta[0]
        except Exception:
            pass

        # Compute 3-month returns for all watchlist tickers
        candidates = []
        for wt in watchlist:
            wt = wt.upper()
            if wt == t or wt in _SAME_COMPANY.get(t, set()):
                continue

            # Skip same sector if known
            if target_sector:
                try:
                    wt_meta = db.execute(
                        "SELECT sector FROM ticker_metadata WHERE ticker = %s",
                        [wt],
                    ).fetchone()
                    if wt_meta and wt_meta[0] == target_sector:
                        continue
                except Exception:
                    pass

            # Get returns
            try:
                df = _cursor_to_df(
                    db.execute(
                        "SELECT close FROM price_history "
                        "WHERE ticker = %s ORDER BY date DESC LIMIT %s",
                        [wt, days + 1],
                    )
                )
                if df.empty or len(df) < 20:
                    continue
                current = df["close"].iloc[0]
                past = df["close"].iloc[-1]
                ret = (current - past) / past * 100
                candidates.append({"ticker": wt, "return_3mo": round(ret, 2)})
            except Exception:
                continue

        if not candidates:
            return None

        # Sort by best return
        candidates.sort(key=lambda x: x["return_3mo"], reverse=True)

        # Check correlation with target for top candidates
        for c in candidates[:10]:
            try:
                # Quick correlation check
                t_df = _cursor_to_df(
                    db.execute(
                        "SELECT date, close FROM price_history "
                        "WHERE ticker = %s ORDER BY date DESC LIMIT %s",
                        [t, days],
                    )
                )
                c_df = _cursor_to_df(
                    db.execute(
                        "SELECT date, close FROM price_history "
                        "WHERE ticker = %s ORDER BY date DESC LIMIT %s",
                        [c["ticker"], days],
                    )
                )

                if t_df.empty or c_df.empty or len(t_df) < 20 or len(c_df) < 20:
                    continue

                t_df = t_df.sort_values("date").set_index("date")
                c_df = c_df.sort_values("date").set_index("date")

                merged = pd.DataFrame({"a": t_df["close"], "b": c_df["close"]}).dropna()

                if len(merged) < 20:
                    continue

                corr = merged["a"].pct_change().corr(merged["b"].pct_change())
                if abs(corr) < max_corr:
                    c["correlation"] = round(corr, 3)
                    c["sector"] = "different"
                    return c
            except Exception:
                continue

        # If no uncorrelated found, return the best performer anyway with note
        best = candidates[0]
        best["correlation"] = None
        best["sector"] = "unknown"
        best["note"] = "no uncorrelated alternative found, returning top performer"
        return best


# ═══════════════════════════════════════════════════════════
# EXHAUSTION SIGNALS — "Trending vs Already Rallied"
# ═══════════════════════════════════════════════════════════


def compute_exhaustion_signals(ticker: str) -> dict:
    """Compute 6 quantitative exhaustion signals for a ticker.

    All pure math — no LLM. These signals tell you whether a stock
    is in a healthy trend or has already run too far.

    Returns dict with labeled signals for each formula.
    """
    t = ticker.upper()
    with get_db() as db:
        signals = {"ticker": t, "signals": {}}

        # Fetch price + volume history
        try:
            df = _cursor_to_df(
                db.execute(
                    "SELECT date, close, high, low, volume "
                    "FROM price_history WHERE ticker = %s "
                    "ORDER BY date ASC LIMIT 100",
                    [t],
                )
            )
        except Exception as e:
            return {"ticker": t, "error": f"no price data: {e}"}

        if df.empty or len(df) < 25:
            return {"ticker": t, "error": f"insufficient data ({len(df)} rows)"}

        close = df["close"].values
        volume = df["volume"].values.astype(float)
        n = len(close)

        # Fetch technicals
        try:
            tech = _cursor_to_df(
                db.execute(
                    "SELECT date, rsi_14, adx_14, bb_upper, bb_lower, bb_mid, "
                    "       macd_hist, stoch_k "
                    "FROM technicals WHERE ticker = %s "
                    "ORDER BY date DESC LIMIT 25",
                    [t],
                )
            )
            if not tech.empty:
                tech = tech.sort_values("date")
        except Exception:
            tech = pd.DataFrame()

        # ── 1. ADX Slope Decay ──
        if not tech.empty and "adx_14" in tech.columns and len(tech) >= 6:
            adx_vals = tech["adx_14"].dropna().values
            if len(adx_vals) >= 6:
                adx_now = adx_vals[-1]
                adx_5ago = adx_vals[-6]
                adx_slope = (adx_now - adx_5ago) / 5.0

                if adx_now > 25 and adx_slope < -0.5:
                    label = "TREND_EXHAUSTING"
                elif adx_now > 25 and adx_slope > 0.5:
                    label = "TREND_ACCELERATING"
                elif adx_now > 25:
                    label = "TREND_STEADY"
                else:
                    label = "NO_TREND"

                signals["signals"]["adx_slope"] = {
                    "adx": round(adx_now, 1),
                    "slope": round(adx_slope, 3),
                    "label": label,
                }

        # ── 2. RSI Divergence ──
        if not tech.empty and "rsi_14" in tech.columns and len(tech) >= 15:
            rsi_vals = tech["rsi_14"].dropna().values
            if len(rsi_vals) >= 15 and len(close) >= 15:
                price_higher = close[-1] > close[-15]
                rsi_lower = rsi_vals[-1] < rsi_vals[-15]
                price_lower = close[-1] < close[-15]
                rsi_higher = rsi_vals[-1] > rsi_vals[-15]

                if price_higher and rsi_lower:
                    label = "BEARISH_DIVERGENCE"
                elif price_lower and rsi_higher:
                    label = "BULLISH_DIVERGENCE"
                else:
                    label = "NO_DIVERGENCE"

                signals["signals"]["rsi_divergence"] = {
                    "rsi_now": round(float(rsi_vals[-1]), 1),
                    "rsi_14ago": round(float(rsi_vals[-15]), 1),
                    "price_direction": "UP" if price_higher else "DOWN",
                    "label": label,
                }

        # ── 3. Volume Climax Detection ──
        if n >= 25 and len(volume) >= 25:
            vol_sma20 = np.mean(volume[-20:])
            vol_today = volume[-1]
            vol_ratio = vol_today / vol_sma20 if vol_sma20 > 0 else 1.0
            price_up = close[-1] > close[-2]

            if price_up and vol_ratio > 2.5:
                label = "BUYING_CLIMAX"
            elif not price_up and vol_ratio > 2.5:
                label = "SELLING_CLIMAX"
            elif vol_ratio > 1.5:
                label = "ELEVATED_VOLUME"
            else:
                label = "NORMAL_VOLUME"

            signals["signals"]["volume_climax"] = {
                "vol_ratio": round(vol_ratio, 2),
                "label": label,
            }

        # ── 4. Bollinger %B Regime ──
        if not tech.empty and all(
            c in tech.columns for c in ["bb_upper", "bb_lower", "bb_mid"]
        ):
            bb_vals = tech[["bb_upper", "bb_lower", "bb_mid"]].dropna()
            if len(bb_vals) >= 1:
                bb_u = bb_vals["bb_upper"].iloc[-1]
                bb_l = bb_vals["bb_lower"].iloc[-1]
                bb_m = bb_vals["bb_mid"].iloc[-1]
                price = close[-1]

                if bb_u != bb_l:
                    pct_b = (price - bb_l) / (bb_u - bb_l)
                    bb_width = (bb_u - bb_l) / bb_m if bb_m > 0 else 0

                    if pct_b > 1.0:
                        label = "ABOVE_UPPER_BAND"
                    elif pct_b < 0.0:
                        label = "BELOW_LOWER_BAND"
                    elif bb_width < 0.04:
                        label = "SQUEEZE"
                    elif pct_b > 0.8:
                        label = "UPPER_ZONE"
                    elif pct_b < 0.2:
                        label = "LOWER_ZONE"
                    else:
                        label = "MID_RANGE"

                    signals["signals"]["bollinger_b"] = {
                        "pct_b": round(pct_b, 3),
                        "bb_width": round(bb_width, 4),
                        "label": label,
                    }

        # ── 5. Rate of Change Deceleration ──
        if n >= 25:
            roc_5 = (close[-1] - close[-6]) / close[-6] * 100 if close[-6] > 0 else 0
            roc_20 = (
                (close[-1] - close[-21]) / close[-21] * 100
                if n >= 22 and close[-21] > 0
                else 0
            )

            if roc_20 != 0:
                if roc_5 > 0 and roc_5 < roc_20 / 4:
                    label = "DECELERATING"
                elif roc_5 > roc_20 and roc_5 > 0:
                    label = "ACCELERATING"
                elif roc_5 < 0 and roc_20 > 0:
                    label = "REVERSING_DOWN"
                elif roc_5 > 0 and roc_20 < 0:
                    label = "REVERSING_UP"
                else:
                    label = "STEADY"
            else:
                label = "FLAT"

            signals["signals"]["rate_of_change"] = {
                "roc_5d": round(roc_5, 2),
                "roc_20d": round(roc_20, 2),
                "label": label,
            }

        # ── 6. Z-Score (mean reversion distance) ──
        zscore_data = get_zscore(t)
        if "zscore" in zscore_data:
            signals["signals"]["zscore"] = {
                "value": zscore_data["zscore"],
                "label": zscore_data["signal"],
            }

        return signals


def get_momentum_regime(ticker: str) -> dict:
    """Classify the current momentum regime of a stock.

    Returns one of:
        EARLY_TREND       — breakout from consolidation, ADX rising from <20
        MID_TREND         — healthy trend, ADX 25-40, RSI 40-65 (bull) or 35-60 (bear)
        LATE_TREND        — extended, ADX >40, RSI >70 or <30
        EXHAUSTED         — multiple exhaustion signals firing
        REVERSAL_CANDIDATE — divergence + oversold/overbought + volume
        CONSOLIDATING     — ADX <20, tight Bollinger, no momentum
    """
    t = ticker.upper()
    exh = compute_exhaustion_signals(t)

    if "error" in exh:
        return {"ticker": t, "regime": "UNKNOWN", "reason": exh["error"]}

    sigs = exh.get("signals", {})

    # Count exhaustion flags
    exhaustion_count = 0
    adx_label = sigs.get("adx_slope", {}).get("label", "")
    rsi_div = sigs.get("rsi_divergence", {}).get("label", "")
    vol_label = sigs.get("volume_climax", {}).get("label", "")
    bb_label = sigs.get("bollinger_b", {}).get("label", "")
    roc_label = sigs.get("rate_of_change", {}).get("label", "")
    z_label = sigs.get("zscore", {}).get("label", "")

    if adx_label == "TREND_EXHAUSTING":
        exhaustion_count += 1
    if rsi_div in ("BEARISH_DIVERGENCE", "BULLISH_DIVERGENCE"):
        exhaustion_count += 1
    if vol_label in ("BUYING_CLIMAX", "SELLING_CLIMAX"):
        exhaustion_count += 1
    if bb_label in ("ABOVE_UPPER_BAND", "BELOW_LOWER_BAND"):
        exhaustion_count += 1
    if roc_label == "DECELERATING":
        exhaustion_count += 1
    if z_label in ("OVERBOUGHT_EXTREME", "OVERSOLD_EXTREME"):
        exhaustion_count += 1

    adx_val = sigs.get("adx_slope", {}).get("adx", 0)
    rsi_val = sigs.get("rsi_divergence", {}).get("rsi_now", 50)

    # Classify
    if exhaustion_count >= 3:
        regime = "EXHAUSTED"
        reason = f"{exhaustion_count}/6 exhaustion signals firing"
    elif rsi_div in ("BEARISH_DIVERGENCE", "BULLISH_DIVERGENCE") and z_label in (
        "OVERBOUGHT_EXTREME",
        "OVERSOLD_EXTREME",
    ):
        regime = "REVERSAL_CANDIDATE"
        reason = f"Divergence + extreme z-score ({rsi_div})"
    elif adx_label == "NO_TREND" and bb_label == "SQUEEZE":
        regime = "CONSOLIDATING"
        reason = "ADX <20 + Bollinger squeeze"
    elif adx_label == "NO_TREND":
        regime = "CONSOLIDATING"
        reason = "ADX <20, no directional trend"
    elif adx_val < 25 and adx_label == "TREND_ACCELERATING":
        regime = "EARLY_TREND"
        reason = "ADX rising from low base"
    elif 25 <= adx_val <= 40 and 35 <= rsi_val <= 70:
        regime = "MID_TREND"
        reason = f"ADX={adx_val:.0f}, RSI={rsi_val:.0f} — healthy trend"
    elif adx_val > 40 or rsi_val > 75 or rsi_val < 25:
        regime = "LATE_TREND"
        reason = f"ADX={adx_val:.0f}, RSI={rsi_val:.0f} — extended"
    else:
        regime = "MID_TREND"
        reason = "Default classification"

    return {
        "ticker": t,
        "regime": regime,
        "reason": reason,
        "exhaustion_count": exhaustion_count,
        "exhaustion_signals": sigs,
    }


# ═══════════════════════════════════════════════════════════
# HEAD-TO-HEAD COMPARISON
# ═══════════════════════════════════════════════════════════


def compare_pair(ticker_a: str, ticker_b: str) -> dict:
    """Head-to-head quant comparison between two tickers.

    Returns structured dict with side-by-side metrics.
    """
    a, b = ticker_a.upper(), ticker_b.upper()

    a_sharpe = get_sharpe(a, days=63)
    b_sharpe = get_sharpe(b, days=63)
    a_sortino = get_sortino(a, days=63)
    b_sortino = get_sortino(b, days=63)
    a_dd = get_drawdown(a, days=63)
    b_dd = get_drawdown(b, days=63)
    a_zscore = get_zscore(a)
    b_zscore = get_zscore(b)
    a_rr = get_risk_reward(a)
    b_rr = get_risk_reward(b)
    a_regime = get_momentum_regime(a)
    b_regime = get_momentum_regime(b)

    # Relative strength (3-month return)
    rs = get_relative_strength([a, b], days=63)
    rs_map = {r["ticker"]: r for r in rs}

    def _pick(data, key, fallback=None):
        if isinstance(data, dict) and "error" not in data:
            return data.get(key, fallback)
        return fallback

    return {
        "ticker_a": a,
        "ticker_b": b,
        "metrics": {
            "return_3mo": {
                a: rs_map.get(a, {}).get("return_pct"),
                b: rs_map.get(b, {}).get("return_pct"),
                "winner": a
                if (
                    rs_map.get(a, {}).get("return_pct", -999)
                    > rs_map.get(b, {}).get("return_pct", -999)
                )
                else b,
            },
            "sharpe_3mo": {
                a: _pick(a_sharpe, "sharpe"),
                b: _pick(b_sharpe, "sharpe"),
                "winner": a
                if (_pick(a_sharpe, "sharpe", -999) > _pick(b_sharpe, "sharpe", -999))
                else b,
            },
            "sortino_3mo": {
                a: _pick(a_sortino, "sortino"),
                b: _pick(b_sortino, "sortino"),
                "winner": a
                if (
                    _pick(a_sortino, "sortino", -999)
                    > _pick(b_sortino, "sortino", -999)
                )
                else b,
            },
            "max_drawdown": {
                a: _pick(a_dd, "max_drawdown_pct"),
                b: _pick(b_dd, "max_drawdown_pct"),
                "winner": a
                if (
                    _pick(a_dd, "max_drawdown_pct", -999)
                    > _pick(b_dd, "max_drawdown_pct", -999)
                )
                else b,
            },
            "zscore": {
                a: _pick(a_zscore, "zscore"),
                b: _pick(b_zscore, "zscore"),
                "a_signal": _pick(a_zscore, "signal", "?"),
                "b_signal": _pick(b_zscore, "signal", "?"),
            },
            "risk_reward": {
                a: _pick(a_rr, "risk_reward_ratio"),
                b: _pick(b_rr, "risk_reward_ratio"),
                "winner": a
                if (
                    _pick(a_rr, "risk_reward_ratio", 0)
                    > _pick(b_rr, "risk_reward_ratio", 0)
                )
                else b,
            },
            "momentum_regime": {
                a: a_regime.get("regime", "?"),
                b: b_regime.get("regime", "?"),
                "a_reason": a_regime.get("reason", ""),
                "b_reason": b_regime.get("reason", ""),
            },
        },
    }


# ═══════════════════════════════════════════════════════════
# CONTEXT BUILDER — formatted text for LLM consumption
# ═══════════════════════════════════════════════════════════


def build_comparison_context(
    ticker: str,
    watchlist: list[str],
) -> str:
    """Build a complete peer comparison text section for LLM context.

    Finds the sector peer + uncorrelated outperformer, runs all quant
    comparisons, and formats into readable text. Zero LLM tokens used.
    """
    t = ticker.upper()
    lines = ["\n## PEER COMPARISON (Pre-Computed — Do NOT Recalculate)"]

    # ── Find sector peer ──
    peer = find_sector_peer(t)
    if peer:
        lines.append(f"\n### Sector Peer: {peer['ticker']}")
        lines.append(f"Source: {peer['source']}")

        comparison = compare_pair(t, peer["ticker"])
        metrics = comparison.get("metrics", {})

        # Format head-to-head table
        lines.append(f"\n{'Metric':<20} {t:<12} {peer['ticker']:<12} Winner")
        lines.append("-" * 56)
        for metric_name, vals in metrics.items():
            if metric_name == "momentum_regime":
                lines.append(
                    f"{'Momentum Regime':<20} "
                    f"{vals.get(t, '?'):<12} "
                    f"{vals.get(peer['ticker'], '?'):<12}"
                )
                if vals.get("a_reason"):
                    lines.append(f"  {t}: {vals['a_reason']}")
                if vals.get("b_reason"):
                    lines.append(f"  {peer['ticker']}: {vals['b_reason']}")
            elif metric_name == "zscore":
                a_z = vals.get(t)
                b_z = vals.get(peer["ticker"])
                a_sig = vals.get("a_signal", "?")
                b_sig = vals.get("b_signal", "?")
                if a_z is not None and b_z is not None:
                    a_str = f"{a_z:.2f} ({a_sig})"
                    b_str = f"{b_z:.2f} ({b_sig})"
                    lines.append(f"{'Z-Score':<20} {a_str:<20} {b_str:<20}")
                else:
                    lines.append(f"{'Z-Score':<20} {'N/A':<20} {'N/A':<20}")
            else:
                a_val = vals.get(t)
                b_val = vals.get(peer["ticker"])
                winner = vals.get("winner", "?")
                fmt_name = metric_name.replace("_", " ").title()
                a_str = f"{a_val:.2f}" if isinstance(a_val, (int, float)) else "N/A"
                b_str = f"{b_val:.2f}" if isinstance(b_val, (int, float)) else "N/A"
                w_str = f"← {winner}" if winner else ""
                lines.append(f"{fmt_name:<20} {a_str:<12} {b_str:<12} {w_str}")
    else:
        lines.append("\nNo sector peer found in database.")

    # ── Find uncorrelated outperformer ──
    outperformer = find_uncorrelated_outperformer(t, watchlist)
    if outperformer and outperformer["ticker"] != (peer or {}).get("ticker"):
        lines.append(f"\n### Best Uncorrelated Outperformer: {outperformer['ticker']}")
        lines.append(
            f"3-Month Return: {outperformer['return_3mo']:+.1f}% | "
            f"Correlation with {t}: {outperformer.get('correlation', 'unknown')}"
        )
        if outperformer.get("note"):
            lines.append(f"Note: {outperformer['note']}")

        # Quick comparison
        comp2 = compare_pair(t, outperformer["ticker"])
        m2 = comp2.get("metrics", {})
        ret_a = m2.get("return_3mo", {}).get(t)
        ret_b = m2.get("return_3mo", {}).get(outperformer["ticker"])
        sharpe_a = m2.get("sharpe_3mo", {}).get(t)
        sharpe_b = m2.get("sharpe_3mo", {}).get(outperformer["ticker"])

        if ret_a is not None and ret_b is not None:
            lines.append(
                f"  Return: {t} {ret_a:+.1f}% vs {outperformer['ticker']} {ret_b:+.1f}%"
            )
        if sharpe_a is not None and sharpe_b is not None:
            lines.append(
                f"  Sharpe: {t} {sharpe_a:.2f} vs "
                f"{outperformer['ticker']} {sharpe_b:.2f}"
            )

    # ── Exhaustion signals for target ──
    regime = get_momentum_regime(t)
    lines.append(f"\n### Momentum Regime: {regime['regime']}")
    lines.append(f"Reason: {regime.get('reason', '?')}")
    lines.append(f"Exhaustion signals firing: {regime.get('exhaustion_count', 0)}/6")

    sigs = regime.get("exhaustion_signals", {})
    if sigs:
        lines.append("\nDetailed Signals:")
        for sig_name, sig_data in sigs.items():
            label = (
                sig_data.get("label", "?")
                if isinstance(sig_data, dict)
                else str(sig_data)
            )
            lines.append(f"  {sig_name}: {label}")
            # Add key values
            if isinstance(sig_data, dict):
                for k, v in sig_data.items():
                    if k != "label" and v is not None:
                        lines.append(f"    {k}: {v}")

    lines.append("")
    return "\n".join(lines)

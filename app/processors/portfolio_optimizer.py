"""
Portfolio Optimizer -- Cross-stock analysis between decision engine and trading phase.

Sits AFTER the decision engine produces individual BUY/SELL/HOLD decisions,
BEFORE the trading phase executes them:

  Decision Engine -> Portfolio Optimizer -> Trading Phase
                          |
                     Rank by score
                     Drop correlated dupes
                     Check portfolio exposure
                     Apply regime multiplier
                     Allocate capital
"""

from app.processors.quant_processor import (
    get_correlations,
    get_ticker_score,
    get_risk_reward,
)
from app.processors.market_regime import get_market_regime
from app.trading.paper_trader import get_portfolio


def optimize_decisions(
    decisions: list[dict],
    bot_id: str = "default",
    max_positions: int = 5,
    max_sector_exposure: float = 0.40,
    correlation_threshold: float = 0.80,
) -> list[dict]:
    """Filter, rank, and size decisions before execution.

    Steps:
        1. Get market regime (adjusts position sizing)
        2. Score each BUY candidate with composite ticker score
        3. Check correlations -- drop correlated duplicates
        4. Check existing portfolio exposure
        5. Rank by composite score, keep top N
        6. Adjust position size by regime and confidence

    Returns filtered + annotated decisions ready for trading_phase.
    """
    print(f"\n{'=' * 60}")
    print(f"PORTFOLIO OPTIMIZER: {len(decisions)} decisions")
    print(f"{'=' * 60}")

    # -- Step 1: Market regime --
    regime = get_market_regime()
    regime_mult = regime.get("position_multiplier", 0.5)
    print(
        f"  [REGIME] {regime['regime']} (bull_score={regime.get('bull_score', '?')}, "
        f"position_mult={regime_mult})"
    )

    # -- Step 2: Separate BUY candidates from HOLD/SELL --
    buys = [
        d for d in decisions if d.get("action") == "BUY" and not d.get("human_review")
    ]
    sells = [d for d in decisions if d.get("action") == "SELL"]
    holds = [d for d in decisions if d.get("action") == "HOLD" or d.get("human_review")]

    print(f"  [INPUT] {len(buys)} BUY, {len(sells)} SELL, {len(holds)} HOLD/REVIEW")

    if not buys:
        print("  [RESULT] No BUY candidates to optimize")
        return sells + holds

    # -- Step 3: Score each BUY candidate --
    scored_buys = []
    for d in buys:
        ticker = d["ticker"]
        score = get_ticker_score(ticker)
        rr = get_risk_reward(ticker)
        d["quant_score"] = score.get("composite_score", 50)
        d["quant_components"] = score.get("components", {})
        d["risk_reward"] = rr.get("risk_reward_ratio", 1.0)
        d["quant_rating"] = score.get("rating", "NEUTRAL")
        scored_buys.append(d)
        print(
            f"  [{ticker}] Score: {d['quant_score']:.1f} | "
            f"R:R: {d['risk_reward']:.1f} | Rating: {d['quant_rating']}"
        )

    # -- Step 4: Correlation check -- drop highly correlated dupes --
    buy_tickers = [d["ticker"] for d in scored_buys]
    if len(buy_tickers) >= 2:
        corr_data = get_correlations(buy_tickers)
        high_corr = corr_data.get("high_correlations", [])

        if high_corr:
            print(f"\n  [CORRELATION] Found {len(high_corr)} highly correlated pairs:")
            to_drop = set()
            for pair in high_corr:
                t1, t2 = pair["pair"].split("/")
                corr_val = pair["correlation"]
                if corr_val > correlation_threshold:
                    # Keep the one with higher composite score
                    s1 = next(
                        (d["quant_score"] for d in scored_buys if d["ticker"] == t1), 0
                    )
                    s2 = next(
                        (d["quant_score"] for d in scored_buys if d["ticker"] == t2), 0
                    )
                    drop = t2 if s1 >= s2 else t1
                    keep = t1 if s1 >= s2 else t2
                    to_drop.add(drop)
                    print(
                        f"    {t1}/{t2} corr={corr_val:.3f} -> DROP {drop} (score {min(s1, s2):.1f}), KEEP {keep} (score {max(s1, s2):.1f})"
                    )

            # Remove dropped tickers
            before = len(scored_buys)
            scored_buys = [d for d in scored_buys if d["ticker"] not in to_drop]
            for dropped in to_drop:
                holds.append(
                    {
                        "ticker": dropped,
                        "action": "HOLD",
                        "confidence": 0,
                        "rationale": "Dropped: correlated with better-scored ticker",
                        "human_review": False,
                        "total_tokens": 0,
                        "total_time_s": 0,
                        "config_used": "optimizer",
                        "escalated": False,
                    }
                )
            print(f"    Dropped {before - len(scored_buys)} correlated tickers")

    # -- Step 5: Check existing portfolio --
    portfolio = get_portfolio(bot_id)
    existing_tickers = {p["ticker"] for p in portfolio.get("positions", [])}
    if existing_tickers:
        print(f"\n  [PORTFOLIO] Already holding: {existing_tickers}")
        # Don't buy what we already own (could average down in future)
        scored_buys = [d for d in scored_buys if d["ticker"] not in existing_tickers]

    # -- Step 6: Rank and cap --
    scored_buys.sort(key=lambda d: d["quant_score"], reverse=True)

    # Fix #5: Minimum score gate — NEUTRAL/BEARISH-rated trades are too weak to execute
    MIN_SCORE = 60
    weak = [
        d
        for d in scored_buys
        if d["quant_score"] < MIN_SCORE or d["quant_rating"] in ("NEUTRAL", "BEARISH")
    ]
    if weak:
        for d in weak:
            print(
                f"  [{d['ticker']}] BLOCKED: score {d['quant_score']:.1f} / rating {d['quant_rating']} below threshold"
            )
            holds.append(
                {
                    "ticker": d["ticker"],
                    "action": "HOLD",
                    "confidence": d.get("confidence", 0),
                    "rationale": f"Blocked by optimizer: score {d['quant_score']:.1f} < {MIN_SCORE} or rating {d['quant_rating']}",
                    "human_review": False,
                    "total_tokens": d.get("total_tokens", 0),
                    "total_time_s": d.get("total_time_s", 0),
                    "config_used": "optimizer",
                    "escalated": False,
                }
            )
        scored_buys = [
            d
            for d in scored_buys
            if d["quant_score"] >= MIN_SCORE
            and d["quant_rating"] not in ("NEUTRAL", "BEARISH")
        ]

    available_slots = max_positions - len(existing_tickers)
    top_picks = scored_buys[: max(available_slots, 0)]

    if len(scored_buys) > available_slots:
        dropped_tickers = [d["ticker"] for d in scored_buys[available_slots:]]
        print(
            f"  [CAP] Max {max_positions} positions, {len(existing_tickers)} held "
            f"-> {available_slots} slots -> dropped {dropped_tickers}"
        )
        for d in scored_buys[available_slots:]:
            holds.append(
                {
                    "ticker": d["ticker"],
                    "action": "HOLD",
                    "confidence": 0,
                    "rationale": "Dropped: position limit reached",
                    "human_review": False,
                    "total_tokens": 0,
                    "total_time_s": 0,
                    "config_used": "optimizer",
                    "escalated": False,
                }
            )

    # -- Step 7: Adjust position sizes --
    for d in top_picks:
        conf = d.get("confidence", 70)
        base_size = 0.05 if conf >= 80 else 0.03
        adjusted_size = base_size * regime_mult
        d["position_size_pct"] = round(adjusted_size, 4)
        d["regime"] = regime["regime"]
        print(
            f"  [{d['ticker']}] Final size: {adjusted_size * 100:.1f}% "
            f"(base {base_size * 100:.0f}% x regime {regime_mult})"
        )

    print(f"\n  [RESULT] {len(top_picks)} BUY, {len(sells)} SELL, {len(holds)} HOLD")
    print(f"{'=' * 60}\n")

    return top_picks + sells + holds

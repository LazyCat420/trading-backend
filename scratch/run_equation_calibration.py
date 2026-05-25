"""
Equation Calibration Script.
Runs backtests on current scoring specs, dispatches the Equation Designer LLM
to optimize weights, and writes the updated spec file under strict constraints.
"""

import sys
import os
import json
import yaml
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.trading.quant_edge_verifier import load_historical_data, backtest_spec_strategy
from app.services.prism_agent_caller import call_prism_agent
from app.services.vllm_client import Priority
from app.db.connection import get_db

EQUATION_DESIGNER_SYSTEM_PROMPT = """You are a meta-quant LLM agent responsible for designing and optimizing mathematical scoring equations for a quantitative trading pipeline.

You are given:
- The current YAML scoring spec configuration.
- The backtest results (total trades, win-rate, cumulative return, max drawdown) when running this formula on historical stock data.

Your goal is to optimize the weight coefficients in the "formula" string to maximize cumulative return and win rate while minimizing drawdown.

CONSTRAINTS:
1. The weights of the inputs (e.g., ev_norm, rr_norm, kelly_norm) must sum to exactly 1.0.
2. The formula must only use addition (+), subtraction (-), multiplication (*), division (/), constants, and variables in the inputs list.
3. The inputs must be monotonically increasing (higher is better for the edge score).

Return ONLY the updated YAML block matching this schema:
edge_score:
  inputs:
    - ev_norm
    - rr_norm
    - kelly_norm
  formula: "Weight1 * ev_norm + Weight2 * rr_norm + Weight3 * kelly_norm"
  mapping:
    type: linear
    to_scale: [1, 10]
  constraints:
    monotonic:
      increasing: ["ev_norm", "rr_norm", "kelly_norm"]

Ensure the output is strictly valid YAML without any markdown formatting around it."""

async def run_calibration():
    # 1. Run backtests using current spec
    print("Running initial spec backtests on database records...")
    with get_db() as db:
        rows = db.execute("SELECT DISTINCT ticker FROM price_history LIMIT 3").fetchall()
        tickers = [r[0] for r in rows]
        
    results = {}
    for ticker in tickers:
        df = load_historical_data(ticker)
        if df.empty or len(df) < 20:
            continue
        res = backtest_spec_strategy(df)
        if "error" not in res:
            results[ticker] = {
                "trades": res["total_trades"],
                "win_rate": res["win_rate_pct"],
                "return": res["cumulative_return_pct"],
                "drawdown": res["max_drawdown_pct"]
            }
            
    print(f"Current backtest results: {json.dumps(results, indent=2)}")
    
    # 2. Load current edge_score spec
    spec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        "app/trading/scoring_specs/edge_score.yaml"
    )
    current_spec_content = ""
    if os.path.exists(spec_path):
        with open(spec_path, "r") as f:
            current_spec_content = f.read()
            
    # 3. Call Equation Designer LLM
    print("Invoking Equation Designer Agent to optimize spec...")
    user_message = f"""Current YAML spec:
{current_spec_content}

Backtest Performance:
{json.dumps(results, indent=2)}
"""
    try:
        response, _, _ = await call_prism_agent(
            agent_id="CUSTOM_TRADING_CYCLE_ANALYSIS_AGENT",
            user_message=user_message,
            fallback_system_prompt=EQUATION_DESIGNER_SYSTEM_PROMPT,
            fallback_agent_name="equation_designer",
            temperature=0.1,
            max_tokens=1024,
            priority=Priority.NORMAL,
            ticker="SYSTEM",
        )
        
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```yaml")[-1].split("```")[0].strip()
            
        print(f"LLM proposed spec:\n{cleaned}")
        
        # Verify YAML is valid
        parsed = yaml.safe_load(cleaned)
        if "edge_score" in parsed and "formula" in parsed["edge_score"]:
            with open(spec_path, "w") as f:
                f.write(cleaned)
            print("Successfully updated edge_score.yaml with optimized spec!")
        else:
            print("Error: Proposed spec structure is invalid.")
            
    except Exception as e:
        print(f"Calibration failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_calibration())

"""
Scoring Engine.
Computes normalized features, parses YAML scoring specs, safely evaluates
mathematical equations, and constructs Hierarchical Pillar Profiles.
"""

import os
import ast
import yaml
import math
import logging
import operator
import pandas as pd
from typing import Dict, List, Any
from app.db.connection import get_db

logger = logging.getLogger(__name__)

# Safe AST operators
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: lambda x: x
}

def safe_eval(expr: str, variables: Dict[str, float]) -> float:
    """Safely evaluates mathematical expressions using AST parsing."""
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Constant value {node.value} not allowed")
        elif isinstance(node, ast.Num):  # Python < 3.8 fallback
            return float(node.n)
        elif isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            raise ValueError(f"Variable {node.id} not defined")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in _ALLOWED_OPERATORS:
                return _ALLOWED_OPERATORS[op_type](_eval(node.left), _eval(node.right))
            raise ValueError(f"Operator {op_type} not allowed")
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in _ALLOWED_OPERATORS:
                return _ALLOWED_OPERATORS[op_type](_eval(node.operand))
            raise ValueError(f"Unary operator {op_type} not allowed")
        else:
            raise ValueError(f"AST node type {type(node)} not allowed")
            
    tree = ast.parse(expr, mode='eval')
    return _eval(tree)

def compute_normalized_features(ticker: str) -> Dict[str, float]:
    """
    Query raw metrics from the DB and normalize them into 0-1 values.
    """
    ticker = ticker.upper()
    features = {
        "ev_norm": 0.5,
        "rr_norm": 0.5,
        "kelly_norm": 0.5,
        "vol_norm": 0.5,
        "dd_norm": 0.5,
        "beta_norm": 0.5,
        "z_score_norm": 0.5,
        "rsi_norm": 0.5,
        "raw_ev": 0.0,
        "raw_rr": 1.0,
        "raw_kelly": 0.0,
        "raw_vol": 0.0,
        "raw_dd": 0.0,
        "raw_beta": 1.0,
        "raw_z_score": 0.0,
        "raw_rsi": 50.0,
    }
    
    try:
        with get_db() as db:
            # 1. Fundamentals (Expected Value, Kelly)
            fund_row = db.execute(
                "SELECT revenue_growth, profit_margin, market_cap FROM fundamentals WHERE ticker = %s ORDER BY snapshot_date DESC LIMIT 1",
                [ticker]
            ).fetchone()
            
            # Simple EV calculation mock/formula based on growth and margins
            if fund_row:
                rev_growth = fund_row[0] or 0.0
                margin = fund_row[1] or 0.0
                raw_ev = rev_growth * 0.7 + margin * 0.3
                features["raw_ev"] = raw_ev
                features["ev_norm"] = 1.0 / (1.0 + math.exp(-raw_ev * 5.0)) # sigmoid
                
                # Simple Kelly estimate
                raw_kelly = max(0.0, min(0.25, rev_growth * 0.2))
                features["raw_kelly"] = raw_kelly
                features["kelly_norm"] = raw_kelly / 0.25

            # 2. Technicals (R/R, Z-Score, Vol, Drawdown, RSI)
            tech_row = db.execute(
                "SELECT rsi_14, atr_14, support, resistance FROM technicals WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                [ticker]
            ).fetchone()
            
            price_row = db.execute(
                "SELECT close FROM price_history WHERE ticker = %s ORDER BY date DESC LIMIT 1",
                [ticker]
            ).fetchone()
            
            if tech_row and price_row:
                rsi, atr, support, resistance = tech_row
                close = price_row[0]
                
                features["raw_rsi"] = rsi or 50.0
                features["rsi_norm"] = (rsi or 50.0) / 100.0
                
                # R/R
                sup_price = support or (close * 0.95)
                res_price = resistance or (close * 1.05)
                risk = close - sup_price
                reward = res_price - close
                raw_rr = reward / risk if risk > 0 else 1.0
                features["raw_rr"] = raw_rr
                features["rr_norm"] = min(1.0, max(0.0, raw_rr / 5.0))
                
                # Volatility (ATR normalized by price)
                raw_vol = (atr / close) if atr and close else 0.02
                features["raw_vol"] = raw_vol
                features["vol_norm"] = min(1.0, max(0.0, (raw_vol - 0.01) / 0.10))

            # 3. Z-Score (rolling)
            price_rows = db.execute(
                "SELECT close FROM price_history WHERE ticker = %s ORDER BY date DESC LIMIT 60",
                [ticker]
            ).fetchall()
            if len(price_rows) >= 20:
                closes = [r[0] for r in price_rows]
                mean_price = sum(closes) / len(closes)
                std_price = math.sqrt(sum((c - mean_price)**2 for c in closes) / len(closes))
                if std_price > 0:
                    raw_z = (closes[0] - mean_price) / std_price
                    features["raw_z_score"] = raw_z
                    features["z_score_norm"] = min(1.0, max(0.0, (raw_z + 3.0) / 6.0))
                    
            # 4. Drawdown
            if len(price_rows) >= 5:
                closes = [r[0] for r in reversed(price_rows)]
                peak = max(closes)
                raw_dd = abs(closes[-1] - peak) / peak if peak > 0 else 0.0
                features["raw_dd"] = raw_dd
                features["dd_norm"] = min(1.0, max(0.0, raw_dd / 0.50))
                
    except Exception as e:
        logger.warning(f"[SCORING] Failed to compute normalized features for {ticker}: {e}")
        
    return features

def calculate_pillar_score(spec_name: str, variables: Dict[str, float]) -> float:
    """
    Load a spec file, evaluate its formula, and scale it to 1-10.
    """
    spec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring_specs")
    spec_path = os.path.join(spec_dir, f"{spec_name}.yaml")
    
    if not os.path.exists(spec_path):
        logger.warning(f"[SCORING] Spec {spec_name} not found. Fallback to default score 5.0")
        return 5.0
        
    try:
        with open(spec_path, "r") as f:
            spec = yaml.safe_load(f)
            
        spec_data = spec.get(spec_name, {})
        formula = spec_data.get("formula", "")
        raw_score = safe_eval(formula, variables)
        
        # Clamp to 0-1
        raw_score = max(0.0, min(1.0, raw_score))
        
        # Scale to 1-10
        mapping = spec_data.get("mapping", {})
        if mapping.get("type") == "linear":
            scale = mapping.get("to_scale", [1, 10])
            final_score = scale[0] + (scale[1] - scale[0]) * raw_score
            return round(final_score, 1)
            
        return round(1.0 + 9.0 * raw_score, 1)
        
    except Exception as e:
        logger.error(f"[SCORING] Failed to evaluate spec {spec_name}: {e}")
        return 5.0

def build_hierarchical_pillar_profiles(ticker: str) -> Dict[str, Any]:
    """
    Calculate base scores, identify outlier active drivers, and assemble
    the Hierarchical Pillar Profiles for a stock.
    """
    ticker = ticker.upper()
    vars_dict = compute_normalized_features(ticker)
    
    # 1. Edge Pillar
    edge_score = calculate_pillar_score("edge_score", vars_dict)
    edge_drivers = []
    if vars_dict["ev_norm"] > 0.7:
        edge_drivers.append(f"EV Norm: {vars_dict['ev_norm']:.2f} (High Expected Value)")
    if vars_dict["rr_norm"] > 0.7:
        edge_drivers.append(f"R/R Ratio: {vars_dict['raw_rr']:.2f} (Attractive Risk/Reward)")
    if vars_dict["z_score_norm"] < 0.3:
        edge_drivers.append(f"Z-Score: {vars_dict['raw_z_score']:.2f} (Oversold Mean Reversion)")
        
    edge_label = "Neutral Setup"
    if edge_score >= 7.5:
        edge_label = "Exceptional Catalyst / Reversion Setup"
    elif edge_score >= 6.0:
        edge_label = "Decent Momentum Setup"
    elif edge_score <= 3.5:
        edge_label = "Unfavorable / Low Edge Setup"

    # 2. Risk Pillar
    risk_score = calculate_pillar_score("risk_score", vars_dict)
    risk_drivers = []
    veto_flags = []
    
    if vars_dict["raw_vol"] > 0.08:
        risk_drivers.append(f"Volatility: {vars_dict['raw_vol']*100:.1f}% (High Daily Volatility)")
    if vars_dict["raw_dd"] > 0.25:
        risk_drivers.append(f"Drawdown: {vars_dict['raw_dd']*100:.1f}% (Significant Drawdown from Peak)")
    if vars_dict["raw_kelly"] > 0.20:
        risk_drivers.append(f"Kelly Fraction: {vars_dict['raw_kelly']:.2f} (Aggressive Sizing)")
        
    # Check vetoes
    if risk_score <= 3.0:
        veto_flags.append("HIGH_TAIL_RISK_VETO")
    if vars_dict["raw_dd"] > 0.40:
        veto_flags.append("EXTREME_DRAWDOWN_VETO")

    risk_label = "Moderate Risk Profile"
    if risk_score >= 7.5:
        risk_label = "Highly Stable / Defensible Profile"
    elif risk_score <= 3.5:
        risk_label = "Speculative / High Variance Profile"

    # 3. Regime Pillar
    regime_score = calculate_pillar_score("regime_fit_score", vars_dict)
    regime_drivers = []
    if vars_dict["z_score_norm"] > 0.8:
        regime_drivers.append(f"Z-Score: {vars_dict['raw_z_score']:.2f} (Extreme Overbought)")
    if vars_dict["rsi_norm"] < 0.3:
        regime_drivers.append(f"RSI: {vars_dict['raw_rsi']:.1f} (Oversold)")
        
    regime_label = "Neutral Regime Fit"
    if regime_score >= 7.0:
        regime_label = "High Alignment with Active Trend/Regime"
    elif regime_score <= 3.5:
        regime_label = "Regime Mismatch / Trend Divergence"

    return {
        "ticker": ticker,
        "pillars": {
            "edge": {
                "base_score": edge_score,
                "profile_label": edge_label,
                "active_drivers": edge_drivers,
                "veto_flags": []
            },
            "risk": {
                "base_score": risk_score,
                "profile_label": risk_label,
                "active_drivers": risk_drivers,
                "veto_flags": veto_flags
            },
            "regime": {
                "base_score": regime_score,
                "profile_label": regime_label,
                "active_drivers": regime_drivers,
                "veto_flags": []
            }
        },
        "raw_features": vars_dict
    }

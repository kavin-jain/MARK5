"""
MARK5 Monte Carlo Bootstrap Simulation v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bootstrap resamples the daily P&L sequence from backtest results to estimate
the distribution of outcomes under different luck/ordering scenarios.

USAGE:
    python3 scripts/monte_carlo.py --n-sims 2000 --seed 42
    python3 scripts/monte_carlo.py --input reports/momentum_portfolio_results.json

OUTPUTS (printed + saved to reports/monte_carlo_results.json):
    - P(CAGR > 10%), P(CAGR > 15%), P(CAGR > 20%)
    - MaxDD distribution: P5/P50/P95
    - Before/after comparison if a prior results file is provided

CHANGELOG:
- [2026-05-26] v1.0: Initial creation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [MONTE_CARLO] | %(levelname)s | %(message)s",
)
logger = logging.getLogger("MARK5.MonteCarlo")


# ── Core simulation ───────────────────────────────────────────────────────────

def _extract_daily_returns(results: Dict) -> np.ndarray:
    """
    Extract daily return series from backtest results JSON.
    Falls back to fold-level returns if daily P&L not present.
    """
    # Prefer equity_curve if present (most reliable)
    eq = results.get("equity_curve")
    if eq and len(eq) > 5:
        eq_arr = np.array(eq, dtype=float)
        rets   = np.diff(eq_arr) / eq_arr[:-1]
        return rets[np.isfinite(rets)]

    # Fall back to fold returns
    fold_results = results.get("fold_results", [])
    if fold_results:
        rets = [f.get("portfolio_ret", 0.0) / 100.0 for f in fold_results if f.get("gate_blocked") is None]
        return np.array(rets, dtype=float)

    return np.array([0.0])


def _simulate_path(
    returns: np.ndarray,
    n_periods: int,
    initial_capital: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Bootstrap one simulated equity path.
    Returns (cagr_pct, max_drawdown_pct).
    """
    sampled = rng.choice(returns, size=n_periods, replace=True)
    equity  = np.cumprod(1 + np.insert(sampled, 0, 0.0)) * initial_capital
    equity[0] = initial_capital

    peak    = np.maximum.accumulate(equity)
    dd      = (equity - peak) / peak
    max_dd  = float(dd.min())

    n_years = n_periods / 252.0
    cagr    = (equity[-1] / initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    return float(cagr * 100), float(max_dd * 100)


def run_simulation(
    input_path: str,
    n_sims: int = 2000,
    seed: int = 42,
    period_years: float = 4.0,
    output_path: str = "reports/monte_carlo_results.json",
) -> Dict:

    if not os.path.exists(input_path):
        logger.error(f"Input not found: {input_path}")
        return {"status": "failed", "reason": f"file not found: {input_path}"}

    with open(input_path) as f:
        results = json.load(f)

    logger.info(f"Loaded backtest results from: {input_path}")

    daily_rets = _extract_daily_returns(results)
    if len(daily_rets) < 10:
        logger.error(f"Too few return observations: {len(daily_rets)}")
        return {"status": "failed", "reason": "insufficient return data"}

    initial_cap = results.get("summary", {}).get("initial_capital", 5_00_00_000.0)
    n_periods   = int(period_years * 252)
    rng         = np.random.default_rng(seed)

    logger.info(f"Running {n_sims} simulations | {period_years}yr | {len(daily_rets)} daily obs | seed={seed}")

    cagrs: List[float] = []
    max_dds: List[float] = []

    for _ in range(n_sims):
        cagr, max_dd = _simulate_path(daily_rets, n_periods, initial_cap, rng)
        cagrs.append(cagr)
        max_dds.append(max_dd)

    cagrs_arr  = np.array(cagrs)
    mdd_arr    = np.array(max_dds)

    p_cagr_10  = float(np.mean(cagrs_arr > 10.0)) * 100
    p_cagr_15  = float(np.mean(cagrs_arr > 15.0)) * 100
    p_cagr_20  = float(np.mean(cagrs_arr > 20.0)) * 100

    cagr_p5    = float(np.percentile(cagrs_arr, 5))
    cagr_p50   = float(np.percentile(cagrs_arr, 50))
    cagr_p95   = float(np.percentile(cagrs_arr, 95))
    mdd_p5     = float(np.percentile(mdd_arr, 5))   # less severe (closer to 0)
    mdd_p50    = float(np.percentile(mdd_arr, 50))
    mdd_p95    = float(np.percentile(mdd_arr, 95))  # most severe

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info(f"  Simulations:      {n_sims:,}")
    logger.info(f"  Period:           {period_years} years")
    logger.info(f"  Input returns:    {len(daily_rets)} observations")
    logger.info(f"  Mean daily ret:   {daily_rets.mean()*100:.4f}%")
    logger.info(f"  Std  daily ret:   {daily_rets.std()*100:.4f}%")
    logger.info("")
    logger.info("  ── CAGR Distribution ──────────────────")
    logger.info(f"  P5:               {cagr_p5:+.1f}%")
    logger.info(f"  P50 (median):     {cagr_p50:+.1f}%")
    logger.info(f"  P95:              {cagr_p95:+.1f}%")
    logger.info("")
    logger.info("  ── Probability Thresholds ─────────────")
    logger.info(f"  P(CAGR > 10%):    {p_cagr_10:.1f}%")
    logger.info(f"  P(CAGR > 15%):    {p_cagr_15:.1f}%")
    logger.info(f"  P(CAGR > 20%):    {p_cagr_20:.1f}%")
    logger.info("")
    logger.info("  ── MaxDD Distribution ─────────────────")
    logger.info(f"  P5  (best case):  {mdd_p5:.1f}%")
    logger.info(f"  P50 (median):     {mdd_p50:.1f}%")
    logger.info(f"  P95 (worst case): {mdd_p95:.1f}%")
    logger.info("═" * 60)

    mc_results = {
        "inputs": {
            "input_path":    input_path,
            "n_sims":        n_sims,
            "seed":          seed,
            "period_years":  period_years,
            "n_daily_obs":   len(daily_rets),
            "mean_daily_ret": round(float(daily_rets.mean()), 6),
            "std_daily_ret":  round(float(daily_rets.std()), 6),
        },
        "cagr": {
            "p5":  round(cagr_p5, 2),
            "p50": round(cagr_p50, 2),
            "p95": round(cagr_p95, 2),
        },
        "max_dd": {
            "p5":  round(mdd_p5, 2),
            "p50": round(mdd_p50, 2),
            "p95": round(mdd_p95, 2),
        },
        "probabilities": {
            "cagr_gt_10_pct": round(p_cagr_10, 1),
            "cagr_gt_15_pct": round(p_cagr_15, 1),
            "cagr_gt_20_pct": round(p_cagr_20, 1),
        },
        "raw": {
            "cagrs":   [round(c, 2) for c in cagrs],
            "max_dds": [round(d, 2) for d in max_dds],
        },
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mc_results, f, indent=2)
    logger.info(f"Results → {output_path}")

    return mc_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARK5 Monte Carlo Bootstrap")
    parser.add_argument("--input",        default="reports/momentum_portfolio_results.json")
    parser.add_argument("--n-sims",       type=int,   default=2000)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--period-years", type=float, default=4.0)
    parser.add_argument("--output",       default="reports/monte_carlo_results.json")
    args = parser.parse_args()

    run_simulation(
        input_path   = args.input,
        n_sims       = args.n_sims,
        seed         = args.seed,
        period_years = args.period_years,
        output_path  = args.output,
    )

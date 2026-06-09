"""
MARK6 — F3 Quality-Factor Test (real fundamentals from indianapi.in)
====================================================================
Tests whether adding TRUE fundamental quality (ROCE, low debt, FCF margin, earnings
stability — disclosure-lagged, causal) to the MARK6 blend adds robust net edge. Run
on the full window AND rolling walk-forward (twice = robustness). KEEP only if it beats
the price-only baseline both recent and on walk-forward average.

  python3 scripts/fundamentals_quality_test.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_quality_factors)
END = "2026-06-05"

CONFIGS = {
    "baseline":     {"momentum": .30, "low_vol": .30, "trend": .20, "stability": .20},
    "F3_q_light":   {"momentum": .26, "low_vol": .26, "trend": .18, "stability": .15,
                     "roce": .06, "low_debt": .05, "fcf_margin": .04},
    "F3_q_med":     {"momentum": .22, "low_vol": .22, "trend": .16, "stability": .12,
                     "roce": .10, "low_debt": .09, "fcf_margin": .06, "earn_stability": .03},
    "F3_q_heavy":   {"momentum": .18, "low_vol": .18, "trend": .12, "stability": .10,
                     "roce": .16, "low_debt": .14, "fcf_margin": .08, "earn_stability": .04},
    "quality_pure": {"momentum": .10, "low_vol": .15, "trend": .05, "stability": .10,
                     "roce": .25, "low_debt": .20, "fcf_margin": .10, "earn_stability": .05},
}
QNAMES = {"roce", "low_debt", "fcf_margin", "earn_stability"}


def build(panel, weights, qf):
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.125, factor_weights=weights)
    needs = bool(set(weights) & QNAMES)
    return Backtester(panel, PortfolioConstructor(cfg), extra_factors=qf if needs else None)


def main():
    panel = DataPanel(discover_tickers(), END)
    qf = load_quality_factors()
    print(f"universe={len(panel.tickers)}  fundamentals tickers={len(qf)}\n")
    if len(qf) < 50:
        print("Fundamentals fetch incomplete (<50) — wait for fetch_fundamentals.py.")
    bts = {n: build(panel, w, qf) for n, w in CONFIGS.items()}

    # headline windows
    wins = [("2016-01-01", "2021-12-31", "holdout16-21"),
            ("2022-01-01", END, "recent22-26"), ("2016-01-01", END, "full16-26")]
    base = {}
    for s, e, lab in wins:
        base[lab] = bts["baseline"].run(s, e)["metrics"]["cagr"]
    print(f"  {'config':<13}{'window':<14}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'vsBase':>8}")
    res = {}
    for n, bt in bts.items():
        for s, e, lab in wins:
            m = bt.run(s, e)["metrics"]
            vs = (m["cagr"] - base[lab]) * 100
            res[(n, lab)] = (m["cagr"], vs)
            tag = "" if n == "baseline" else f"{vs:>+7.1f}"
            print(f"  {n:<13}{lab:<14}{m['cagr']*100:>+7.1f}%{m['sharpe']:>8.2f}{m['max_dd']*100:>+7.1f}%{tag:>8}")
        print()

    # walk-forward (robustness)
    print("  ROLLING 3-YR WALK-FORWARD vs baseline (avgΔ, beats):")
    years = list(range(2016, 2024))
    bwf = {y: bts["baseline"].run(f"{y}-01-01", min(f"{y+2}-12-31", END))["metrics"]["cagr"] for y in years}
    for n, bt in bts.items():
        if n == "baseline":
            print(f"    baseline avg {np.mean(list(bwf.values()))*100:+.1f}%"); continue
        ds = [bt.run(f"{y}-01-01", min(f"{y+2}-12-31", END))["metrics"]["cagr"] - bwf[y] for y in years]
        beats = sum(1 for x in ds if x > 0.002)
        print(f"    {n:<13} avgΔ {np.mean(ds)*100:>+5.1f}pp  beats {beats}/{len(years)}")

    print("\n  VERDICT:")
    for n in CONFIGS:
        if n == "baseline":
            continue
        rec = res[(n, "recent22-26")][1]
        ds = [bts[n].run(f"{y}-01-01", min(f"{y+2}-12-31", END))["metrics"]["cagr"] - bwf[y] for y in years]
        wf = np.mean(ds) * 100
        keep = rec > 0 and wf > 0
        print(f"    {n:<13} recent {rec:>+5.1f}pp | wf {wf:>+5.1f}pp | "
              f"{'KEEP candidate' if keep else 'no robust edge'}")


if __name__ == "__main__":
    main()

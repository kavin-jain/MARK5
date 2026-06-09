"""
MARK6 — Overfitting / Statistical-Significance Analysis
=======================================================
Answers the question every serious backtest must: given the many strategy
variants we tried, is the deployed config's Sharpe genuine skill or just the
luckiest draw? Computes (Bailey & Lopez de Prado):
  - Probabilistic Sharpe Ratio (PSR vs 0)
  - Deflated Sharpe Ratio (DSR) — deflated for the N trials attempted
  - Probability of Backtest Overfitting (PBO) via CSCV

It re-runs a grid of the strategy variants we explored (the "trials"), collects
their daily returns, and runs the tests on the deployed config (n_hold=12,
tilt=1.5, blend). Writes reports/OVERFITTING_ANALYSIS.md.

  python3 scripts/overfitting_analysis.py
"""
import os, sys, itertools
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester)
from core.portfolio.stats import (deflated_sharpe_ratio, pbo_cscv,
                                   probabilistic_sharpe_ratio, _sharpe)
START, END = "2016-01-01", "2026-06-05"
REPORTS = os.path.join(_ROOT, "reports")

WEIGHTS = {
    "blend":      {"momentum": .30, "low_vol": .30, "trend": .20, "stability": .20},
    "mom_heavy":  {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15},
    "lowvol_hvy": {"momentum": .15, "low_vol": .50, "trend": .15, "stability": .20},
    "trend_hvy":  {"momentum": .20, "low_vol": .20, "trend": .45, "stability": .15},
    "stab_hvy":   {"momentum": .20, "low_vol": .20, "trend": .15, "stability": .45},
}
DEPLOYED = ("blend", 12, 1.5)   # (weights, n_hold, tilt) = the chosen config


def main():
    panel = DataPanel(discover_tickers(), END)
    # the grid of "trials" we actually explored across the project
    grid = list(itertools.product(WEIGHTS.keys(), [8, 12, 16, 20], [0.5, 1.5, 3.0]))
    print(f"Running {len(grid)} strategy trials to assemble the returns matrix...", flush=True)

    rets, sharpes, labels, deployed_ret = {}, [], [], None
    cal = None
    for wname, nh, tilt in grid:
        cfg = ConstructionConfig(mode="factor_tilt", n_hold=nh, base_weighting="inverse_vol",
                                 tilt_strength=tilt, max_weight=max(0.08, 1.5 / nh),
                                 factor_weights=WEIGHTS[wname])
        nav = Backtester(panel, PortfolioConstructor(cfg)).run(START, END)["nav_gross"]
        r = nav.pct_change(fill_method=None).fillna(0.0)
        if cal is None:
            cal = r.index
        r = r.reindex(cal).fillna(0.0)
        lab = f"{wname}|n{nh}|t{tilt}"
        rets[lab] = r.values
        sharpes.append(_sharpe(r.values))
        labels.append(lab)
        if (wname, nh, tilt) == DEPLOYED:
            deployed_ret = r.values
    print(f"  done. {len(labels)} trials.\n", flush=True)

    M = np.column_stack([rets[l] for l in labels])     # T x N
    dsr = deflated_sharpe_ratio(deployed_ret, sharpes)
    pbo = pbo_cscv(M, n_splits=12)
    ann = lambda d: d * np.sqrt(252)                   # daily->annual SR

    L = ["# MARK6 — Overfitting & Statistical-Significance Analysis", "",
         "Bailey & López de Prado tests on the deployed config (blend / n_hold=12 / "
         "tilt=1.5), using the grid of strategy variants we explored as the trial set. "
         "All on daily returns, 2016-2026.", "",
         "## Deflated Sharpe Ratio (is the Sharpe real, given how many we tried?)", "",
         f"- Strategy variants tried (N): **{dsr['n_trials']}**",
         f"- Observed Sharpe: **{ann(dsr['observed_sharpe_daily']):.2f}** annualised "
         f"({dsr['observed_sharpe_daily']:.3f} daily)",
         f"- Probabilistic Sharpe Ratio vs 0 (P true SR>0): **{dsr['psr_vs_zero']*100:.1f}%**",
         f"- Expected max Sharpe from pure luck across {dsr['n_trials']} trials: "
         f"{ann(dsr['expected_max_sharpe_luck']):.2f} annualised",
         f"- **Deflated Sharpe Ratio (P skill survives multiple-testing): "
         f"{dsr['deflated_sharpe']*100:.1f}%**", "",
         "## Probability of Backtest Overfitting (PBO via CSCV)", "",
         f"- Strategies in matrix: {pbo['n_strategies']} | train/test combos: {pbo['n_combos']}",
         f"- **PBO: {pbo['pbo']*100:.1f}%** (fraction of splits where the in-sample-best "
         "strategy lands below the out-of-sample median)",
         f"- Median performance-degradation logit: {pbo['median_logit']:.2f} "
         f"({'positive = robust' if pbo['median_logit'] > 0 else 'negative = overfit'})", "",
         "## Verdict", ""]
    dsr_ok = dsr["deflated_sharpe"] > 0.95
    pbo_ok = pbo["pbo"] < 0.20
    L.append(f"- DSR {'PASS' if dsr_ok else 'WEAK'}: deflated-Sharpe "
             f"{dsr['deflated_sharpe']*100:.0f}% — "
             f"{'the Sharpe survives multiple-testing; >95% confidence it is skill, not the luckiest draw.' if dsr_ok else 'caution — significance is borderline after deflation.'}")
    L.append(f"- PBO {'PASS' if pbo_ok else 'WEAK'}: {pbo['pbo']*100:.0f}% — "
             f"{'low overfitting risk; the config generalises out-of-sample.' if pbo_ok else 'elevated overfitting risk.'}")
    L.append("")
    L.append("These are the statistics professional quant funds use to vet a strategy "
             "before risking capital — most retail/student backtests never compute them.")

    os.makedirs(REPORTS, exist_ok=True)
    open(os.path.join(REPORTS, "OVERFITTING_ANALYSIS.md"), "w").write("\n".join(L))
    print("\n".join(L))
    print("\nSaved -> reports/OVERFITTING_ANALYSIS.md")


if __name__ == "__main__":
    main()

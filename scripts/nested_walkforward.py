"""
Nested (anchored) walk-forward — what does the system earn when the CONFIG ITSELF
is chosen out-of-sample?
================================================================================
The deployed config (momentum-heavy weights, n_hold=12, tilt 1.5, 126-bar refresh)
was chosen with full-sample knowledge and THEN walk-forward validated. Selection and
evaluation touched the same data, which is *flat* model selection — Cawley & Talbot
(JMLR 2010) showed this biases the reported OOS number upward and pushes you toward
over-complex models. The system's own PBO of 76.7% is the same warning.

This script removes that leak: for each test year, the config is re-selected using
ONLY data available before that year, then run on the unseen year, and the yearly
out-of-sample segments are chained into one equity curve (Pardo, "The Evaluation and
Optimization of Trading Strategies", ch. 11).

Reported: Walk-Forward Efficiency (WFE = OOS CAGR / IS CAGR; Pardo's pass bar is
0.50-0.60), config stability, and the IS->OOS rank correlation that says whether
config selection carries any signal at all.

HONEST EXPECTATION: this number should come out BELOW the headline. That is the
point — it is the credibility number, not a return improvement.

  python3 scripts/nested_walkforward.py
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as ss

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig, metrics)

START, END = "2016-01-01", "2026-06-09"
REPORTS = os.path.join(_ROOT, "reports")
TEST_YEARS = list(range(2020, 2026))          # each needs >=4y of prior data to select on
WEIGHTS = {
    "blend":      {"momentum": .30, "low_vol": .30, "trend": .20, "stability": .20},
    "mom_heavy":  {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15},
    "lowvol_hvy": {"momentum": .15, "low_vol": .50, "trend": .15, "stability": .20},
    "trend_hvy":  {"momentum": .20, "low_vol": .20, "trend": .45, "stability": .15},
    "stab_hvy":   {"momentum": .20, "low_vol": .20, "trend": .15, "stability": .45},
}
DEPLOYED = "mom_heavy|n12|r126"


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    bt = Backtester(panel, PortfolioConstructor(ConstructionConfig()), BacktestConfig())
    grid = [(w, n, r) for w, n, r in itertools.product(WEIGHTS, [8, 12, 16, 20], [126, 252])]
    print(f"Running {len(grid)} configs once each over {START}..{END} "
          f"(the NAV matrix makes every selection rule free to evaluate)...", flush=True)

    navs = {}
    for i, (w, n, r) in enumerate(grid, 1):
        bt.con = PortfolioConstructor(ConstructionConfig(
            mode="factor_tilt", n_hold=n, base_weighting="inverse_vol", tilt_strength=1.5,
            max_weight=max(0.08, 1.5 / n), factor_weights=WEIGHTS[w]))
        bt.cfg = BacktestConfig(rebal_bars=r)
        navs[f"{w}|n{n}|r{r}"] = bt.run(START, END)["nav_net"]
        if i % 10 == 0:
            print(f"  {i}/{len(grid)}", flush=True)

    nav_df = pd.DataFrame(navs)
    ret = nav_df.pct_change(fill_method=None).fillna(0.0)

    def seg_cagr(label_ret, lo, hi):
        r = label_ret.loc[lo:hi]
        if len(r) < 20:
            return np.nan
        n = (1 + r).prod()
        return n ** (252 / len(r)) - 1

    # ── anchored selection: choose on [START, Y-1], trade Y ────────────────────
    print("\n" + "=" * 88)
    print("  ANCHORED NESTED WALK-FORWARD — config re-chosen each year on prior data only")
    print("=" * 88)
    print(f"  {'year':<6}{'selected config':<22}{'IS CAGR':>9}{'OOS CAGR':>10}"
          f"{'deployed':>10}{'1/N ens':>9}")
    chained, chained_dep, chained_ens, picks, is_oos = [], [], [], [], []
    for y in TEST_YEARS:
        is_lo, is_hi = START, f"{y-1}-12-31"
        oos_lo, oos_hi = f"{y}-01-01", min(f"{y}-12-31", END)
        is_c = {c: seg_cagr(ret[c], is_lo, is_hi) for c in ret.columns}
        oos_c = {c: seg_cagr(ret[c], oos_lo, oos_hi) for c in ret.columns}
        if not np.isfinite(list(oos_c.values())).any():
            continue
        best = max(is_c, key=lambda c: (is_c[c] if np.isfinite(is_c[c]) else -9))
        ens = float(np.nanmean([oos_c[c] for c in ret.columns]))
        picks.append(best)
        chained.append(oos_c[best]); chained_dep.append(oos_c[DEPLOYED]); chained_ens.append(ens)
        is_oos.append((list(is_c.values()), list(oos_c.values())))
        print(f"  {y:<6}{best:<22}{is_c[best]*100:>8.1f}%{oos_c[best]*100:>9.1f}%"
              f"{oos_c[DEPLOYED]*100:>9.1f}%{ens*100:>8.1f}%")

    def chain(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return (np.prod([1 + x for x in xs]) ** (1 / len(xs)) - 1) if xs else np.nan

    sel, dep, ens = chain(chained), chain(chained_dep), chain(chained_ens)
    is_of_picks = chain([seg_cagr(ret[p], START, f"{y-1}-12-31")
                         for p, y in zip(picks, TEST_YEARS[:len(picks)])])
    rho = np.nanmean([ss.spearmanr(a, b).statistic for a, b in is_oos])

    print("-" * 88)
    print(f"  CHAINED OOS CAGR   selection rule {sel*100:+.2f}%   "
          f"deployed-config {dep*100:+.2f}%   1/N ensemble {ens*100:+.2f}%")
    print(f"  Walk-Forward Efficiency (OOS/IS) = {sel/is_of_picks:.2f}  "
          f"(Pardo pass bar 0.50-0.60)  -> {'PASS' if sel/is_of_picks >= 0.5 else 'FAIL'}")
    print(f"  Config stability: {len(set(picks))} distinct configs chosen across "
          f"{len(picks)} years {picks}")
    print(f"  Mean IS->OOS config rank correlation (Spearman) = {rho:+.3f}")
    print(f"    {'selection carries signal' if rho > 0.3 else 'SELECTION IS NOISE — picking a config on past performance does not predict future rank'}")

    verdict = ("Config selection adds nothing; the honest deployment is a fixed, "
               "economically-motivated config (or the 1/N ensemble), NOT a learned choice."
               if rho <= 0.3 or sel <= dep else
               "Config selection carries some signal.")
    print(f"\n  VERDICT: {verdict}")

    os.makedirs(REPORTS, exist_ok=True)
    out = {"chained_oos_selection": sel, "chained_oos_deployed": dep,
           "chained_oos_ensemble": ens, "walk_forward_efficiency": sel / is_of_picks,
           "is_oos_rank_corr": rho, "picks": picks, "test_years": TEST_YEARS[:len(picks)],
           "verdict": verdict}
    json.dump(out, open(os.path.join(REPORTS, "nested_walkforward.json"), "w"),
              indent=2, default=float)
    print(f"  Saved -> reports/nested_walkforward.json")


if __name__ == "__main__":
    main()

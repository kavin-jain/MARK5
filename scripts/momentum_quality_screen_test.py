"""
MARK6 — Phase C exploit test (2026-06-10): momentum-heavy book x quality SCREEN
===============================================================================
Two surviving levers from the research log, tested individually and combined,
on the CLEAN uniform-end data (post refetch_all):

  1. n12_mom        — momentum-heavy concentrated book (P5 momentum variant:
                      already 7/8 walk-forward windows, +2.6pp avg in the 2026-06-08
                      risk-dial test; re-validated here on clean data).
  2. quality SCREEN — exclude the bottom-q quality names (ROCE / low-debt /
                      FCF / earnings stability, disclosure-lagged) BEFORE factor
                      ranking. Distinct from quality-as-TILT, which was K15-killed.
                      Fail-open: unscored names pass through (no coverage bias).

KEEP bar (Operating Mandate): beats baseline on recent window AND walk-forward
average, net of tax, with no walk-forward collapse (>= 5/8 windows).

  python3 scripts/momentum_quality_screen_test.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_quality_factors)

END = "2026-06-09"
BLEND = {"momentum": .30, "low_vol": .30, "trend": .20, "stability": .20}
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
QCOLS = ("roce", "low_debt", "fcf_margin", "earn_stability")


def make_quality_screen(qf: dict, drop_q: float = 0.30):
    """Point-in-time exclusion of the bottom `drop_q` quality quantile.

    Scores only names with a disclosure strictly before `asof` (causal); unscored
    names always pass (fail-open). Cross-sectional z per factor, mean of available.
    """
    def screen(asof: pd.Timestamp, elig: list[str]) -> list[str]:
        rows = {}
        for t in elig:
            df = qf.get(t)
            if df is None:
                continue
            past = df.loc[:asof]
            if past.empty:
                continue
            rows[t] = past.iloc[-1]
        if len(rows) < 20:           # too few scored names to screen meaningfully
            return elig
        panel = pd.DataFrame(rows).T
        z = pd.DataFrame(index=panel.index)
        for c in QCOLS:
            if c in panel.columns:
                col = pd.to_numeric(panel[c], errors="coerce")
                sd = col.std()
                if sd and sd > 0:
                    z[c] = (col - col.mean()) / sd
        if z.empty:
            return elig
        score = z.mean(axis=1).dropna()
        cut = score.quantile(drop_q)
        bad = set(score[score < cut].index)
        return [t for t in elig if t not in bad]
    return screen


def build(panel, weights, screen=None):
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.125, factor_weights=weights)
    return Backtester(panel, PortfolioConstructor(cfg), screen=screen)


def main():
    panel = DataPanel(discover_tickers(), END)
    qf = load_quality_factors()
    print(f"universe={len(panel.tickers)}  fundamentals coverage={len(qf)}  "
          f"stale={len(panel.stale_tickers)}\n")
    screen = make_quality_screen(qf)
    bts = {
        "baseline_blend": build(panel, BLEND),
        "mom_heavy":      build(panel, MOM),
        "blend_qscreen":  build(panel, BLEND, screen),
        "mom_qscreen":    build(panel, MOM, screen),
    }

    wins = [("2016-01-01", "2021-12-31", "holdout16-21"),
            ("2022-01-01", END, "recent22-26"),
            ("2016-01-01", END, "full16-26")]
    base = {lab: bts["baseline_blend"].run(s, e)["metrics"] for s, e, lab in wins}
    print(f"  {'config':<15}{'window':<14}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'vsBase':>8}")
    for n, bt in bts.items():
        for s, e, lab in wins:
            m = bt.run(s, e)["metrics"]
            vs = (m["cagr"] - base[lab]["cagr"]) * 100
            tag = "" if n == "baseline_blend" else f"{vs:>+7.1f}"
            print(f"  {n:<15}{lab:<14}{m['cagr']*100:>+7.1f}%{m['sharpe']:>8.2f}"
                  f"{m['max_dd']*100:>+7.1f}%{tag:>8}")
        print()

    print("  ROLLING 3-YR WALK-FORWARD vs baseline_blend (avgΔ net CAGR, beats):")
    years = list(range(2016, 2024))
    bwf = {y: bts["baseline_blend"].run(f"{y}-01-01", min(f"{y+2}-12-31", END))["metrics"]["cagr"]
           for y in years}
    print(f"    baseline avg {np.mean(list(bwf.values()))*100:+.1f}%")
    for n, bt in bts.items():
        if n == "baseline_blend":
            continue
        ds = [bt.run(f"{y}-01-01", min(f"{y+2}-12-31", END))["metrics"]["cagr"] - bwf[y]
              for y in years]
        beats = sum(1 for x in ds if x > 0.002)
        print(f"    {n:<15} avgΔ {np.mean(ds)*100:>+5.1f}pp  beats {beats}/{len(years)}"
              f"   per-window: {[f'{x*100:+.1f}' for x in ds]}")


if __name__ == "__main__":
    main()

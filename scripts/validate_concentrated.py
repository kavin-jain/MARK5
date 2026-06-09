"""
Validate the concentrated config across rolling 3-yr walk-forward windows before
adopting it. A higher full-period CAGR is worthless if it only came from one
regime. KEEP only if it beats the current baseline in most windows.
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester)
END = "2026-05-21"
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}

CANDS = {
    "baseline_n20":  dict(n_hold=20, tilt_strength=0.5, max_weight=0.08),
    "n12_blend":     dict(n_hold=12, tilt_strength=1.5, max_weight=0.125),
    "n12_mom":       dict(n_hold=12, tilt_strength=1.5, max_weight=0.125, factor_weights=MOM),
}


def main():
    panel = DataPanel(discover_tickers(), END)
    bts = {}
    for name, kw in CANDS.items():
        cfg = ConstructionConfig(mode="factor_tilt", base_weighting="inverse_vol", **kw)
        bts[name] = Backtester(panel, PortfolioConstructor(cfg))
    years = list(range(2016, 2024))
    print(f"  {'window':<12}" + "".join(f"{n:>14}" for n in CANDS))
    agg = {n: [] for n in CANDS}
    dd = {n: [] for n in CANDS}
    for y in years:
        s, e = f"{y}-01-01", min(f"{y+2}-12-31", END)
        row = f"  {y}-{y+2:<7}"
        for n, bt in bts.items():
            m = bt.run(s, e)["metrics"]
            agg[n].append(m["cagr"]); dd[n].append(m["max_dd"])
            row += f"{m['cagr']*100:>+8.1f}%/{m['max_dd']*100:>+4.0f}"
        print(row)
    print("\n  " + "-"*70)
    base = np.mean(agg["baseline_n20"])
    for n in CANDS:
        avg = np.mean(agg[n]); wdd = np.min(dd[n])
        beats = sum(1 for i in range(len(years)) if agg[n][i] > agg["baseline_n20"][i] + 0.002)
        tag = "" if n == "baseline_n20" else f"  beats baseline {beats}/{len(years)}, avgΔ {(avg-base)*100:+.1f}pp"
        print(f"  {n:<14} avg net CAGR {avg*100:+.1f}%  worst DD {wdd*100:+.0f}%{tag}")


if __name__ == "__main__":
    main()

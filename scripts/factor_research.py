"""
MARK6 — Factor Research Harness (F2 / F3 / F6)
==============================================
Tests candidate factor extensions against the proven baseline MARK6 blend, all
NET of Indian tax & costs, on the true holdout (2016-2021), the recent window
(2022-2026), and a rolling 3-yr walk-forward. A variant is only worth KEEPing if
it beats the baseline robustly — not on one cherry-picked window.

Variants:
  baseline      : current MARK6 (momentum .30 / low_vol .30 / trend .20 / stability .20)
  F2_lowvol     : heavier low-vol tilt (the strongest documented Indian anomaly)
  F2_lowvol_max : low-vol dominant
  F3_quality    : add promoter_level (governance/skin-in-game quality proxy)
  F6_promoter   : add promoter_chg (the one ownership signal with consistent +IC)
  F3+F6         : both shareholding factors
  inst_chg      : sanity control (I1 says IC≈0 -> should NOT help)

Shareholding factors only exist from ~2018 (real disclosure dates); pre-2018 they
are neutral. So judge them mainly on 2022-2026 and the 2019+ walk-forward windows.

  python3 scripts/factor_research.py
"""
import os, sys, json
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_external_factors)

END = "2026-05-21"
REPORTS = os.path.join(_ROOT, "reports")

# weight sets (price factors must keep their relative shape; external added on top)
CONFIGS = {
    "baseline":      {"momentum": .30, "low_vol": .30, "trend": .20, "stability": .20},
    "F2_lowvol":     {"momentum": .20, "low_vol": .45, "trend": .15, "stability": .20},
    "F2_lowvol_max": {"momentum": .15, "low_vol": .60, "trend": .10, "stability": .15},
    "F3_quality":    {"momentum": .25, "low_vol": .25, "trend": .15, "stability": .20, "promoter_level": .15},
    "F6_promoter":   {"momentum": .25, "low_vol": .25, "trend": .20, "stability": .15, "promoter_chg": .15},
    "F3+F6":         {"momentum": .22, "low_vol": .24, "trend": .15, "stability": .17, "promoter_level": .12, "promoter_chg": .10},
    "inst_chg_ctrl": {"momentum": .25, "low_vol": .25, "trend": .20, "stability": .15, "inst_chg": .15},
}
EXTERNAL = {"promoter_level", "promoter_chg", "inst_chg"}
WINDOWS = [("2016-01-01", "2021-12-31", "holdout16-21"),
           ("2022-01-01", END, "recent22-26"),
           ("2016-01-01", END, "full16-26")]


def build_bt(panel, weights, ext):
    needs_ext = bool(set(weights) & EXTERNAL)
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=0.5, max_weight=0.08, factor_weights=weights)
    return Backtester(panel, PortfolioConstructor(cfg),
                      extra_factors=ext if needs_ext else None)


def main():
    print("Loading panel + external factors...", flush=True)
    panel = DataPanel(discover_tickers(), END)
    ext = load_external_factors()
    print(f"  universe={len(panel.tickers)}  shareholding tickers={len(ext)}\n", flush=True)

    bts = {name: build_bt(panel, w, ext) for name, w in CONFIGS.items()}
    base = bts["baseline"]

    # ── headline windows ──────────────────────────────────────────────────────
    res = {"windows": {}, "walk_forward": {}}
    print("="*94)
    print(f"  {'config':<15}{'window':<14}{'netCAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'vs base':>9}")
    print("="*94)
    base_cagr = {}
    for s, e, lab in WINDOWS:
        bm = base.run(s, e)["metrics"]; base_cagr[lab] = bm["cagr"]
    for name, bt in bts.items():
        res["windows"][name] = {}
        for s, e, lab in WINDOWS:
            m = bt.run(s, e)["metrics"]
            vs = (m["cagr"] - base_cagr[lab]) * 100
            res["windows"][name][lab] = {"cagr": m["cagr"], "sharpe": m["sharpe"],
                                         "max_dd": m["max_dd"], "vs_base_pp": vs}
            tag = "" if name == "baseline" else f"{vs:>+8.1f}"
            print(f"  {name:<15}{lab:<14}{m['cagr']*100:>+8.1f}%{m['sharpe']:>8.2f}"
                  f"{m['max_dd']*100:>+7.1f}%{tag:>9}")
        print()

    # ── rolling 3-yr walk-forward: how often does each beat baseline? ──────────
    print("="*94)
    print("  ROLLING 3-YR WALK-FORWARD — net CAGR vs baseline (shareholding factors live 2019+)")
    print("="*94)
    years = list(range(2016, 2024))
    base_wf = {}
    for y in years:
        s, e = f"{y}-01-01", min(f"{y+2}-12-31", END)
        base_wf[y] = base.run(s, e)["metrics"]["cagr"]
    print(f"  {'config':<15}" + "".join(f"{str(y)[2:]:>7}" for y in years) + f"{'avgΔ':>8}{'beats':>7}")
    for name, bt in bts.items():
        deltas, line = [], f"  {name:<15}"
        for y in years:
            s, e = f"{y}-01-01", min(f"{y+2}-12-31", END)
            c = bt.run(s, e)["metrics"]["cagr"]
            d = (c - base_wf[y]) * 100
            deltas.append(d)
            line += f"{d:>+7.1f}" if name != "baseline" else f"{c*100:>+7.1f}"
        avg = np.mean(deltas); beats = sum(1 for d in deltas if d > 0.05)
        res["walk_forward"][name] = {"avg_delta_pp": avg, "beats_base": beats, "n": len(years)}
        line += f"{avg:>+8.1f}{beats:>5}/{len(years)}" if name != "baseline" else f"{'—':>8}{'—':>7}"
        print(line)

    os.makedirs(REPORTS, exist_ok=True)
    json.dump(res, open(os.path.join(REPORTS, "factor_research.json"), "w"), indent=2, default=float)
    print(f"\nSaved -> reports/factor_research.json")

    # ── verdict ───────────────────────────────────────────────────────────────
    print("\n" + "="*94 + "\n  VERDICT (KEEP only if beats baseline robustly: recent>0 AND walk-forward avgΔ>0)\n" + "="*94)
    for name in CONFIGS:
        if name == "baseline":
            continue
        rec = res["windows"][name]["recent22-26"]["vs_base_pp"]
        wf = res["walk_forward"][name]["avg_delta_pp"]
        keep = rec > 0 and wf > 0
        print(f"  {name:<15} recent {rec:>+5.1f}pp | walk-fwd avgΔ {wf:>+5.1f}pp | "
              f"{'✅ KEEP candidate' if keep else '❌ KILL (no robust edge)'}")


if __name__ == "__main__":
    main()

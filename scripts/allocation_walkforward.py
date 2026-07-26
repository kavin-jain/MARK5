"""
Is the Sharpe-1.1 allocation reachable OUT of sample, or only with hindsight?
============================================================================
sharpe_ceiling.py showed the best long-only mix of {equity book, gold, Nasdaq}
scores Sharpe 1.16 in sample, above the 1.1 target, at roughly eq 24 / gold 46 /
US 30 rather than the deployed 50/25/25. That is an in-sample optimum computed
with perfect hindsight and must never be deployed as-is.

P15 already established, for the equity CONFIG, that in-sample selection has
essentially no predictive power out of sample (rank correlation -0.126) and that
a 1/N ensemble beat the learned rule. The same question has never been asked of
the ALLOCATION. This asks it.

Two experiments, both net of tax and costs through the real wrapper:

  A. LEARNED     re-pick the allocation every year using only prior data
                 (max-Sharpe on an expanding window), hold it the next year,
                 chain the out-of-sample years together. If this beats the fixed
                 deployment, allocation selection carries information. If it does
                 not, the in-sample 1.16 is a mirage and fixed weights win.

  B. FIXED GRID  a coarse grid of fixed allocations, each evaluated on the SAME
                 rolling 3-year walk-forward the rest of this repo uses. This
                 asks a different and more useful question: is there a fixed,
                 economically-motivated allocation that robustly beats 50/25/25
                 on Sharpe / MaxDD / Calmar?

The bar for B is the usual one: >=6/8 windows, per metric.

  MARK5_CACHE=data/pit_cache python3 scripts/allocation_walkforward.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            load_ohlcv, load_nifty, metrics, load_sector_map)

START, END = "2016-01-01", "2026-07-21"
TD, RF, TAX = 252, 0.065, 0.15
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
KEYS = ["eq", "GOLDBEES", "MON100"]
DEPLOYED = (0.50, 0.25, 0.25)

# economically-motivated fixed allocations, NOT a fine-grained search.
# Coarse steps on purpose: a 1% grid would just be curve-fitting the past.
GRID = [(.50, .25, .25), (.40, .30, .30), (.34, .33, .33), (.30, .40, .30),
        (.25, .45, .30), (.40, .40, .20), (.30, .35, .35), (.60, .20, .20),
        (.20, .50, .30), (.35, .40, .25)]


def blend(R: pd.DataFrame, w, rebal=TD, tax=TAX):
    """Fixed-weight sleeve blend, rebalanced every `rebal` bars, terminal tax.
    Identical mechanics to scripts/export_dashboard.py so results are comparable."""
    cur = {k: w[i] for i, k in enumerate(KEYS)}
    nav, out = 1.0, {}
    idx = R.index
    for i, d in enumerate(idx):
        if i > 0:
            prev = sum(cur.values())
            for j, k in enumerate(KEYS):
                cur[k] *= (1 + R[k].iloc[i])
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % rebal == 0:
            tot = sum(cur.values())
            cur = {k: tot * w[j] for j, k in enumerate(KEYS)}
    s = pd.Series(out)
    net = s.copy()
    net.iloc[-1] = s.iloc[-1] - max(0.0, s.iloc[-1] - 1) * tax
    return net


def max_sharpe_long_only(R: pd.DataFrame, seed=3):
    """Long-only max-Sharpe on the given return window (projected gradient)."""
    mu = R.mean().values * TD - RF
    S = R.cov().values * TD
    n = len(mu)
    rng = np.random.default_rng(seed)
    best, bs = None, -np.inf
    for _ in range(8):
        w = rng.dirichlet(np.ones(n))
        step = 0.02
        for _ in range(12000):
            var = w @ S @ w
            if var <= 0:
                break
            sd = np.sqrt(var)
            w = w + step * (mu / sd - (mu @ w) * (S @ w) / sd ** 3)
            w = np.maximum(w, 0.0)
            t = w.sum()
            if t <= 0:
                break
            w /= t
            step *= 0.9997
        var = w @ S @ w
        s = (mu @ w) / np.sqrt(var) if var > 0 else -np.inf
        if s > bs:
            best, bs = w.copy(), s
    return best


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=MOM)
    run = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                     BacktestConfig(rebal_bars=126, top_n_liquid=300)).run(START, END)
    eq = run["nav_gross"]
    cal = eq.index
    R = pd.DataFrame({
        "eq": eq.pct_change(fill_method=None),
        "GOLDBEES": load_ohlcv("GOLDBEES")["close"].astype(float)
                    .reindex(cal, method="ffill").pct_change(fill_method=None),
        "MON100": load_ohlcv("MON100")["close"].astype(float)
                  .reindex(cal, method="ffill").pct_change(fill_method=None),
    }).dropna()

    # ── A. learned allocation, re-picked yearly on prior data only ───────────
    print("\n" + "=" * 96)
    print("  A. LEARNED ALLOCATION — re-picked each year on PRIOR data only")
    print("=" * 96)
    print(f"  {'year':<8}{'weights chosen on prior data':<38}{'that year: system':>20}"
          f"{'fixed 50/25/25':>18}")
    print("  " + "-" * 92)
    years = sorted({d.year for d in R.index})
    learned, fixed, rows = [], [], []
    for y in years:
        prior = R[R.index.year < y]
        cur = R[R.index.year == y]
        if len(prior) < 2 * TD or len(cur) < 60:
            continue
        w = max_sharpe_long_only(prior)
        rl = float((cur @ w).sum())
        rf_ = float((cur @ np.array(DEPLOYED)).sum())
        learned.append(rl)
        fixed.append(rf_)
        wtxt = "  ".join(f"{k} {x*100:.0f}%" for k, x in zip(KEYS, w))
        print(f"  {y:<8}{wtxt:<38}{rl*100:>19.1f}%{rf_*100:>17.1f}%")
        rows.append({"year": int(y), "weights": dict(zip(KEYS, map(float, w))),
                     "learned_ret_pct": rl * 100, "fixed_ret_pct": rf_ * 100})
    print("  " + "-" * 92)
    wins = sum(1 for a, b in zip(learned, fixed) if a > b)
    print(f"  learned mean {np.mean(learned)*100:+.2f}%/yr   fixed mean "
          f"{np.mean(fixed)*100:+.2f}%/yr   learned beats fixed in "
          f"{wins}/{len(learned)} years")
    verdict_a = ("allocation selection CARRIES information"
                 if wins > len(learned) * 0.6 else
                 "allocation selection is NOISE — the in-sample optimum does not persist")
    print(f"  -> {verdict_a}")

    # ── B. fixed allocations on the standard rolling walk-forward ────────────
    print("\n" + "=" * 96)
    print("  B. FIXED ALLOCATIONS — rolling 3-year walk-forward, net of tax")
    print("=" * 96)
    windows = []
    for y0 in range(2016, 2024):
        e = f"{y0+2}-12-31"
        if pd.Timestamp(e) > pd.Timestamp(END):
            e = END
        windows.append((f"{y0}-01-01", e))

    base_w = DEPLOYED
    res = {}
    for w in GRID:
        full = metrics(blend(R, w))
        wf = []
        for s, e in windows:
            seg = R.loc[s:e]
            if len(seg) < 200:
                continue
            wf.append(metrics(blend(seg, w)))
        res[w] = {"full": full, "wf": wf}
    bwf = res[base_w]["wf"]

    print(f"  {'eq/gold/US':<14}{'CAGR':>8}{'shExc':>8}{'vol':>7}{'MaxDD':>8}{'Calmar':>8}"
          f"   {'wf wins':>8}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}")
    print("  " + "-" * 92)
    for w in GRID:
        m = res[w]["full"]
        cells = []
        for k in ("sharpe_excess", "max_dd", "calmar"):
            n = sum(1 for a, b in zip(res[w]["wf"], bwf) if a[k] > b[k])
            cells.append(f"{n}/{len(bwf)}")
        tag = ["", "", ""] if w == base_w else cells
        mark = "  <- deployed" if w == base_w else ""
        print(f"  {w[0]*100:.0f}/{w[1]*100:.0f}/{w[2]*100:.0f}".ljust(16)
              + f"{m['cagr']*100:>+7.2f}%{m['sharpe_excess']:>8.2f}{m['vol']*100:>6.1f}%"
                f"{m['max_dd']*100:>+7.1f}%{m['calmar']:>8.2f}   {'':>8}{tag[0]:>8}"
                f"{tag[1]:>8}{tag[2]:>8}{mark}")

    best_sh = max(GRID, key=lambda w: res[w]["full"]["sharpe_excess"])
    best_ca = max(GRID, key=lambda w: res[w]["full"]["calmar"])
    print(f"\n  best full-period Sharpe: {best_sh} -> "
          f"{res[best_sh]['full']['sharpe_excess']:.2f}")
    print(f"  best full-period Calmar: {best_ca} -> "
          f"{res[best_ca]['full']['calmar']:.2f} "
          f"(MaxDD {res[best_ca]['full']['max_dd']*100:.1f}%)")

    out = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
           "learned": {"rows": rows, "mean_learned_pct": float(np.mean(learned) * 100),
                       "mean_fixed_pct": float(np.mean(fixed) * 100),
                       "wins": wins, "n": len(learned), "verdict": verdict_a},
           "fixed_grid": [{"weights": list(w), **{k: float(v) for k, v in res[w]["full"].items()},
                           "wf_sharpe_wins": sum(1 for a, b in zip(res[w]["wf"], bwf)
                                                 if a["sharpe_excess"] > b["sharpe_excess"]),
                           "wf_maxdd_wins": sum(1 for a, b in zip(res[w]["wf"], bwf)
                                                if a["max_dd"] > b["max_dd"]),
                           "wf_calmar_wins": sum(1 for a, b in zip(res[w]["wf"], bwf)
                                                 if a["calmar"] > b["calmar"]),
                           "n_windows": len(bwf)} for w in GRID]}
    p = os.path.join(_ROOT, "reports", "allocation_walkforward.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

"""
What does honesty cost? — pricing the fix for PBO 42.4%
=======================================================
`reports/nested_walkforward.json` reports IS->OOS rank correlation of -0.126 and
concludes: "Config selection adds nothing; the honest deployment is a fixed,
economically-motivated config (or the 1/N ensemble), NOT a learned choice."

The deployed book is `mom_heavy` — the config that looked best with full-sample
knowledge. This script prices the two honest alternatives against it so the
choice is made on numbers instead of on which one flatters the headline:

  deployed   mom_heavy      the current book (selected in-sample -> not clean OOS)
  neutral    blend          economically-motivated, no tilt toward the winner
  ensemble   1/N over all 5 weight schemes, capital split equally

Expect the honest options to come out BELOW the deployed number. That is the
point: the gap IS the selection bias, measured in basis points.

  MARK5_CACHE=data/pit_cache python3 scripts/deoverfit_cost.py
"""
import json
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            metrics_after_exit_tax, load_ohlcv, load_nifty, load_sector_map)

REPORTS = os.path.join(_ROOT, "reports")
START, END = "2016-01-01", "2026-07-21"
TD, TAX, NIFTY_TAX = 252, 0.15, 0.125
SLEEVES = {"eq": .5, "GOLDBEES": .25, "MON100": .25}

# the same family nested_walkforward.py searched over
WEIGHTS = {
    "blend":      {"momentum": .30, "low_vol": .30, "trend": .20, "stability": .20},
    "mom_heavy":  {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15},
    "lowvol_hvy": {"momentum": .15, "low_vol": .50, "trend": .15, "stability": .20},
    "trend_hvy":  {"momentum": .20, "low_vol": .20, "trend": .45, "stability": .15},
    "stab_hvy":   {"momentum": .20, "low_vol": .20, "trend": .15, "stability": .45},
}
DEPLOYED, NEUTRAL = "mom_heavy", "blend"


def wrap(eq_nav):
    """Deployed 50/25/25 sleeve blend, annual sleeve rebalance. Gross of exit tax."""
    cal = eq_nav.index
    ser = {"eq": eq_nav.pct_change(fill_method=None).fillna(0.0)}
    for k in SLEEVES:
        if k != "eq":
            s = load_ohlcv(k)["close"].astype(float).reindex(cal, method="ffill")
            ser[k] = s.pct_change().fillna(0.0)
    cur, nav, out = dict(SLEEVES), 1.0, {}
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur.values())
            for k in cur:
                cur[k] *= (1 + ser[k].iloc[i])
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % TD == 0:
            tot = sum(cur.values())
            cur = {k: tot * SLEEVES[k] for k in SLEEVES}
    return pd.Series(out)


def summarise(name, sleeve_nav):
    m = metrics_after_exit_tax(wrap(sleeve_nav), TAX)
    return {"name": name, "cagr": m["cagr"] * 100, "sharpe": m["sharpe_excess"],
            "vol": m["vol"] * 100, "max_dd": m["max_dd"] * 100, "calmar": m["calmar"],
            "eq_only_cagr": metrics_after_exit_tax(sleeve_nav, TAX)["cagr"] * 100}


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    smap = load_sector_map()
    print(f"  universe {len(panel.tickers)} names · {len(WEIGHTS)} configs · "
          f"{START}..{END}\n", flush=True)

    navs = {}
    for name, fw in WEIGHTS.items():
        cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                                 tilt_strength=1.5, max_weight=0.08, factor_weights=fw)
        run = Backtester(panel, PortfolioConstructor(cfg, sector_map=smap),
                         BacktestConfig(rebal_bars=126, top_n_liquid=300)).run(START, END)
        navs[name] = run["nav_gross"]
        c = metrics_after_exit_tax(navs[name], TAX)["cagr"] * 100
        print(f"    {name:<12} equity sleeve {c:+6.2f}% net", flush=True)

    # 1/N ensemble: equal capital in each config, i.e. the average of their curves
    rets = pd.DataFrame({k: v.pct_change().fillna(0.0) for k, v in navs.items()})
    navs["__ensemble__"] = (1 + rets.mean(axis=1)).cumprod()

    rows = [summarise("deployed (mom_heavy)", navs[DEPLOYED]),
            summarise("neutral (blend)", navs[NEUTRAL]),
            summarise("1/N ensemble", navs["__ensemble__"])]

    nifty = load_nifty(True).reindex(navs[DEPLOYED].index, method="ffill")
    nb = metrics_after_exit_tax(nifty / nifty.iloc[0], NIFTY_TAX)["cagr"] * 100

    print("\n" + "=" * 88)
    print(f"  FULL SYSTEM (50/25/25), net of tax and costs   ·   Nifty 50 TRI {nb:+.2f}%")
    print("=" * 88)
    print(f"  {'book':<24}{'CAGR':>9}{'Sharpe':>9}{'Vol':>8}{'MaxDD':>9}"
          f"{'Calmar':>9}{'vs Nifty':>10}")
    print("  " + "-" * 76)
    for r in rows:
        print(f"  {r['name']:<24}{r['cagr']:>+8.2f}%{r['sharpe']:>9.2f}{r['vol']:>7.1f}%"
              f"{r['max_dd']:>+8.1f}%{r['calmar']:>9.2f}{r['cagr']-nb:>+9.2f}pp")
    print("  " + "-" * 76)
    dep = rows[0]["cagr"]
    for r in rows[1:]:
        print(f"  cost of {r['name']:<24} {r['cagr']-dep:+.2f}pp CAGR   "
              f"{r['sharpe']-rows[0]['sharpe']:+.2f} Sharpe")
    print("\n  The gap is the selection bias the deployed config carries. Publishing the\n"
          "  deployed number as out-of-sample is the thing PBO 42.4% says you cannot do.")

    out = {"generated": str(pd.Timestamp.today().date()), "window": [START, END],
           "nifty_cagr_pct": nb, "books": rows,
           "verdict": ("Deployed config was chosen with full-sample knowledge. Quote the "
                       "neutral or ensemble figure as the honest expectation; the deployed "
                       "figure is an upper bound, not a forecast.")}
    p = os.path.join(REPORTS, "deoverfit_cost.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()

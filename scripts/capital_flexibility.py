"""
Capital flexibility: what does whole-share granularity actually cost?
====================================================================
The backtest holds infinitely divisible positions. A real book at Rs 5 lakh
cannot: a Rs 6,442 share in a Rs 12,500 slot buys 1 share and lands 48% away
from its target. The engine has never charged itself for this, so every headline
in this repo is quoted at a capital level where the constraint does not bite.

This measures the drag exactly, at each real rebalance, against real forward
prices — no simulation of a simulation:

    for each rebalance date d_i:
        target book return over [d_i, d_{i+1}]  =  sum_t w_t * r_t
        achievable book return                  =  sum_t w'_t * r_t
        drag                                    =  the difference
        (uninvested residual cash earns 0, which is what actually happens)

Two allocators are compared:
  FLOOR  qty = floor(budget_t / price_t). What scripts/paper_track.py does today.
         It rounds every position DOWN, so it systematically under-invests.
  LR     largest-remainder: floor first, then spend the residual cash one share
         at a time on whichever name it helps most, while cash lasts. Standard
         apportionment; costs nothing and uses the cash that FLOOR strands.

The question this answers: can a Rs 5 lakh book behave like a large one?

  MARK5_CACHE=data/pit_cache python3 scripts/capital_flexibility.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig)

START, END = "2016-01-01", "2026-07-21"
EQUITY_FRAC = 0.50
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
CAPITALS = [1e5, 2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7, 5e7]
N_HOLDS = [10, 15, 20, 25]


def alloc_floor(w: pd.Series, px: pd.Series, budget: float):
    """qty = floor(slot / price). Always rounds down; strands cash."""
    q = np.floor(budget * w / px)
    return q


def alloc_lr(w: pd.Series, px: pd.Series, budget: float):
    """Largest-remainder: floor, then spend residual cash where it helps most.

    'Helps most' = the largest remaining weight shortfall that the cash can
    still afford. This is apportionment, not optimisation — it terminates, it
    never overspends, and it cannot do worse than FLOOR.
    """
    q = np.floor(budget * w / px).values.astype(float)
    p = px.values.astype(float)
    tgt = (budget * w).values.astype(float)
    cash = budget - float((q * p).sum())
    for _ in range(4000):
        short = tgt - q * p                      # rupees still owed each name
        afford = (p <= cash + 1e-9) & (short > 0)
        if not afford.any():
            break
        i = int(np.argmax(np.where(afford, short, -np.inf)))
        q[i] += 1
        cash -= p[i]
    return pd.Series(q, index=w.index)


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    results = {}

    for n_hold in N_HOLDS:
        cfg = ConstructionConfig(mode="factor_tilt", n_hold=n_hold,
                                 base_weighting="inverse_vol", tilt_strength=1.5,
                                 max_weight=0.08, factor_weights=MOM)
        run = Backtester(panel, PortfolioConstructor(cfg),
                         BacktestConfig(rebal_bars=126, top_n_liquid=300)).run(START, END)
        dates = list(run["weights"])
        close = panel.close

        for cap in CAPITALS:
            budget0 = cap * EQUITY_FRAC
            drag = {"FLOOR": [], "LR": []}
            unbuyable, cashdrag = [], []
            nav = {"FLOOR": 1.0, "LR": 1.0, "IDEAL": 1.0}
            for i, d in enumerate(dates):
                w = run["weights"][d]
                w = w[w > 0]
                w = w / w.sum()
                nxt = dates[i + 1] if i + 1 < len(dates) else close.index[-1]
                px = close.loc[:d, w.index].ffill().iloc[-1]
                fwd = close.loc[:nxt, w.index].ffill().iloc[-1] / px - 1.0
                ok = px.notna() & fwd.notna()
                if ok.sum() < 3:
                    continue
                w, px, fwd = w[ok], px[ok], fwd[ok]
                w = w / w.sum()
                # capital available to the equity sleeve, compounded so far
                budget = budget0 * nav["IDEAL"]
                r_ideal = float((w * fwd).sum())
                for name, fn in (("FLOOR", alloc_floor), ("LR", alloc_lr)):
                    q = fn(w, px, budget)
                    val = q * px
                    wa = val / budget                     # residual cash earns 0
                    drag[name].append(float((wa * fwd).sum()) - r_ideal)
                    nav[name] *= 1 + float((wa * fwd).sum())
                nav["IDEAL"] *= 1 + r_ideal
                unbuyable.append(float((np.floor(budget * w / px) < 1).mean()))
                cashdrag.append(1 - float((np.floor(budget * w / px) * px).sum() / budget))

            yrs = (dates[-1] - dates[0]).days / 365.25
            results[(n_hold, cap)] = {
                "n_hold": n_hold, "capital": cap,
                "cagr_ideal": nav["IDEAL"] ** (1 / yrs) - 1,
                "cagr_floor": nav["FLOOR"] ** (1 / yrs) - 1,
                "cagr_lr": nav["LR"] ** (1 / yrs) - 1,
                "unbuyable_pct": float(np.mean(unbuyable)) * 100,
                "idle_cash_pct": float(np.mean(cashdrag)) * 100,
            }
        print(f"  ran n_hold={n_hold}")

    print("\n" + "=" * 106)
    print("  COST OF WHOLE SHARES — net CAGR of the equity sleeve, by capital")
    print("  IDEAL = infinitely divisible (what every headline in this repo assumes)")
    print("=" * 106)
    print(f"  {'capital':>10}  {'n_hold':>6}{'IDEAL':>9}{'FLOOR':>9}{'LR':>9}"
          f"{'FLOOR drag':>12}{'LR drag':>10}{'unbuyable':>11}{'idle cash':>11}")
    print("  " + "-" * 102)
    for n_hold in N_HOLDS:
        for cap in CAPITALS:
            r = results[(n_hold, cap)]
            print(f"  Rs {cap/1e5:>6,.1f}L  {n_hold:>6}{r['cagr_ideal']*100:>8.2f}%"
                  f"{r['cagr_floor']*100:>8.2f}%{r['cagr_lr']*100:>8.2f}%"
                  f"{(r['cagr_floor']-r['cagr_ideal'])*100:>+11.2f}pp"
                  f"{(r['cagr_lr']-r['cagr_ideal'])*100:>+9.2f}pp"
                  f"{r['unbuyable_pct']:>10.1f}%{r['idle_cash_pct']:>10.1f}%")
        print("  " + "-" * 102)

    print("\n  BEST n_hold AT EACH CAPITAL (LR allocator, the deployable one)")
    print("  " + "-" * 60)
    for cap in CAPITALS:
        best = max(N_HOLDS, key=lambda n: results[(n, cap)]["cagr_lr"])
        r = results[(best, cap)]
        r20 = results[(20, cap)]
        print(f"  Rs {cap/1e5:>6,.1f}L   best n_hold = {best:<3} "
              f"({r['cagr_lr']*100:+.2f}%)   deployed n_hold=20 gives "
              f"{r20['cagr_lr']*100:+.2f}%  -> {(r20['cagr_lr']-r['cagr_lr'])*100:+.2f}pp")

    p = os.path.join(_ROOT, "reports", "capital_flexibility.json")
    json.dump([v for v in results.values()], open(p, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

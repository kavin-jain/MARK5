"""
Capacity: how much money can this strategy actually run?
=======================================================
The published headline is quoted on Rs 5cr. That number has never been checked
against the liquidity of the names the system actually buys — and the book
reaches to the 300th-most-traded symbol on NSE, which is genuinely small.

This is the question an allocator asks first and the one a backtest never
answers, because a backtest fills every order at the close no matter the size.

Method (standard sell-side practice):
  - ADV        = 20-day median rupee turnover of each held name.
  - participation = position size / ADV. The convention is that you can trade
    about 10% of a day's volume without moving the price much; above that you
    either pay impact or take days to build the position.
  - impact is modelled with the square-root law, the standard practitioner
    form:  impact_bps ~ Y * sigma * sqrt(Q/ADV), Y ~ 1 (Almgren et al.).
    This is a model, not a measurement — the point is the ORDER of magnitude
    and where it crosses from negligible to material.

Reported: at each capital level, the worst-case single-name participation, the
weighted-average impact cost per rebalance, and the resulting drag on annual
return at the deployed 2 rebalances a year.

  MARK5_CACHE=data/pit_cache python3 scripts/capacity_analysis.py
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

END = "2026-07-21"
START = "2016-01-01"
EQUITY_FRAC = 0.50          # deployed 50/25/25 — only the equity half hits single names
PARTICIPATION_OK = 0.10     # 10% of ADV = the conventional comfort limit
CAPITALS = [5e5, 25e5, 1e7, 5e7, 1e8, 5e8, 1e9]
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=MOM)
    run = Backtester(panel, PortfolioConstructor(cfg),
                     BacktestConfig(rebal_bars=126, top_n_liquid=300)).run(START, END)

    # rupee turnover per name per day, and 252d vol, for impact
    turn = panel.close * panel.volume
    dret = panel.close.pct_change(fill_method=None)

    rows = []
    for d, w in run["weights"].items():
        w = w[w > 0]
        for t, wt in w.items():
            if t not in turn.columns:
                continue
            adv = turn[t].loc[:d].tail(20).median()
            vol = dret[t].loc[:d].tail(252).std()
            if not np.isfinite(adv) or adv <= 0 or not np.isfinite(vol):
                continue
            rows.append({"date": d, "ticker": t, "weight": float(wt),
                         "adv": float(adv), "dvol": float(vol)})
    df = pd.DataFrame(rows)

    print(f"\n  {len(run['weights'])} rebalances, {len(df)} position-observations")
    print(f"  Median ADV of a held name: Rs {df['adv'].median()/1e7:,.1f} cr/day")
    print(f"  10th-pctile (the thin ones): Rs {df['adv'].quantile(.10)/1e7:,.2f} cr/day")
    print(f"  Deployed book puts {EQUITY_FRAC*100:.0f}% of capital into these names.\n")

    print("=" * 100)
    print("  CAPACITY — participation and modelled market impact at the deployed config")
    print("=" * 100)
    print(f"  {'capital':>12}{'worst name':>13}{'median name':>14}{'names >10% ADV':>17}"
          f"{'impact/rebal':>15}{'drag /yr':>11}")
    print("  " + "-" * 96)

    out = []
    for cap in CAPITALS:
        # position value per name = capital * equity_frac * weight
        pos = cap * EQUITY_FRAC * df["weight"]
        part = pos / df["adv"]
        # square-root impact law, one-way, in bps of the traded value
        impact = 1.0 * df["dvol"] * np.sqrt(part.clip(lower=0))
        # weight the impact by how much of the book each name is
        wavg_impact = float((impact * df["weight"]).sum() / df["weight"].sum())
        # 2 rebalances/yr, ~100% of the equity book turned over each time,
        # both sides -> approximate annual drag on TOTAL capital
        drag = wavg_impact * 2 * EQUITY_FRAC
        over = float((part > PARTICIPATION_OK).mean())
        r = {"capital": cap, "worst_participation": float(part.max()),
             "median_participation": float(part.median()),
             "frac_over_10pct": over, "impact_per_rebal_bps": wavg_impact * 1e4,
             "annual_drag_pct": drag * 100}
        out.append(r)
        print(f"  Rs {cap/1e7:>7,.1f}cr{part.max()*100:>12.1f}%{part.median()*100:>13.2f}%"
              f"{over*100:>16.0f}%{wavg_impact*1e4:>13.0f}bp{drag*100:>10.2f}%")

    print("  " + "-" * 96)
    ok = [r for r in out if r["worst_participation"] <= PARTICIPATION_OK]
    lim = max((r["capital"] for r in ok), default=0)
    soft = [r for r in out if r["frac_over_10pct"] <= 0.05]
    softlim = max((r["capital"] for r in soft), default=0)
    print(f"\n  READING")
    print(f"  - Every position stays under {PARTICIPATION_OK*100:.0f}% of a day's volume up to "
          f"Rs {lim/1e7:,.1f}cr.")
    print(f"  - Up to Rs {softlim/1e7:,.1f}cr, at most 5% of positions breach that limit "
          f"(tradeable over 2-3 days).")
    print(f"  - The published headline is quoted on Rs 5cr: worst-case participation "
          f"{[r for r in out if r['capital']==5e7][0]['worst_participation']*100:.1f}%, "
          f"modelled drag "
          f"{[r for r in out if r['capital']==5e7][0]['annual_drag_pct']:.2f}%/yr.")
    print(f"  - Impact is NOT in the backtest, which fills everything at the close. Subtract "
          f"the drag column\n    from any headline quoted at that capital.")

    p = os.path.join(_ROOT, "reports", "capacity_analysis.json")
    json.dump({"generated": pd.Timestamp.now().isoformat(), "levels": out,
               "median_adv_inr": float(df["adv"].median()),
               "p10_adv_inr": float(df["adv"].quantile(.10)),
               "hard_limit_inr": lim, "soft_limit_inr": softlim},
              open(p, "w"), indent=1)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

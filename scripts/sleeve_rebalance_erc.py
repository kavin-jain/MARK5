"""
Does the risk-parity book need faster sleeve rebalancing to capture its Sharpe?
==============================================================================
path_to_sharpe_11.py found a 0.15 Sharpe gap between theory and measurement at
risk-parity weights:

    continuously-rebalanced fixed weights   Sharpe 1.155
    annually-rebalanced, net of tax         Sharpe 1.00

That gap is not a modelling error, it is a real cost: between rebalances the
weights drift with performance, so the book stops being risk-balanced exactly
when one sleeve has run. K20 already tested sleeve-rebalance frequency and found
noise — but at 50/25/25, where the drift is small. At ~27/49/24 the gold sleeve
is nearly half the book, so drift is much larger and the same test may not hold.

Rebalancing is NOT free and this test charges for it properly, which the
dashboard's wrap() does not:
  - ETF round-trip cost on the traded notional
  - STCG at 20% on gains realised inside a year, LTCG 12.5% beyond it
A faster cadence must beat a slower one AFTER paying both, or it is not real.

Bar: the usual >=6/8 rolling 3-year windows, per metric.

  MARK5_CACHE=data/pit_cache python3 scripts/sleeve_rebalance_erc.py
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
                            load_ohlcv, metrics, metrics_after_exit_tax, load_sector_map)

START, END = "2016-01-01", "2026-07-21"
TD, RF = 252, 0.065
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}
KEYS = ["eq", "GOLDBEES", "MON100"]
# Real ETF delivery friction, round trip: brokerage 0 on delivery, STT 0.1% sell,
# plus exchange/SEBI/GST/stamp and a conservative spread allowance.
SLEEVE_COST = 0.0015
STCG, LTCG, TERMTAX = 0.20, 0.125, 0.15
DEPLOYED = np.array([.50, .25, .25])
ERC = np.array([.29, .45, .26])
CADENCES = [21, 42, 63, 126, 252, 504]


def blend_costed(R, w, rebal):
    """Fixed-weight blend with REAL rebalancing friction and tax.

    Each rebalance trades |w_drifted - w_target| of the book. That notional pays
    SLEEVE_COST, and the sold portion realises a gain taxed at STCG or LTCG
    depending on the cadence (a cadence under a year can only ever realise STCG).
    Cost and tax are deducted from NAV at the moment they occur.
    """
    cur = {k: w[i] for i, k in enumerate(KEYS)}
    basis = {k: w[i] for i, k in enumerate(KEYS)}      # cost basis per sleeve
    nav, out, traded, tax_paid = 1.0, {}, 0.0, 0.0
    rate = LTCG if rebal > TD else STCG
    for i, d in enumerate(R.index):
        if i > 0:
            prev = sum(cur.values())
            for j, k in enumerate(KEYS):
                cur[k] *= (1 + R[k].iloc[i])
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % rebal == 0:
            tot = sum(cur.values())
            tgt = {k: tot * w[j] for j, k in enumerate(KEYS)}
            turn = sum(abs(tgt[k] - cur[k]) for k in KEYS) / 2.0
            cost = turn * SLEEVE_COST
            # tax only on the sleeves being TRIMMED, on their embedded gain
            tax = 0.0
            for k in KEYS:
                if cur[k] > tgt[k]:
                    sold = cur[k] - tgt[k]
                    frac = sold / cur[k] if cur[k] > 0 else 0.0
                    gain = (cur[k] - basis[k]) * frac
                    tax += max(0.0, gain) * rate
            drag = (cost + tax) / tot if tot > 0 else 0.0
            nav *= (1 - drag)
            traded += turn / tot if tot > 0 else 0.0
            tax_paid += tax / tot if tot > 0 else 0.0
            cur = {k: tgt[k] * (1 - drag) for k in KEYS}
            basis = dict(cur)
    # GROSS of exit tax — priced by metrics_after_exit_tax at the measurement
    # site, so it never enters the return series.
    return pd.Series(out), traded, tax_paid


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

    windows = []
    for y0 in range(2016, 2024):
        e = f"{y0+2}-12-31"
        if pd.Timestamp(e) > pd.Timestamp(END):
            e = END
        windows.append((f"{y0}-01-01", e))

    res = {}
    for name, w in (("deployed 50/25/25", DEPLOYED), ("risk-parity 29/45/26", ERC)):
        for cad in CADENCES:
            nav, turn, tax = blend_costed(R, w, cad)
            m = metrics_after_exit_tax(nav, TERMTAX)
            wf = []
            for a, b in windows:
                seg = R.loc[a:b]
                if len(seg) < max(200, cad + 20):
                    continue
                wf.append(metrics_after_exit_tax(blend_costed(seg, w, cad)[0], TERMTAX))
            res[(name, cad)] = {"m": m, "wf": wf, "turnover": turn, "tax": tax}

    base = res[("deployed 50/25/25", TD)]
    print("\n" + "=" * 108)
    print("  SLEEVE REBALANCE CADENCE — with real ETF cost and realised tax charged")
    print("  baseline = deployed 50/25/25 rebalanced annually (the shipped system)")
    print("=" * 108)
    print(f"  {'allocation':<22}{'cadence':>9}{'CAGR':>9}{'shExc':>8}{'vol':>7}{'MaxDD':>8}"
          f"{'Calmar':>8}{'turn/yr':>9}{'tax':>7}   {'Sharpe':>7}{'MaxDD':>7}{'Calmar':>8}")
    print("  " + "-" * 104)
    rows = []
    for name, w in (("deployed 50/25/25", DEPLOYED), ("risk-parity 29/45/26", ERC)):
        for cad in CADENCES:
            r = res[(name, cad)]
            m = r["m"]
            cells = []
            for k in ("sharpe_excess", "max_dd", "calmar"):
                n = sum(1 for a, b in zip(r["wf"], base["wf"]) if a[k] > b[k])
                cells.append(f"{n}/{len(base['wf'])}")
            yrs = m["years"]
            mark = "  <- shipped" if (name.startswith("deployed") and cad == TD) else ""
            print(f"  {name:<22}{cad:>9}{m['cagr']*100:>+8.2f}%{m['sharpe_excess']:>8.2f}"
                  f"{m['vol']*100:>6.1f}%{m['max_dd']*100:>+7.1f}%{m['calmar']:>8.2f}"
                  f"{r['turnover']/yrs*100:>8.0f}%{r['tax']*100:>6.1f}%   "
                  f"{cells[0]:>7}{cells[1]:>7}{cells[2]:>8}{mark}")
            rows.append({"allocation": name, "cadence": cad,
                         **{k: float(v) for k, v in m.items()},
                         "turnover_yr_pct": r["turnover"] / yrs * 100,
                         "tax_pct": r["tax"] * 100,
                         "wf_sharpe": cells[0], "wf_maxdd": cells[1],
                         "wf_calmar": cells[2]})
        print("  " + "-" * 104)

    best = max(rows, key=lambda r: r["sharpe_excess"])
    bestc = max(rows, key=lambda r: r["calmar"])
    print(f"\n  best Sharpe : {best['allocation']} @ {best['cadence']}d -> "
          f"{best['sharpe_excess']:.2f}  (MaxDD {best['max_dd']*100:.1f}%, "
          f"Calmar {best['calmar']:.2f})")
    print(f"  best Calmar : {bestc['allocation']} @ {bestc['cadence']}d -> "
          f"{bestc['calmar']:.2f}  (Sharpe {bestc['sharpe_excess']:.2f}, "
          f"MaxDD {bestc['max_dd']*100:.1f}%)")
    p = os.path.join(_ROOT, "reports", "sleeve_rebalance_erc.json")
    json.dump({"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
               "sleeve_cost": SLEEVE_COST, "rows": rows}, open(p, "w"), indent=1, default=float)
    print(f"\n  saved -> {p}\n")


if __name__ == "__main__":
    main()

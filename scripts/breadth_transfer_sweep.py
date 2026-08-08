"""
P2.1 + P2.2 — does breadth raise the information ratio?  (run me)
================================================================
PRE-REGISTERED in docs/RESEARCH_PLAN_2026-08.md before this was run.

  HYPOTHESIS   IR rises monotonically with n_hold up to roughly 60-80 names, then
               flattens as the marginal name adds more correlation than
               information. Sector-neutral ranking raises IR at EVERY n_hold.
  FALSIFIED IF IR is flat or falling in n_hold across the range. That would mean
               breadth is already saturated and P5's concentrate-to-12 was right.

WHY. The full Fundamental Law (Clarke, de Silva & Thorley 2002) is

    IR = IC * sqrt(BR) * TC

Back-solving today's book at BR=40 (20 names x 2 rebalances) and TC~0.55 gives
IC ~= 0.105, which matches the knowledge base's independently-derived 0.05-0.10.
Raising n_hold lifts BR *and* TC together — more of the ranking reaches the
portfolio — which is why it is the single highest-value structural change.

The catch is that sqrt(BR) counts INDEPENDENT bets. Twenty momentum names riding
one hot sector is closer to one bet than twenty, so nominal breadth overstates
real breadth. Sector-neutral ranking is what converts one into the other; it is
tested here as a crossed factor, not as a separate idea.

SCORED ON IR, never on CAGR. CAGR carries a +/-9pp standard error on this sample
and cannot rank anything (Mandate §2).

  MARK5_CACHE=data/pit_cache_2007 python3 scripts/breadth_transfer_sweep.py
"""
import os, sys, json

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            load_sector_map, load_delivery_factors, metrics)

REPORTS = os.path.join(_ROOT, "reports")
START = os.environ.get("MARK5_START", "2007-01-01")
END = os.environ.get("MARK5_END", "2026-07-21")
TRADING_DAYS = 252
Z95 = 1.6448536269514722
N_HOLDS = [12, 20, 40, 60, 80, 100]


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def info_ratio(strat: pd.Series, bench: pd.Series) -> dict:
    """IR of a NAV series against a benchmark NAV series, plus its t-stat."""
    a = strat.pct_change(fill_method=None).dropna()
    b = bench.pct_change(fill_method=None).dropna()
    a, b = a.align(b, join="inner")
    act = (a - b).dropna()
    mu = float(act.mean() * TRADING_DAYS)
    te = float(act.std() * np.sqrt(TRADING_DAYS))
    yrs = len(act) / TRADING_DAYS
    ir = mu / te if te else 0.0
    t = ir * np.sqrt(yrs)
    return {"active_pp": mu * 100, "te_pp": te * 100, "ir": ir, "t_stat": t,
            "p_value": float(1 - _norm_cdf(t)), "years": yrs,
            "significant_95": bool(t > Z95)}


def effective_n(weights_hist: dict) -> float:
    """Inverse Herfindahl of the weights, averaged over rebalances.

    The count of holdings overstates diversification when weights are skewed.
    1/sum(w^2) is how many EQUALLY-weighted names the book actually behaves like,
    so effective_n / n_hold is a direct read on how much of the nominal breadth
    the weighting scheme is throwing away.
    """
    vals = []
    for w in weights_hist.values():
        w = np.asarray(list(w.values()) if isinstance(w, dict) else w, float)
        w = w[w > 0]
        if len(w):
            w = w / w.sum()
            vals.append(1.0 / np.sum(w ** 2))
    return float(np.mean(vals)) if vals else 0.0


def main():
    os.makedirs(REPORTS, exist_ok=True)
    cache = os.environ.get("MARK5_CACHE", "data/cache")
    print(f"Loading panel from {cache} ...", flush=True)
    panel = DataPanel(discover_tickers(), END)
    smap = load_sector_map()
    dfac = load_delivery_factors(universe=panel.tickers)
    base_weights = {"momentum": 0.45, "low_vol": 0.15, "trend": 0.25, "stability": 0.15}
    if dfac:
        base_weights = {**base_weights, "deliv_chg": 0.10}
    bt_cfg = BacktestConfig(rebal_bars=126,
                            top_n_liquid=int(os.environ.get("MARK5_TOP_N", "300")))
    print(f"  {len(panel.tickers)} symbols, sector map covers "
          f"{sum(1 for t in panel.tickers if t in smap)}\n", flush=True)

    print("Benchmark: equal-weight of the same point-in-time universe...", flush=True)
    bench = Backtester(panel, PortfolioConstructor(
        ConstructionConfig(mode="equal_weight", base_weighting="equal")),
        bt_cfg).run(START, END)["nav_gross"]

    rows = []
    for neutral in (False, True):
        for nh in N_HOLDS:
            cfg = ConstructionConfig(
                mode="factor_tilt", n_hold=nh, base_weighting="inverse_vol",
                tilt_strength=1.5, max_weight=max(0.08, 1.5 / nh),
                factor_weights=dict(base_weights), sector_neutral=neutral)
            r = Backtester(panel, PortfolioConstructor(cfg, sector_map=smap),
                           bt_cfg, extra_factors=dfac).run(START, END)
            m, ir = r["metrics"], info_ratio(r["nav_gross"], bench)
            eff = effective_n(r["weights"])
            rows.append({"n_hold": nh, "sector_neutral": neutral,
                         "ir": ir["ir"], "t_stat": ir["t_stat"], "p_value": ir["p_value"],
                         "active_pp": ir["active_pp"], "te_pp": ir["te_pp"],
                         "cagr": m["cagr"], "sharpe": m.get("sharpe_excess", m["sharpe"]),
                         "max_dd": m["max_dd"], "calmar": m["calmar"],
                         "turnover_yr": m.get("turnover_yr", 0),
                         "effective_n": eff, "transfer_ratio": eff / nh})
            print(f"  n={nh:<4} neutral={str(neutral):<5}  IR {ir['ir']:.3f}  "
                  f"t {ir['t_stat']:.2f}  CAGR {m['cagr']*100:+6.2f}%  "
                  f"MaxDD {m['max_dd']*100:+6.1f}%  effN {eff:.1f}/{nh}", flush=True)

    _report(rows)
    json.dump({"window": [START, END], "cache": cache, "rows": rows},
              open(os.path.join(REPORTS, "breadth_transfer_sweep.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/breadth_transfer_sweep.json")
    return rows


def _report(rows):
    print("\n" + "=" * 88)
    print("  RESULT — scored on IR vs equal-weight of the same universe")
    print("=" * 88)
    print(f"  {'n_hold':>7}{'raw IR':>10}{'neutral IR':>12}{'delta':>9}"
          f"{'raw effN':>10}{'neut effN':>11}")
    print("  " + "-" * 78)
    raw = {r["n_hold"]: r for r in rows if not r["sector_neutral"]}
    neu = {r["n_hold"]: r for r in rows if r["sector_neutral"]}
    for nh in N_HOLDS:
        a, b = raw.get(nh), neu.get(nh)
        if not a or not b:
            continue
        print(f"  {nh:>7}{a['ir']:>10.3f}{b['ir']:>12.3f}{b['ir']-a['ir']:>+9.3f}"
              f"{a['effective_n']:>10.1f}{b['effective_n']:>11.1f}")

    best = max(rows, key=lambda r: r["ir"])
    base = raw.get(20)
    print("\n  " + "-" * 78)
    print(f"  BEST: n_hold={best['n_hold']} neutral={best['sector_neutral']} "
          f"-> IR {best['ir']:.3f} (t={best['t_stat']:.2f}, p={best['p_value']:.4f})")
    if base:
        print(f"  DEPLOYED TODAY (n=20, raw): IR {base['ir']:.3f}")
        print(f"  improvement: {best['ir']-base['ir']:+.3f} IR")

    # --- adjudicate the pre-registered hypotheses, in both directions ---
    ir_by_n = [raw[n]["ir"] for n in N_HOLDS if n in raw]
    rising = ir_by_n.index(max(ir_by_n)) >= 2      # peak at n>=40
    neutral_helps = sum(1 for n in N_HOLDS if n in raw and n in neu
                        and neu[n]["ir"] > raw[n]["ir"])
    print("\n  PRE-REGISTERED VERDICTS")
    print(f"    P2.1 breadth raises IR, peak at n>=40 : "
          f"{'SUPPORTED' if rising else 'FALSIFIED — breadth is saturated'}")
    print(f"    P2.2 sector-neutral raises IR at every n : "
          f"{neutral_helps}/{len(N_HOLDS)} "
          f"{'SUPPORTED' if neutral_helps >= 5 else 'FALSIFIED' if neutral_helps <= 2 else 'PARTIAL'}")


if __name__ == "__main__":
    main()

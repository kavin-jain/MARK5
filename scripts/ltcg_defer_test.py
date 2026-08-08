"""
P4.1 — does deferring winners past the LTCG boundary pay?  (run me)
===================================================================
PRE-REGISTERED in docs/RESEARCH_PLAN_2026-08.md.

  HYPOTHESIS   Deferring the sale of profitable positions past 365 days (unless
               badly deranked) raises net CAGR by ~0.3-0.7pp/yr at no risk cost.
  FALSIFIED IF net return does not improve, because holding deranked names longer
               costs more than the tax it saves.

MOTIVATION, from the deployed book's own trade ledger: 306 winning sells sit in
the 6-10 month bucket carrying Rs 20,03,544 of gains taxed at 20% instead of
12.5%, and the 10-12 month bucket is COMPLETELY EMPTY. The 126-bar cadence lands
the first exit at ~182 days — squarely in the worst tax zone. Direct saving if
those were deferred: Rs 1,51,800.

NOT a re-run of K3/K18/K20. Those changed rebalance FREQUENCY or harvested losses.
This changes only the exit condition for profitable lots; cadence is untouched,
buys are untouched, and nothing moves to cash. It cannot defer a loss.

Reported NET of tax, which is the only way this can be judged, and across the
sub-periods so a single regime cannot carry the result.

  MARK5_CACHE=data/pit_cache_2007 python3 scripts/ltcg_defer_test.py
"""
import os, sys, json

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            load_sector_map, load_delivery_factors)

REPORTS = os.path.join(_ROOT, "reports")
START = os.environ.get("MARK5_START", "2007-01-01")
END = os.environ.get("MARK5_END", "2026-07-21")
MULTS = [1.0, 1.5, 2.0, 3.0]
WINDOWS = [(START, END, "FULL"), ("2007-01-01", "2012-12-31", "crisis 07-12"),
           ("2013-01-01", "2021-12-31", "bull 13-21"), ("2022-01-01", END, "recent 22-26")]


def main():
    print(f"Loading panel from {os.environ.get('MARK5_CACHE','data/cache')} ...", flush=True)
    panel = DataPanel(discover_tickers(), END)
    smap, dfac = load_sector_map(), load_delivery_factors(universe=panel.tickers)
    fw = {"momentum": 0.45, "low_vol": 0.15, "trend": 0.25, "stability": 0.15}
    if dfac:
        fw["deliv_chg"] = 0.10
    bt_cfg = BacktestConfig(rebal_bars=126,
                            top_n_liquid=int(os.environ.get("MARK5_TOP_N", "300")))

    runs, rows = {}, []
    for m in MULTS:
        cfg = ConstructionConfig(mode="factor_tilt", n_hold=20,
                                 base_weighting="inverse_vol", tilt_strength=1.5,
                                 max_weight=0.08, factor_weights=dict(fw),
                                 ltcg_defer_mult=m)
        r = Backtester(panel, PortfolioConstructor(cfg, sector_map=smap),
                       bt_cfg, extra_factors=dfac).run(START, END)
        runs[m] = r
        met = r["metrics"]
        rows.append({"mult": m, "cagr_net": met["cagr"], "cagr_gross": met.get("cagr_gross"),
                     "sharpe": met.get("sharpe_excess", met["sharpe"]),
                     "max_dd": met["max_dd"], "turnover_yr": met.get("turnover_yr", 0),
                     "tax_paid": met.get("tax_paid", 0)})
        print(f"  mult={m:<4} net CAGR {met['cagr']*100:+6.2f}%  "
              f"Sharpe {met.get('sharpe_excess', met['sharpe']):.2f}  "
              f"MaxDD {met['max_dd']*100:+6.1f}%  turnover {met.get('turnover_yr',0)*100:.0f}%  "
              f"tax {met.get('tax_paid',0):.3f}", flush=True)

    print("\n" + "=" * 76)
    print("  HOLDING-PERIOD SHIFT — did the winners actually move past 365 days?")
    print("=" * 76)
    for m in MULTS:
        tr = [t for t in runs[m]["trades"] if t.get("side") == "SELL"]
        if not tr:
            continue
        held = pd.Series([t.get("held_days", 0) for t in tr])
        gains = [t for t in tr if t.get("gain", 0) > 0]
        g = pd.Series([t.get("held_days", 0) for t in gains]) if gains else pd.Series(dtype=float)
        st = int((g < 365).sum()) if len(g) else 0
        lt = int((g >= 365).sum()) if len(g) else 0
        print(f"  mult={m:<4} sells {len(tr):>4}  median held {held.median():>5.0f}d  "
              f"winning sells: STCG {st:>3} / LTCG {lt:>3}  "
              f"({lt/(st+lt)*100 if st+lt else 0:.0f}% long-term)")

    print("\n" + "=" * 76)
    print("  SUB-PERIODS (net CAGR) — one regime must not carry the result")
    print("=" * 76)
    print(f"  {'window':<16}" + "".join(f"{'mult ' + str(m):>13}" for m in MULTS))
    sub = {}
    for a, b, lab in WINDOWS:
        cells = []
        for m in MULTS:
            nav = runs[m]["nav_net"].loc[a:b]
            if len(nav) < 60:
                cells.append(float("nan"))
                continue
            yrs = (nav.index[-1] - nav.index[0]).days / 365.25
            cells.append((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1)
        sub[lab] = cells
        print(f"  {lab:<16}" + "".join(f"{c*100:>+12.2f}%" for c in cells))

    base = rows[0]["cagr_net"]
    best = max(rows[1:], key=lambda r: r["cagr_net"])
    delta = (best["cagr_net"] - base) * 100
    wins = sum(1 for lab, c in sub.items()
               if c[MULTS.index(best["mult"])] == c[MULTS.index(best["mult"])]
               and c[MULTS.index(best["mult"])] > c[0])
    verdict = ("SUPPORTED" if delta > 0.10 and wins >= 3
               else "FALSIFIED — deferral does not pay" if delta <= 0
               else "MARGINAL — inside noise")
    print(f"\n  best mult {best['mult']} -> {delta:+.2f}pp net CAGR vs off, "
          f"winning {wins}/{len(WINDOWS)} windows")
    print(f"  PRE-REGISTERED VERDICT: {verdict}")

    json.dump({"window": [START, END], "rows": rows,
               "sub_periods": {k: list(v) for k, v in sub.items()},
               "verdict": verdict, "delta_pp": delta},
              open(os.path.join(REPORTS, "ltcg_defer_test.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/ltcg_defer_test.json")


if __name__ == "__main__":
    main()

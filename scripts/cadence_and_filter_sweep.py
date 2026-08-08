"""
Two questions the owner asked directly, answered with a table rather than a claim.
=================================================================================

  Q1  "Have you tested rebalancing every six months against three months, or
       any other period? I want to see all of them."

  Q2  "Is filtering to the 300 most liquid names and holding the top 20 the
       right way, or is there a better one?"

Both have been tested before (RESEARCH_LOG K3, K20, K21, K22 for cadence; P2.1
for breadth). The cadence sweep's raw table was stripped from the repo when it
was cut back to a rules-only engine, and the universe-size cut has never been
swept at all — it was inherited, not chosen. This regenerates the first and runs
the second for the first time, on the DEPLOYED config, over the LONG window.

WHY THE HEADLINE NUMBER IS NOT THE ANSWER. A full-period CAGR is one number on
one path. K21 is the cautionary tale: 21-day rebalancing posted a *higher*
full-period return (+22.6%) than the deployed 126-day, then lost 5 of 8
walk-forward windows. The mean was better and the strategy was worse. So every
config here is scored on ROLLING 3-YEAR WINDOWS, and ranked on how many of them
it wins — consistency, not the best single number.

PRE-REGISTERED, before the run:

  Q1 HYPOTHESIS   126 bars (~6 months) is not beaten on walk-forward consistency.
     FALSIFIED IF some other cadence beats 126d in >=6 of 8 rolling windows AND
                  does not have a worse worst-window.
  Q2 HYPOTHESIS   The top-300 liquidity cut is not the best available.
     FALSIFIED IF no universe size beats 300 on walk-forward consistency.

Note on direction: Q2's hypothesis is deliberately written so that CONFIRMING it
means changing the system. An inherited default should have to earn its place.

  MARK5_CACHE=data/pit_cache_2007 python3 scripts/cadence_and_filter_sweep.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_sector_map,
                            load_delivery_factors)
from core.portfolio.backtest import BacktestConfig

REPORTS = os.path.join(_ROOT, "reports")
START = os.environ.get("MARK5_START", "2007-01-01")
END = os.environ.get("MARK5_END", "2026-07-21")

# The deployed equity config, unchanged. Only the swept knob moves.
FW = {"momentum": 0.45, "low_vol": 0.15, "trend": 0.25, "stability": 0.15}
DEPLOYED_BARS, DEPLOYED_TOPN = 126, 300

# ~1 month to ~2 years. 126 bars ~ 6 calendar months on a 252-day year.
CADENCES = [21, 42, 63, 126, 189, 252, 378, 504]
UNIVERSES = [100, 200, 300, 500, 800]
WF_YEARS = 3


def build(panel, dfac, bars, top_n, min_turnover=0.0):
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08,
                             factor_weights={**FW, "deliv_chg": 0.10} if dfac else dict(FW))
    return Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                      BacktestConfig(rebal_bars=bars, top_n_liquid=top_n,
                                     min_turnover=min_turnover),
                      extra_factors=dfac)


def walk_forward(bt, cal):
    """Net CAGR in each rolling 3-year window, stepped by one year."""
    out = []
    years = sorted({d.year for d in cal})
    for y in years[:-WF_YEARS]:
        a, b = f"{y}-01-01", f"{y + WF_YEARS}-01-01"
        try:
            r = bt.run(a, b)
        except Exception:                                    # noqa: BLE001
            continue
        m = r.get("metrics") or {}
        if m.get("cagr") is not None:
            out.append((f"{y}-{y + WF_YEARS}", float(m["cagr"])))
    return out


def sweep(panel, dfac, cal, knob, values, fixed, label):
    rows = {}
    for v in values:
        bars, top_n = (v, fixed) if knob == "bars" else (fixed, v)
        print(f"  {label} = {v} ...", flush=True)
        bt = build(panel, dfac, bars, top_n)
        full = bt.run(START, END).get("metrics") or {}
        rows[v] = {"full": {k: full.get(k) for k in
                            ("cagr", "sharpe_excess", "max_dd", "calmar",
                             "turnover_yr", "tax_paid")},
                   "wf": walk_forward(bt, cal)}
    return rows


def report(rows, baseline, label, unit=""):
    base_wf = dict(rows[baseline]["wf"])
    print("\n" + "=" * 96)
    print(f"  {label}  —  full period {START} to {END}, then rolling {WF_YEARS}-year windows")
    print("=" * 96)
    print(f"  {label:>10}{'net CAGR':>10}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
          f"{'turnover':>10}{'| wf mean':>11}{'worst':>9}{'beats 126d':>12}")
    print("  " + "-" * 92)
    best = None
    for v, r in rows.items():
        f, wf = r["full"], r["wf"]
        vals = [c for _, c in wf]
        if not vals:
            continue
        wins = sum(1 for w, c in wf if c > base_wf.get(w, -9))
        mark = "  <- deployed" if v == baseline else ""
        print(f"  {str(v) + unit:>10}{f['cagr']*100:>+9.2f}%{f['sharpe_excess']:>8.2f}"
              f"{f['max_dd']*100:>+8.1f}%{f['calmar']:>8.2f}{f['turnover_yr']*100:>9.0f}%"
              f"{np.mean(vals)*100:>+10.2f}%{min(vals)*100:>+8.1f}%"
              f"{wins:>7}/{len(wf)}{mark}")
        if v != baseline and (best is None or wins > best[1]):
            best = (v, wins, len(wf), min(vals))
    return best, len(base_wf)


def sweep_turnover(panel, dfac, cal, floors):
    """SWEEP 3 — an ABSOLUTE rupee liquidity floor instead of a rank cut.

    Sweep 2's rank cut is not a fixed standard. Measured on this panel, the 500th
    most liquid NSE name traded Rs 0.01cr/day in 2013 and Rs 11.50cr/day in 2026 —
    the market deepened roughly 100x, so "top 500" silently means something
    completely different at each end of the window. Widening the rank cut
    therefore BUYS UNTRADEABLE MICROCAPS in the early years and books them at
    closing prices with flat slippage, which is a measurement error that flatters
    the result — Mandate §0, the exact failure this project keeps finding.

    A rupee floor is time-consistent: "everything trading more than Rs Xcr/day"
    means the same thing in every year, and the universe grows on its own as the
    market grows. That is the honest version of the same question.
    """
    rows = {}
    for f in floors:
        print(f"  min_turnover = Rs {f/1e7:.0f}cr/day ...", flush=True)
        bt = build(panel, dfac, DEPLOYED_BARS, 0, min_turnover=f)
        full = bt.run(START, END).get("metrics") or {}
        rows[int(f / 1e7)] = {"full": {k: full.get(k) for k in
                                       ("cagr", "sharpe_excess", "max_dd", "calmar",
                                        "turnover_yr", "tax_paid")},
                              "wf": walk_forward(bt, cal)}
    return rows


def main():
    print(f"Loading panel ({START} -> {END}) ...", flush=True)
    panel = DataPanel(discover_tickers(), END)
    dfac = load_delivery_factors(universe=panel.tickers)
    cal = panel.trading_calendar(START, END)
    print(f"  {len(panel.tickers)} names, {len(cal)} trading days, "
          f"delivery factors for {len(dfac) if dfac else 0}\n", flush=True)

    res = {"window": [START, END], "deployed": {"rebal_bars": DEPLOYED_BARS,
                                                "top_n_liquid": DEPLOYED_TOPN}}

    only = os.environ.get("MARK5_ONLY", "")

    if only in ("", "3"):
        # The baseline for this sweep is the DEPLOYED top-300 rank cut, re-run so
        # both arms are measured on identical machinery.
        print("SWEEP 3 — an absolute rupee liquidity floor, not a rank cut", flush=True)
        base = build(panel, dfac, DEPLOYED_BARS, DEPLOYED_TOPN)
        tr = {0: {"full": {k: (base.run(START, END).get("metrics") or {}).get(k) for k in
                           ("cagr", "sharpe_excess", "max_dd", "calmar",
                            "turnover_yr", "tax_paid")},
                  "wf": walk_forward(base, cal)}}
        tr.update(sweep_turnover(panel, dfac, cal, [1e7, 3e7, 5e7, 1e8, 2.5e8]))
        best_t, n_t = report(tr, 0, "min_turn", "cr")
        res["turnover_floor"] = {str(k): v for k, v in tr.items()}
        base_worst = min(c for _, c in tr[0]["wf"])
        v = ((f"CHANGE INDICATED — a Rs {best_t[0]}cr/day floor beats the top-300 rank cut "
              f"in {best_t[1]}/{best_t[2]} windows without a worse worst-window")
             if best_t and best_t[1] / best_t[2] >= 0.75 and best_t[3] >= base_worst else
             (f"INCUMBENT HOLDS — best floor Rs {best_t[0]}cr wins {best_t[1]}/{best_t[2]}"
              if best_t else "INCONCLUSIVE"))
        print(f"\n  liquidity floor: {v}")
        res["turnover_floor_verdict"] = v
        json.dump(res, open(os.path.join(REPORTS, "liquidity_floor_sweep.json"), "w"),
                  indent=1, default=float)
        print("  Saved -> reports/liquidity_floor_sweep.json")
        if only:
            return

    print("SWEEP 1 — how often to rebalance", flush=True)
    cad = sweep(panel, dfac, cal, "bars", CADENCES, DEPLOYED_TOPN, "rebal_bars")
    best_c, n_c = report(cad, DEPLOYED_BARS, "rebal_bars", "d")
    res["cadence"] = {str(k): v for k, v in cad.items()}

    print("\nSWEEP 2 — how wide to cast the net", flush=True)
    uni = sweep(panel, dfac, cal, "top_n", UNIVERSES, DEPLOYED_BARS, "top_n_liquid")
    best_u, n_u = report(uni, DEPLOYED_TOPN, "top_n", "")
    res["universe"] = {str(k): v for k, v in uni.items()}

    print("\n" + "=" * 96)
    print("  VERDICTS  (pre-registered: a challenger must win >=6 of 8 windows")
    print("             AND not have a worse worst-window than the incumbent)")
    print("=" * 96)
    for name, best, n, base_key, rows_ in (
            ("cadence", best_c, n_c, DEPLOYED_BARS, cad),
            ("universe size", best_u, n_u, DEPLOYED_TOPN, uni)):
        base_worst = min(c for _, c in rows_[base_key]["wf"])
        if best and best[1] >= 6 and best[3] >= base_worst:
            v = (f"CHANGE INDICATED — {best[0]} beats the incumbent in "
                 f"{best[1]}/{best[2]} windows without a worse worst-window")
        elif best:
            v = (f"INCUMBENT HOLDS — best challenger {best[0]} wins only "
                 f"{best[1]}/{best[2]} windows (worst {best[3]*100:+.1f}% vs "
                 f"incumbent {base_worst*100:+.1f}%)")
        else:
            v = "INCONCLUSIVE — no usable windows"
        print(f"  {name:>14}: {v}")
        res[f"{name.replace(' ', '_')}_verdict"] = v

    os.makedirs(REPORTS, exist_ok=True)
    p = os.path.join(REPORTS, "cadence_and_filter_sweep.json")
    json.dump(res, open(p, "w"), indent=1, default=float)
    print(f"\n  Saved -> {os.path.relpath(p, _ROOT)}")


if __name__ == "__main__":
    main()

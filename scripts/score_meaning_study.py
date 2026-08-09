"""What does a score actually MEAN? Forward 6-month returns by score band.
==========================================================================
The owner asked for "how much the system thinks this stock will grow in 6
months". That number does not exist and cannot be manufactured: the engine
produces a RANKING, not a return forecast, and its information coefficient is
~0.05-0.10 — it explains under 1% of the variance in any individual name's
return. Printing "expected +18%" beside a ticker would be inventing a price
target, in a chat a family member reads and may act on.

This is the honest form of the same question: not what WILL this stock do, but
what have stocks at this score ACTUALLY done, historically, over the following
six months — including how often they lost money. A base rate with its spread,
not a point forecast.

The spread is the part that matters and the part a forecast would hide. If the
top band's median is +9% but a quarter of those names still lost 12%, then "this
scored 94/100" cannot honestly be read as "this will go up", and the output has
to make that impossible to miss.

  python3 scripts/score_meaning_study.py

Writes reports/score_meaning.json, which /why reads.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (Backtester, BacktestConfig, ConstructionConfig,     # noqa: E402
                            DataPanel, PortfolioConstructor, discover_tickers,
                            load_delivery_factors, load_sector_map)

REPORTS = os.path.join(_ROOT, "reports")
END = os.environ.get("MARK5_END", "2026-07-21")
START = os.environ.get("MARK5_START", "2019-11-01")   # delivery factor archive begins here
FWD = 126        # one rebalance period ~ six months, the horizon the book trades on
STEP = 21        # evaluate monthly: more resolution, but the windows OVERLAP (see below)
WEIGHTS = {"momentum": 0.45, "low_vol": 0.15, "trend": 0.25, "stability": 0.15}

# Reported as "top N%" bands because that is what a 0-100 score means: a position
# in the field on the day, not an absolute quantity.
BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 90), (90, 100)]


def summarise(rets):
    a = np.asarray(rets, float) * 100
    if len(a) < 30:
        return None
    return {"n": int(len(a)),
            "median": round(float(np.median(a)), 2),
            "p25": round(float(np.percentile(a, 25)), 2),
            "p75": round(float(np.percentile(a, 75)), 2),
            "worst_decile": round(float(np.percentile(a, 10)), 2),
            "best_decile": round(float(np.percentile(a, 90)), 2),
            "pct_negative": round(float((a < 0).mean() * 100), 1)}


def main():
    print(f"Loading panel ({START} -> {END}) ...", flush=True)
    panel = DataPanel(discover_tickers(), END)
    dfac = load_delivery_factors(universe=panel.tickers)
    w = {**WEIGHTS, "deliv_chg": 0.10} if dfac else dict(WEIGHTS)
    print(f"  delivery factors for {len(dfac or {})} names — weights {w}", flush=True)

    bt = Backtester(panel, PortfolioConstructor(
        ConstructionConfig(mode="factor_tilt", n_hold=20, factor_weights=w),
        sector_map=load_sector_map()),
        BacktestConfig(rebal_bars=FWD, top_n_liquid=300), extra_factors=dfac or None)

    cal = panel.trading_calendar(START, END)
    dates = list(cal[::STEP])
    close = panel.close

    by_band = {f"{lo}-{hi}": [] for lo, hi in BANDS}
    top20, used = [], 0
    for d in dates:
        fi = close.index.searchsorted(d)
        if fi + FWD >= len(close.index):
            continue
        elig = [t for t in panel.eligible(d, 252, 0.0, top_n=300) if t in close.columns]
        if len(elig) < 50:
            continue
        comp, _ = bt._factor_panel(d, elig)
        comp = comp.dropna()
        if len(comp) < 50:
            continue
        fwd = ((close.iloc[fi + FWD] / close.iloc[fi]) - 1).reindex(comp.index).dropna()
        comp = comp.reindex(fwd.index)
        if len(fwd) < 50:
            continue
        used += 1

        # Score exactly as /why displays it: position in THAT DAY's field, 0-100.
        score = comp.rank(pct=True) * 100
        for lo, hi in BANDS:
            sel = fwd[(score > lo) & (score <= hi)] if lo else fwd[score <= hi]
            by_band[f"{lo}-{hi}"].extend(sel.tolist())
        # what the book actually buys, rather than a band nobody holds
        top20.extend(fwd.reindex(comp.nlargest(20).index).dropna().tolist())

    out = {"window": f"{START} to {END}", "horizon_bars": FWD,
           "evaluation_dates": used, "step_bars": STEP,
           "independent_periods": round(used * STEP / FWD, 1),
           "factor_weights": w,
           "bands": {k: summarise(v) for k, v in by_band.items()},
           "top20_actually_bought": summarise(top20),
           "caveats": [
               "Windows OVERLAP: dates step 21 bars but returns span 126, so these "
               "are not independent observations. `independent_periods` is the "
               "honest count.",
               "Single-name returns, GROSS of costs and tax. The book pays both.",
               "This is what stocks at a score DID, not what any stock WILL do. "
               "The spread between p25 and p75 is the point.",
               "Survivorship: names are those cached today, so companies that "
               "delisted during the window are under-represented and these figures "
               "are therefore optimistic.",
           ]}
    os.makedirs(REPORTS, exist_ok=True)
    p = os.path.join(REPORTS, "score_meaning.json")
    json.dump(out, open(p, "w"), indent=1, allow_nan=False)

    print(f"\n{used} evaluation dates ~ {out['independent_periods']} independent periods\n")
    print(f"{'score band':<12}{'n':>7}{'median':>9}{'p25':>9}{'p75':>9}{'lost money':>12}")
    for k, v in out["bands"].items():
        if v:
            print(f"{k:<12}{v['n']:>7}{v['median']:>8.1f}%{v['p25']:>8.1f}%"
                  f"{v['p75']:>8.1f}%{v['pct_negative']:>11.0f}%")
    t = out["top20_actually_bought"]
    if t:
        print(f"{'TOP 20':<12}{t['n']:>7}{t['median']:>8.1f}%{t['p25']:>8.1f}%"
              f"{t['p75']:>8.1f}%{t['pct_negative']:>11.0f}%")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

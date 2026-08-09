"""
Can the January config actually be BOUGHT at this book's size?
==============================================================
DEPLOYMENT_2027-01.md stages two approved changes that interact badly, and the
interaction is invisible in the evidence for either one:

  * equity 50% -> 25%  (four equal sleeves)
  * n_hold 20 -> 60    (IR 0.365 -> 0.433 on the 19-year breadth sweep)

Together they cut the rupees behind each name by SIX times. At the book's current
NAV that is about Rs 2,200 a position, and a great many NSE shares cost more than
that ONE SHARE. Names the allocator cannot afford silently get zero.

Why the breadth sweep could not have caught this: it runs in NAV units — a
scale-free world where a weight of 1.6% is always exactly 1.6% and fractional
shares are implicit. Whole shares only exist in the live book. So the evidence
for n_hold=60 is real and says nothing about whether n_hold=60 is IMPLEMENTABLE
at Rs 5 lakh.

The failure mode is worse than "some names missing". Affordability correlates
with SHARE PRICE, which is arbitrary — a company can trade at Rs 8,000 or do a
1:100 split and trade at Rs 80 with nothing about the business changed. Dropping
what it cannot afford tilts the book toward low-priced shares: a systematic bet
nobody chose, nobody tested, and nobody would disclose, because it does not
appear anywhere in the config.

  IR = IC x sqrt(BR) x TC.  n_hold=60 buys BREADTH (BR). If a third of the names
  cannot be filled, TRANSFER (TC) pays for it. This measures which wins.

  MEASURED, not argued:
    - how many target names round to zero shares
    - total weight error vs target, in percentage points
    - the price bias: median share price of what is held vs what was dropped
    - the capital at which each n_hold becomes viable

  python3 scripts/nhold_feasibility.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import scripts.paper_track as pt

REPORTS = os.path.join(_ROOT, "reports")
# A position is "comfortably fillable" if it can hold several shares; one share
# is a 100% rounding error on that name.
MIN_SHARES = 3


def assess(tgt, px, capital, sleeves=None):
    """tgt = EQUITY targets only; sleeves = the ETF legs.

    allocate() must see the WHOLE book. Handed only the equity slice — weights
    summing to 0.25 — it treats the other 75% as unspent residual and pours it
    back into those same names. The first run of this script reported Rs 224k
    invested against a Rs 130k budget and zero unaffordable names, i.e. exactly
    the opposite of the effect being measured. The sleeves are excluded from the
    statistics afterwards: they are two cheap ETFs and were never the question.
    """
    q = pt.allocate({**tgt, **(sleeves or {})}, px, capital)
    zero = [t for t in tgt if q.get(t, 0) == 0]
    thin = [t for t in tgt if 0 < q.get(t, 0) < MIN_SHARES]
    err = sum(abs(q.get(t, 0) * px[t] / capital - w) for t, w in tgt.items()) * 100
    held_px = [px[t] for t in tgt if q.get(t, 0)]
    drop_px = [px[t] for t in zero]
    return {"n_target": len(tgt), "n_zero": len(zero), "n_thin": len(thin),
            "weight_err_pp": err,
            "invested": sum(q.get(t, 0) * px[t] for t in tgt),
            "median_px_held": float(np.median(held_px)) if held_px else float("nan"),
            "median_px_dropped": float(np.median(drop_px)) if drop_px else float("nan"),
            "dropped": sorted(((px[t], t) for t in zero), reverse=True)[:8]}


def main():
    nav = float(json.load(open(os.path.join(_ROOT, "data", "paper",
                                            "paper_export.json")))["nav"])
    print(f"book NAV Rs {nav:,.0f}\n")
    res = {"nav": nav, "min_shares_for_comfort": MIN_SHARES, "configs": {}}

    # Build each candidate target book once, then price it at several capitals.
    cases = [(20, 0.50, "TODAY   50% equity / 20 names"),
             (20, 0.25, "JAN-A   25% equity / 20 names"),
             (40, 0.25, "JAN-B   25% equity / 40 names"),
             (60, 0.25, "JAN-C   25% equity / 60 names   <- as documented")]
    built = {}
    for n_hold, eq, label in cases:
        if n_hold not in built:
            pt.N_HOLD = n_hold
            w_eq, asof, _, _ = pt.target_book()
            built[n_hold] = (w_eq, asof)
        w_eq, asof = built[n_hold]
        # sorted() on a pandas Series iterates its VALUES, so this passed the
        # weights themselves as ticker symbols and yfinance dutifully looked up
        # "0.0335.NS". Build the target dict first, then price ITS keys.
        want = {t: float(x) * eq for t, x in w_eq.items()}
        sl = {k: (1 - eq) / len(pt.SLEEVES) for k in pt.SLEEVES}
        px = pt.live_prices(sorted(set(want) | set(sl)))
        tgt = {t: w for t, w in want.items() if t in px}
        a = assess(tgt, px, nav, {k: v for k, v in sl.items() if k in px})
        res["configs"][label.split()[0]] = {"n_hold": n_hold, "equity_frac": eq,
                                            "asof": str(asof.date()), **a,
                                            "dropped": [[p, t] for p, t in a["dropped"]]}
        print("=" * 78)
        print(f"  {label}")
        print("=" * 78)
        print(f"  rupees per name (pre-tilt)   Rs {nav*eq/n_hold:>10,.0f}")
        print(f"  names targeted               {a['n_target']:>10}")
        print(f"  ...that get ZERO shares      {a['n_zero']:>10}  "
              f"({a['n_zero']/max(a['n_target'],1)*100:.0f}%)")
        print(f"  ...that get 1-2 shares       {a['n_thin']:>10}  (>50% rounding error each)")
        print(f"  weight error vs target       {a['weight_err_pp']:>10.1f}pp")
        print(f"  equity actually invested     Rs {a['invested']:>10,.0f} "
              f"of Rs {nav*eq:,.0f}")
        if a["n_zero"]:
            print(f"  median share price HELD      Rs {a['median_px_held']:>10,.0f}")
            print(f"  median share price DROPPED   Rs {a['median_px_dropped']:>10,.0f}"
                  f"   <- the price bias")
            print("  dropped: " + ", ".join(f"{t} Rs{p:,.0f}" for p, t in a["dropped"][:5]))
        print()

    # At what capital does each n_hold become viable? Same target book, resized.
    print("=" * 78)
    print("  WHAT CAPITAL WOULD n_hold=60 NEED?  (25% equity sleeve)")
    print("=" * 78)
    w_eq, _ = built[60]
    want60 = {t: float(x) * 0.25 for t, x in w_eq.items()}
    sl60 = {k: 0.75 / len(pt.SLEEVES) for k in pt.SLEEVES}
    px = pt.live_prices(sorted(set(want60) | set(sl60)))
    tgt = {t: w for t, w in want60.items() if t in px}
    sl60 = {k: v for k, v in sl60.items() if k in px}
    res["capital_ladder"] = {}
    print(f"  {'capital':>14}{'zero-share names':>20}{'weight error':>15}")
    for cap in (5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7):
        a = assess(tgt, px, cap, sl60)
        res["capital_ladder"][f"{cap:.0f}"] = {k: a[k] for k in
                                               ("n_zero", "n_thin", "weight_err_pp")}
        print(f"  Rs {cap/1e5:>9,.1f}L{a['n_zero']:>16} / {a['n_target']}"
              f"{a['weight_err_pp']:>14.1f}pp")

    os.makedirs(REPORTS, exist_ok=True)
    p = os.path.join(REPORTS, "nhold_feasibility.json")
    json.dump(res, open(p, "w"), indent=1, default=float)
    print(f"\n  Saved -> {os.path.relpath(p, _ROOT)}")


if __name__ == "__main__":
    main()

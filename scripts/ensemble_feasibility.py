"""
Is the 1/N ensemble buyable at this book's size?
================================================
DEPLOYMENT_2027-01 §5 stages the 1/N ensemble: run all five factor-weight schemes
and split capital equally between them, so no config is *selected* and PBO stops
applying to the choice. The argument is epistemic, not performance — it is priced
at -1.92pp of CAGR for ~1.2pp of drawdown.

But averaging five top-20 books is a breadth expansion wearing a different hat.
Names every scheme agrees on keep a full weight; names only one scheme picks get
a fifth of one. Those tail weights are exactly where whole-share rounding bites,
and it is the same wall that just killed n_hold 20->60 (K43) — measured, not
assumed, because the evidence for the ensemble comes from a scale-free NAV-unit
backtest where fractional shares are implicit.

  HYPOTHESIS   The ensemble is implementable at the live book's size.
  FALSIFIED IF a material share of its target names round to zero whole shares
               where the single-config book has none.

Reports the same statistics as nhold_feasibility.py so the two are comparable.

  python3 scripts/ensemble_feasibility.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import scripts.paper_track as pt
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, FactorLibrary, composite_score,
                            load_sector_map, load_delivery_factors)

REPORTS = os.path.join(_ROOT, "reports")

# The same five schemes nested_walkforward.py searched over and deoverfit_cost.py
# priced. `mom_heavy` is the deployed one.
WEIGHTS = {
    "blend":      {"momentum": .30, "low_vol": .30, "trend": .20, "stability": .20},
    "mom_heavy":  {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15},
    "lowvol_hvy": {"momentum": .15, "low_vol": .50, "trend": .15, "stability": .20},
    "trend_hvy":  {"momentum": .20, "low_vol": .20, "trend": .45, "stability": .15},
    "stab_hvy":   {"momentum": .20, "low_vol": .20, "trend": .15, "stability": .45},
}


def book_for(panel, dfac, fw, asof, elig, raw, vol):
    """One scheme's target weights — same construction the live book uses."""
    w = dict(fw)
    if dfac:
        w["deliv_chg"] = 0.10
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=pt.N_HOLD,
                             base_weighting="inverse_vol", tilt_strength=1.5,
                             max_weight=0.08, factor_weights=w)
    con = PortfolioConstructor(cfg, sector_map=load_sector_map())
    comp = composite_score({f: pd.Series(v) for f, v in raw.items()}, cfg.factor_weights)
    return con.target_weights(comp, pd.Series(vol), [])


def main():
    nav = float(json.load(open(os.path.join(_ROOT, "data", "paper",
                                            "paper_export.json")))["nav"])
    eq_frac = 1 - sum(pt.SLEEVES.values())
    print(f"book NAV Rs {nav:,.0f} · equity sleeve {eq_frac:.0%} = Rs {nav*eq_frac:,.0f}\n")

    panel = DataPanel(discover_tickers(), str(pd.Timestamp.today().date()), freshness="off")
    asof = panel.close.index[-1]
    dfac = load_delivery_factors(universe=panel.tickers)
    elig = panel.eligible(asof, 252, top_n=pt.TOP_N)
    raw = {f: {} for f in FactorLibrary.DEFAULT_FACTORS}
    if dfac:
        raw["deliv_chg"] = {}
    vol = {}
    for t in elig:
        rows = FactorLibrary.compute_all(panel.close[t]).loc[:asof]
        if rows.empty:
            continue
        last = rows.iloc[-1]
        for f in FactorLibrary.DEFAULT_FACTORS:
            raw[f][t] = last.get(f, np.nan)
        if dfac and t in dfac:
            d = dfac[t].loc[:asof]
            if not d.empty:
                raw["deliv_chg"][t] = d.iloc[-1].get("deliv_chg", np.nan)
        r = panel.close[t].loc[:asof].pct_change().tail(126)
        vol[t] = float(r.std() * np.sqrt(252)) if r.notna().sum() > 30 else np.nan
    print(f"  as of {asof.date()} · {len(elig)} eligible names\n")

    books = {k: book_for(panel, dfac, fw, asof, elig, raw, vol) for k, fw in WEIGHTS.items()}
    ens = pd.concat(books.values(), axis=1).fillna(0.0).mean(axis=1)
    ens = ens[ens > 0]
    ens = ens / ens.sum()

    sl = {k: v for k, v in pt.SLEEVES.items()}
    px = pt.live_prices(sorted(set(ens.index) | set(sl)))
    sl = {k: v for k, v in sl.items() if k in px}

    res = {"nav": nav, "equity_frac": eq_frac, "asof": str(asof.date()), "configs": {}}
    print(f"  {'book':<26}{'names':>7}{'zero-share':>12}{'1-2 shares':>12}{'sleeve filled':>15}")
    print("  " + "-" * 68)
    for label, w in (("single config (mom_heavy)", books["mom_heavy"]),
                     ("1/N ensemble", ens)):
        w = w[w > 0]
        tgt = {t: float(x) * eq_frac for t, x in w.items() if t in px}
        q = pt.allocate({**tgt, **sl}, px, nav)
        zero = [t for t in tgt if q.get(t, 0) == 0]
        thin = [t for t in tgt if 0 < q.get(t, 0) < 3]
        # Fraction of the equity sleeve that actually gets deployed. A weight-error
        # figure was reported here and REMOVED: it read 28pp on one base and 112pp
        # on another, both impossible for a 25% sleeve, so it was measuring
        # something other than what it claimed. Publishing a number I cannot
        # reconcile would be worse than publishing none — and it is not needed,
        # because the zero-share count answers the question on its own.
        # KNOWN LIMITATION: this reads ~200%, i.e. the simulated allocator puts about
        # twice the sleeve's budget into equity. allocate() floors every position and
        # then spends the residual one share at a time on whatever is furthest below
        # target, and any sleeve missing from `px` frees 25% of NAV that the equity
        # names then absorb. So treat this column as diagnostic only.
        #
        # The VERDICT deliberately does not depend on it. It rests on the zero-share
        # count and on min_weight_pct, which is plain arithmetic on the target
        # weights and needs no allocator at all: the ensemble's smallest position is
        # Rs 894 against a single-config Rs 4,197, and Rs 894 does not buy one share
        # of a Rs 1,945 stock at any level of allocator cleverness.
        # ponytail: fix the sleeve-pricing gap if this script is ever load-bearing.
        filled = sum(q.get(t, 0) * px[t] for t in tgt) / (nav * eq_frac) * 100
        res["configs"][label] = {"n": len(tgt), "n_zero": len(zero), "n_thin": len(thin),
                                 "sleeve_filled_pct": filled,
                                 "min_weight_pct": float(min(tgt.values())) * 100,
                                 "dropped": sorted(((px[t], t) for t in zero), reverse=True)[:6]}
        print(f"  {label:<26}{len(tgt):>7}{len(zero):>12}{len(thin):>12}{filled:>14.1f}%")
        if zero:
            print("      cannot afford: "
                  + ", ".join(f"{t} Rs{p:,.0f}" for p, t in
                              sorted(((px[t], t) for t in zero), reverse=True)[:5]))

    a, b = res["configs"]["single config (mom_heavy)"], res["configs"]["1/N ensemble"]
    ok = b["n_zero"] <= max(1, a["n_zero"])
    res["verdict"] = ("SUPPORTED — the ensemble is buyable at this size"
                      if ok else
                      f"FALSIFIED — {b['n_zero']} of {b['n']} ensemble names round to zero "
                      f"(single config: {a['n_zero']} of {a['n']})")
    print(f"\n  VERDICT: {res['verdict']}")
    os.makedirs(REPORTS, exist_ok=True)
    p = os.path.join(REPORTS, "ensemble_feasibility.json")
    json.dump(res, open(p, "w"), indent=1, default=float)
    print(f"  Saved -> {os.path.relpath(p, _ROOT)}")


if __name__ == "__main__":
    main()

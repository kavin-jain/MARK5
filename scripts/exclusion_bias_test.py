"""
Is STRUCTURAL_EXCLUDE a look-ahead? — pricing the two hand-removed blowups
=========================================================================
`core/portfolio/universe.py` removes YESBANK and IDEA from every universe with
the note "a-priori exclusions, NOT performance-based". The justifications given
("RBI-administered bailout", "AGR overhang / perennial dilution") are facts that
did not exist at the 2016-01-01 start of the backtest: YES Bank's moratorium was
March 2020, the AGR judgement October 2019.

Both names are cached, both clear the liquidity screen by a wide margin
(YESBANK ~Rs 228cr/day, IDEA ~Rs 91cr/day median), and both fell ~94% from peak.
YESBANK in particular was a momentum darling into 2018 — exactly the profile a
momentum-heavy factor book buys.

So the page can claim a survivorship-free universe, or it can hand-remove the era's
two most famous blowups, but not both. This script measures the gap.

  MARK5_CACHE=data/pit_cache python3 scripts/exclusion_bias_test.py
"""
import json
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, PortfolioConstructor, ConstructionConfig,
                            Backtester, BacktestConfig, metrics_after_exit_tax,
                            load_ohlcv, load_nifty, load_sector_map, load_delivery_factors)
from core.portfolio import universe as U

REPORTS = os.path.join(_ROOT, "reports")
START, END = "2016-01-01", "2026-07-21"
TD, TAX = 252, 0.15
SLEEVES = {"eq": .5, "GOLDBEES": .25, "MON100": .25}
CONTESTED = {"YESBANK", "IDEA"}          # the two justified on post-2019 knowledge
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def wrap(eq_nav):
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


def run(label, exclude):
    """Backtest the deployed config against a universe built with `exclude`.

    Must mirror export_dashboard.py exactly, delivery factor included: the v7.7
    blend carries deliv_chg @10%, and a price-only stand-in ranks names
    differently. Testing an exclusion against a book you do not actually run
    answers the wrong question.
    """
    U.STRUCTURAL_EXCLUDE = set(exclude)
    tickers = U.discover_tickers()
    panel = DataPanel(tickers, END, freshness="off")
    dfac = load_delivery_factors(universe=panel.tickers)
    fw = dict(MOM)
    if dfac:
        fw["deliv_chg"] = 0.10
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=fw)
    r = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                   BacktestConfig(rebal_bars=126, top_n_liquid=300),
                   extra_factors=dfac).run(START, END)
    eq, full = metrics_after_exit_tax(r["nav_gross"], TAX), metrics_after_exit_tax(wrap(r["nav_gross"]), TAX)
    held = sorted({t for w in r["weights"].values() for t in w.index} & CONTESTED)
    # every date the contested names were actually in the book, and at what weight
    touches = [{"date": str(d.date()), "ticker": t, "weight_pct": round(float(w[t]) * 100, 2)}
               for d, w in r["weights"].items() for t in CONTESTED if t in w.index]
    print(f"  {label:<34}{full['cagr']*100:>+8.2f}%{full['sharpe_excess']:>9.2f}"
          f"{full['max_dd']*100:>+8.1f}%{full['calmar']:>8.2f}   eq-sleeve "
          f"{eq['cagr']*100:+.2f}% / DD {eq['max_dd']*100:+.1f}%", flush=True)
    return {"label": label, "universe_size": len(panel.tickers), "full": full,
            "equity_sleeve": eq, "contested_held": held, "touches": touches}


def main():
    base = set(U.STRUCTURAL_EXCLUDE)
    print(f"  contested exclusions: {sorted(CONTESTED)}")
    print(f"  structural (kept either way): {sorted(base - CONTESTED)}\n")
    print(f"  {'universe':<34}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>8}{'Calmar':>8}")
    print("  " + "-" * 86)
    shipped = run("as shipped (blowups removed)", base)
    honest = run("honest (blowups restored)", base - CONTESTED)
    print("  " + "-" * 86)

    d_cagr = (honest["full"]["cagr"] - shipped["full"]["cagr"]) * 100
    d_dd = (honest["full"]["max_dd"] - shipped["full"]["max_dd"]) * 100
    print(f"\n  Effect of the hand-exclusion on the published headline: "
          f"{-d_cagr:+.2f}pp CAGR, {-d_dd:+.2f}pp MaxDD")
    if honest["touches"]:
        print(f"  The book DID buy them once restored — {len(honest['touches'])} "
              f"position-dates:")
        for t in honest["touches"]:
            print(f"    {t['date']}  {t['ticker']:<10} {t['weight_pct']:>5.2f}%")
    else:
        print("  The factor engine never selected them even when allowed to. The "
              "exclusion is cosmetic and the honest fix is to delete it, not defend it.")

    out = {"generated": str(pd.Timestamp.today().date()), "window": [START, END],
           "contested": sorted(CONTESTED), "shipped": shipped, "honest": honest,
           "headline_effect_pp": {"cagr": -d_cagr, "max_dd": -d_dd}}
    p = os.path.join(REPORTS, "exclusion_bias.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()

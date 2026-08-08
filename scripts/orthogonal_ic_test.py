"""
P3.1 / P3.2 — incremental IC: does a signal add anything momentum doesn't?
=========================================================================
PRE-REGISTERED in docs/RESEARCH_PLAN_2026-08.md.

  HYPOTHESIS   At least one candidate signal carries material IC ORTHOGONAL to the
               existing composite, and was judged wrongly because it was scored on
               RAW IC while being correlated with a factor already in the book.
  FALSIFIED IF residual IC after regressing out the existing composite is below
               0.03 for every candidate.

WHY THIS IS NOT RE-LITIGATING A KILL. Mandate §4 Group B: eight signals were
killed on raw IC of 0.02-0.07. Raw IC cannot distinguish "carries no information"
from "carries no INCREMENTAL information". A weak signal correlated with a strong
one you already own adds nothing and dilutes the strong one — the correct test is
the IC of the RESIDUAL after the existing composite is projected out.

SCOPE LIMIT, stated honestly. The original plan named ownership flow, fundamental
quality, FIP and candlestick. **None of that data is in this repo** — there is no
shareholding directory, the fundamentals API was quota-blocked (K17), and
data/sentiment/scores is empty. F&O features cover 64 tickers, too narrow to rank
a 300-name universe. So the test runs on the signals that DO exist: the two
delivery factors, which are the only genuinely non-price signals available.
`deliv_chg` is currently deployed at 10% weight; `deliv_per_z` was rejected in its
favour. Conclusions therefore apply to the delivery family, NOT to the whole of
Group B, and the Group B verdicts remain formally untested by this run.

  MARK5_CACHE=data/pit_cache_2007 python3 scripts/orthogonal_ic_test.py
"""
import os, sys, json

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            load_sector_map, load_delivery_factors)
from core.portfolio.factors import FactorLibrary, composite_score

REPORTS = os.path.join(_ROOT, "reports")
END = os.environ.get("MARK5_END", "2026-07-21")
# delivery data begins 2019-10; anything earlier cannot test these signals
START = os.environ.get("MARK5_START", "2019-11-01")
FWD = 126           # forward horizon = one rebalance period


def spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 20:
        return np.nan
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def residualise(sig: pd.Series, base: pd.Series) -> pd.Series:
    """sig with the component explained by `base` removed (OLS residual).

    This is the whole point: what survives here is information the existing
    composite does NOT already contain.
    """
    d = pd.concat([sig, base], axis=1).dropna()
    d.columns = ["s", "b"]
    if len(d) < 20 or d["b"].std() == 0:
        return pd.Series(dtype=float)
    beta = d["s"].cov(d["b"]) / d["b"].var()
    return d["s"] - beta * d["b"]


def main():
    print(f"Loading panel ({START} -> {END}) ...", flush=True)
    panel = DataPanel(discover_tickers(), END)
    dfac = load_delivery_factors(universe=panel.tickers)
    if not dfac:
        sys.exit("No delivery factors available — nothing to test.")
    print(f"  delivery factors for {len(dfac)} names", flush=True)

    base_w = {"momentum": 0.45, "low_vol": 0.15, "trend": 0.25, "stability": 0.15}
    bt = Backtester(panel, PortfolioConstructor(
        ConstructionConfig(mode="factor_tilt", n_hold=20,
                           factor_weights={**base_w, "deliv_chg": 0.10}),
        sector_map=load_sector_map()),
        BacktestConfig(rebal_bars=126, top_n_liquid=300), extra_factors=dfac)

    cal = panel.trading_calendar(START, END)
    dates = list(cal[::FWD])[:-1]
    print(f"  {len(dates)} evaluation dates, {FWD}-bar forward horizon\n", flush=True)

    cands = ["deliv_chg", "deliv_per_z"]
    raw_ic = {c: [] for c in cands}
    res_ic = {c: [] for c in cands}
    corr_to_base = {c: [] for c in cands}
    base_ic = []

    close = panel.close
    for d in dates:
        elig = panel.eligible(d, 252, 0.0, top_n=300)
        elig = [t for t in elig if t in close.columns]
        if len(elig) < 50:
            continue
        comp, _ = bt._factor_panel(d, elig)
        # forward return over the next FWD bars, the horizon the book trades on
        fi = close.index.searchsorted(d)
        if fi + FWD >= len(close.index):
            continue
        p0, p1 = close.iloc[fi], close.iloc[fi + FWD]
        fwd = ((p1 / p0) - 1).reindex(elig).dropna()

        # composite WITHOUT the candidate, so the residual is against what is
        # genuinely already owned
        panels = {f: pd.Series({t: bt._factors[t].loc[:d].iloc[-1].get(f, np.nan)
                                for t in elig if not bt._factors[t].loc[:d].empty})
                  for f in FactorLibrary.DEFAULT_FACTORS}
        base = composite_score(panels, base_w, rank_transform=True)
        base_ic.append(spearman(base, fwd))

        for c in cands:
            sig = pd.Series({t: dfac[t].loc[:d].iloc[-1].get(c, np.nan)
                             for t in elig if t in dfac and not dfac[t].loc[:d].empty})
            sig = sig.dropna()
            if len(sig) < 50:
                continue
            raw_ic[c].append(spearman(sig, fwd))
            corr_to_base[c].append(spearman(sig, base))
            r = residualise(sig, base.reindex(sig.index))
            if len(r):
                res_ic[c].append(spearman(r, fwd))

    print("=" * 78)
    print("  INCREMENTAL IC — does the signal add what momentum does not?")
    print("=" * 78)
    bi = np.nanmean(base_ic)
    print(f"  existing composite (momentum/low_vol/trend/stability): IC {bi:+.4f}"
          f"  over {len(base_ic)} dates\n")
    print(f"  {'signal':<14}{'raw IC':>10}{'corr to base':>14}{'RESIDUAL IC':>14}{'n':>6}")
    print("  " + "-" * 62)
    out = {}
    for c in cands:
        if not raw_ic[c]:
            continue
        r, cb, ri = (np.nanmean(raw_ic[c]), np.nanmean(corr_to_base[c]),
                     np.nanmean(res_ic[c]))
        out[c] = {"raw_ic": r, "corr_to_base": cb, "residual_ic": ri,
                  "n_dates": len(raw_ic[c])}
        print(f"  {c:<14}{r:>+10.4f}{cb:>+14.4f}{ri:>+14.4f}{len(raw_ic[c]):>6}")

    passed = [c for c, v in out.items() if abs(v["residual_ic"]) >= 0.03]
    verdict = (f"SUPPORTED — {', '.join(passed)} carries residual IC >= 0.03"
               if passed else
               "FALSIFIED — no candidate carries residual IC >= 0.03")
    print(f"\n  PRE-REGISTERED VERDICT: {verdict}")

    # ---- P3.2: ablate deliv_chg from the deployed book, scored on IR ----
    print("\n" + "=" * 78)
    print("  P3.2 — ABLATION: is deliv_chg earning its 10% weight?")
    print("=" * 78)
    bt_cfg = BacktestConfig(rebal_bars=126, top_n_liquid=300)
    bench = Backtester(panel, PortfolioConstructor(
        ConstructionConfig(mode="equal_weight", base_weighting="equal")),
        bt_cfg).run(START, END)["nav_gross"]
    abl = {}
    for lab, fw, ex in [("with deliv_chg", {**base_w, "deliv_chg": 0.10}, dfac),
                        ("without", dict(base_w), None)]:
        r = Backtester(panel, PortfolioConstructor(
            ConstructionConfig(mode="factor_tilt", n_hold=20,
                               base_weighting="inverse_vol", tilt_strength=1.5,
                               max_weight=0.08, factor_weights=fw),
            sector_map=load_sector_map()), bt_cfg, extra_factors=ex).run(START, END)
        a = r["nav_gross"].pct_change(fill_method=None).dropna()
        b = bench.pct_change(fill_method=None).dropna()
        a, b = a.align(b, join="inner")
        act = (a - b).dropna()
        ir = float(act.mean() * 252 / (act.std() * np.sqrt(252))) if act.std() else 0.0
        abl[lab] = {"ir": ir, "cagr": r["metrics"]["cagr"], "max_dd": r["metrics"]["max_dd"]}
        print(f"  {lab:<16} IR {ir:+.3f}   net CAGR {r['metrics']['cagr']*100:+6.2f}%   "
              f"MaxDD {r['metrics']['max_dd']*100:+6.1f}%")
    d_ir = abl["with deliv_chg"]["ir"] - abl["without"]["ir"]
    print(f"\n  deliv_chg contributes {d_ir:+.3f} IR")
    print(f"  P3.2 VERDICT: {'KEEP' if d_ir > 0.03 else 'DROP — not earning its weight'}")

    json.dump({"window": [START, END], "base_ic": float(bi), "signals": out,
               "verdict": verdict, "ablation": abl, "deliv_chg_ir_contribution": d_ir},
              open(os.path.join(REPORTS, "orthogonal_ic_test.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/orthogonal_ic_test.json")


if __name__ == "__main__":
    main()

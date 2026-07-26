"""
Attribution: what did the SYSTEM add, versus what anyone could have bought?
==========================================================================
Half this book is two ETFs — GOLDBEES and MON100. Anybody can buy those. The
only part that is "the system" is the 20-name equity selection, and a single
blended headline hides that completely. This is the first question a quant or an
allocator asks, and until now the page could not answer it.

The decomposition, from the actual daily series (no model, no regression):

  total return
    = gold sleeve contribution        (25%, passive, buyable by anyone)
    + US Nasdaq sleeve contribution   (25%, passive, buyable by anyone)
    + equity sleeve contribution      (50%, the part this repo builds)
    + rebalancing effect              (annual sleeve reset; the residual)

and then, inside the equity sleeve, the part that is actually skill:

  equity sleeve return
    = same-universe equal-weight      (owning the universe, no selection)
    + SELECTION alpha                 (factor ranking + the 6-month refresh)

The second split is the one that matters. Beating the Nifty by holding gold and
Nasdaq is asset allocation, not stock selection, and saying so plainly is the
difference between a research artefact and a sales pitch.

  MARK5_CACHE=data/pit_cache python3 scripts/attribution.py
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
                            load_ohlcv, load_nifty, metrics, load_sector_map)

START, END = "2016-01-01", "2026-07-21"
TD = 252
SLEEVES = {"eq": 0.50, "GOLDBEES": 0.25, "MON100": 0.25}
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def cagr(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1


def main():
    panel = DataPanel(discover_tickers(), END, freshness="off")
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08, factor_weights=MOM)
    bt = BacktestConfig(rebal_bars=126, top_n_liquid=300)
    run = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                     bt).run(START, END)
    ew = Backtester(panel, PortfolioConstructor(
        ConstructionConfig(mode="equal_weight", base_weighting="equal")), bt).run(START, END)

    eq = run["nav_gross"]
    cal = eq.index
    r = {"eq": eq.pct_change(fill_method=None).fillna(0.0)}
    for k in SLEEVES:
        if k == "eq":
            continue
        r[k] = (load_ohlcv(k)["close"].astype(float)
                .reindex(cal, method="ffill").pct_change().fillna(0.0))

    # walk the deployed blend, recording each sleeve's rupee contribution
    cur = dict(SLEEVES)
    nav, out, contrib = 1.0, {}, {k: 0.0 for k in SLEEVES}
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur.values())
            for k in cur:
                gain = cur[k] * r[k].iloc[i]
                contrib[k] += gain / prev * nav      # rupee P&L attributable to k
                cur[k] += gain
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % TD == 0:
            tot = sum(cur.values())
            cur = {k: tot * SLEEVES[k] for k in SLEEVES}
    sysnav = pd.Series(out)

    total_gain = sysnav.iloc[-1] - 1.0
    yrs = (cal[-1] - cal[0]).days / 365.25
    nifty = load_nifty(True).reindex(cal, method="ffill")
    nifty_nav = nifty / nifty.iloc[0]

    # standalone sleeve CAGRs, for context
    stand = {k: cagr((1 + r[k]).cumprod()) for k in SLEEVES}
    ew_cagr = ew["metrics"]["cagr"]
    eq_cagr = run["metrics"]["cagr"]

    print("\n" + "=" * 92)
    print("  WHERE THE RETURN CAME FROM   " + f"{START} -> {END}  ({yrs:.1f} years)")
    print("=" * 92)
    print(f"  System net CAGR {cagr(sysnav)*100:+.2f}%   vs Nifty50 TRI "
          f"{cagr(nifty_nav)*100:+.2f}%   excess {(cagr(sysnav)-cagr(nifty_nav))*100:+.2f}pp\n")
    print(f"  {'sleeve':<34}{'weight':>8}{'standalone':>12}{'contribution':>14}"
          f"{'share of gain':>15}")
    print("  " + "-" * 88)
    labels = {"eq": "Equity book (THE SYSTEM)",
              "GOLDBEES": "Gold ETF (anyone can buy)",
              "MON100": "US Nasdaq-100 (anyone can buy)"}
    for k in ("eq", "GOLDBEES", "MON100"):
        print(f"  {labels[k]:<34}{SLEEVES[k]*100:>7.0f}%{stand[k]*100:>11.1f}%"
              f"{contrib[k]:>13.2f}x{contrib[k]/total_gain*100:>14.0f}%")
    resid = total_gain - sum(contrib.values())
    print(f"  {'Sleeve-rebalancing effect':<34}{'':>8}{'':>12}{resid:>13.2f}x"
          f"{resid/total_gain*100:>14.0f}%")
    print("  " + "-" * 88)
    print(f"  {'TOTAL':<34}{'100':>7}%{'':>12}{total_gain:>13.2f}x{100:>14.0f}%")

    passive = (contrib["GOLDBEES"] + contrib["MON100"]) / total_gain * 100
    print(f"\n  {passive:.0f}% of the total gain came from the two passive ETF sleeves.")
    print(f"  {contrib['eq']/total_gain*100:.0f}% came from the equity book.\n")

    print("=" * 92)
    print("  IS THE EQUITY BOOK ACTUALLY SKILL?  (equity sleeve, isolated)")
    print("=" * 92)
    print(f"  {'Own the same universe, equal-weighted':<48}{ew_cagr*100:>+8.2f}%  "
          f"<- no selection at all")
    print(f"  {'MARK6 factor selection + 6-month refresh':<48}{eq_cagr*100:>+8.2f}%")
    print("  " + "-" * 88)
    print(f"  {'SELECTION ALPHA (what the ranking earns)':<48}"
          f"{(eq_cagr-ew_cagr)*100:>+8.2f}pp/yr")
    print(f"\n  This is the honest measure of the system's stock-picking. It is "
          f"computed against\n  the SAME point-in-time universe, so it cannot be "
          f"flattered by universe choice, and\n  both sides pay the same Indian tax "
          f"and costs.\n")

    doc = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "period": {"start": START, "end": END, "years": round(yrs, 2)},
        "system_cagr": cagr(sysnav) * 100, "nifty_cagr": cagr(nifty_nav) * 100,
        "sleeves": [
            {"key": k, "label": labels[k], "weight_pct": SLEEVES[k] * 100,
             "standalone_cagr": stand[k] * 100,
             "contribution_x": contrib[k],
             "share_of_gain_pct": contrib[k] / total_gain * 100,
             "passive": k != "eq"}
            for k in ("eq", "GOLDBEES", "MON100")],
        "rebalancing_effect_x": resid,
        "rebalancing_share_pct": resid / total_gain * 100,
        "passive_share_pct": passive,
        "selection": {
            "equal_weight_cagr": ew_cagr * 100,
            "factor_cagr": eq_cagr * 100,
            "selection_alpha_pp": (eq_cagr - ew_cagr) * 100,
            "note": ("Selection alpha is measured against equal-weight of the SAME "
                     "point-in-time universe, net of the same tax and costs. The "
                     "gold and US sleeves are passive ETFs and are not skill."),
        },
    }
    p = os.path.join(_ROOT, "reports", "attribution.json")
    json.dump(doc, open(p, "w"), indent=1, default=float)
    print(f"  saved -> {p}\n")


if __name__ == "__main__":
    main()

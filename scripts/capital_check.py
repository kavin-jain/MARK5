"""
How much money do you actually need to run this? (Read this before funding anything.)
=====================================================================================
A backtest works in percentages, so it silently assumes you can buy 0.37 of a share
and that costs scale with size. Neither is true in real life:

  1. WHOLE SHARES. You cannot buy half a share on NSE. If a stock trades at Rs 3,000
     and your slot is Rs 250, you simply cannot hold it — your real portfolio is not
     the one that was backtested.
  2. FIXED COSTS. Zerodha (and every Indian broker) charges a DP/demat debit fee of
     about Rs 15.34 + GST per SCRIP PER SELL, no matter how small the position. That
     is a flat rupee cost against a percentage-based edge, so it explodes as capital
     shrinks.

This script prices both effects against today's actual holdings and prints the
minimum capital at which the strategy is worth running at all.

  python3 scripts/capital_check.py
"""
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, FactorLibrary, composite_score)

DP_FEE = 15.34 * 1.18          # Rs per scrip per SELL, incl. 18% GST (CDSL/Zerodha)
REBALANCES_PER_YEAR = 2        # semi-annual equity refresh
TURNOVER = 0.5                 # ~half the book changes at each refresh
GROSS_EDGE = 0.104             # measured excess over Nifty TRI, pp/yr (v7.2)


def main():
    panel = DataPanel(discover_tickers(), str(pd.Timestamp.today().date()), freshness="off")
    asof = panel.close.index[-1]
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08,
                             factor_weights={"momentum": 0.45, "low_vol": 0.15,
                                             "trend": 0.25, "stability": 0.15})
    elig = panel.eligible(asof, 252, 0.40)
    raw = {f: {} for f in FactorLibrary.DEFAULT_FACTORS}
    vol = {}
    for t in elig:
        row = FactorLibrary.compute_all(panel.close[t]).loc[:asof]
        if row.empty:
            continue
        last = row.iloc[-1]
        for f in raw:
            raw[f][t] = last.get(f, float("nan"))
        vol[t] = -last.get("low_vol", float("nan"))
    comp = composite_score({f: pd.Series(v) for f, v in raw.items()}, cfg.factor_weights)
    w = PortfolioConstructor(cfg).target_weights(comp, pd.Series(vol), [])
    px = {t: float(panel.close[t].loc[:asof].dropna().iloc[-1]) for t in w.index}

    print(f"\n  Today's book: {len(w)} stocks, as-of {asof.date()}")
    print(f"  Share prices: cheapest Rs {min(px.values()):,.0f}, "
          f"dearest Rs {max(px.values()):,.0f}, median Rs {sorted(px.values())[len(px)//2]:,.0f}")
    sells_per_year = len(w) * TURNOVER * REBALANCES_PER_YEAR
    print(f"  Fixed cost: ~{sells_per_year:.0f} sells/yr x Rs {DP_FEE:.0f} DP fee "
          f"= Rs {sells_per_year * DP_FEE:,.0f}/yr, WHATEVER your capital is\n")

    print(f"  {'capital':>12}{'per stock':>11}{'buyable':>9}{'DP cost/yr':>12}"
          f"{'as % cap':>10}{'edge left':>11}  verdict")
    for cap in (10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000):
        eq = cap * 0.5                      # 50% equity sleeve
        slot = eq / len(w)
        buyable = sum(1 for t, p in px.items() if p <= slot)
        dp = sells_per_year * DP_FEE
        dp_pct = dp / cap
        left = GROSS_EDGE - dp_pct
        if buyable < len(w) * 0.75:
            verdict = "BROKEN - can't buy most of the book"
        elif dp_pct > GROSS_EDGE * 0.5:
            verdict = "BAD - fees eat most of the edge"
        elif dp_pct > GROSS_EDGE * 0.2:
            verdict = "marginal"
        else:
            verdict = "OK"
        print(f"  {cap:>12,}{slot:>11,.0f}{buyable:>6}/{len(w)}{dp:>12,.0f}"
              f"{dp_pct*100:>9.2f}%{left*100:>10.1f}%  {verdict}")

    print(f"\n  Reading it: 'edge left' is the measured {GROSS_EDGE*100:.1f}pp/yr advantage over the")
    print(f"  index MINUS the flat DP fees. When that number stops being clearly positive,")
    print(f"  you are working hard to underperform a plain index fund.\n")
    print(f"  A one-line alternative that needs no minimum, no rebalancing and no discipline:")
    print(f"  a monthly SIP into a Nifty 50 index fund. At small capital it wins on cost alone.\n")


if __name__ == "__main__":
    main()

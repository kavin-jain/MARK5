"""
MARK6 — Leverage + Hedge Test (the honest hunt for ~20%)
========================================================
The council's bottleneck thesis: funds reach 20% by LEVERING a HEDGED (low-
drawdown) book. Retail can lever (MTF/futures) but hedging is dear and Indian
financing is ~14%/yr. This tests, with our own data, whether a levered and/or
Nifty-hedged MARK6 can net ~20% at a survivable drawdown — AFTER realistic
financing, hedge drag, costs and tax. No p-hacking: every config is reported.

Model (daily compounding, captures ruin/vol-drag honestly):
  NAV_t = NAV_{t-1} * (1 + L*r_long - (L-1)*fin/252 - h*r_nifty)
    L  = gross leverage on the long book (1.0 = unlevered)
    h  = Nifty hedge ratio (0 = naked long; 1.0 = fully beta-hedged short)
    fin= annual financing on borrowed (L-1) capital (Indian MTF ~14%)
  Terminal tax haircut on total gain (blended, same for all configs -> fair).

  python3 scripts/leverage_hedge_test.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, metrics)

CACHE = os.path.join(_ROOT, "data", "cache")
END = "2026-05-21"
FIN_ANNUAL = 0.14         # Indian MTF/leverage financing cost
HEDGE_DRAG_ANNUAL = 0.01  # futures roll/slippage cost on the hedged notional
TAX = 0.15                # blended terminal haircut on gains (fair across configs)
TRADING = 252


def nifty_returns(cal):
    df = pd.read_parquet(os.path.join(CACHE, "sector_NSEI.parquet"))
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
    s = df["close"].astype(float).sort_index()
    return s.reindex(cal).ffill().pct_change(fill_method=None).fillna(0.0)


def lever_hedge(r_long, r_nifty, L, h):
    fin = FIN_ANNUAL / TRADING
    drag = HEDGE_DRAG_ANNUAL / TRADING
    daily = L * r_long - max(0.0, L - 1.0) * fin - h * r_nifty - h * drag
    nav = (1 + daily).cumprod()
    return nav


def summarize(nav):
    m = metrics(nav)
    # terminal tax on total gain
    net = nav.copy()
    g = nav.iloc[-1] - 1.0
    net.iloc[-1] = nav.iloc[-1] - max(0.0, g) * TAX
    mnet = metrics(net)
    return mnet


def main():
    print("Building MARK6 long book...", flush=True)
    panel = DataPanel(discover_tickers(), END)
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=0.5, max_weight=0.08)
    bt = Backtester(panel, PortfolioConstructor(cfg))

    windows = [("2016-01-01", END, "full16-26"), ("2022-01-01", END, "recent22-26")]
    Ls = [1.0, 1.25, 1.5, 1.75, 2.0]
    hs = [0.0, 0.5, 1.0]

    for s, e, lab in windows:
        run = bt.run(s, e)
        nav_g = run["nav_gross"]
        r_long = nav_g.pct_change(fill_method=None).fillna(0.0)
        cal = nav_g.index
        r_n = nifty_returns(cal)
        base = summarize(nav_g)   # L=1,h=0 net (sanity vs known ~13%)
        print("\n" + "=" * 84)
        print(f"  {lab}   (unlevered net CAGR sanity: {base['cagr']*100:+.1f}%)")
        print(f"  financing {FIN_ANNUAL*100:.0f}%/yr · hedge drag {HEDGE_DRAG_ANNUAL*100:.0f}% · tax {TAX*100:.0f}%")
        print("=" * 84)
        print(f"  {'L':>4} {'hedge':>6} {'netCAGR':>9} {'Sharpe':>7} {'MaxDD':>8} {'flag'}")
        for L in Ls:
            for h in hs:
                nav = lever_hedge(r_long, r_n, L, h)
                m = summarize(nav)
                flag = ""
                if m["cagr"] >= 0.20 and m["max_dd"] > -0.50:
                    flag = "  <<< 20%+ & survivable"
                elif m["cagr"] >= 0.20:
                    flag = "  (20%+ but DD too deep)"
                print(f"  {L:>4.2f} {h:>6.1f} {m['cagr']*100:>+8.1f}%"
                      f"{m['sharpe']:>7.2f}{m['max_dd']*100:>+7.1f}%{flag}")
    print("\nReads: with financing≈book return, leverage adds little CAGR but multiplies"
          "\ndrawdown; hedging removes beta (most of the return) so levered-hedged nets ~0."
          "\nIf NO config shows '20%+ & survivable', that path to 20% is closed (honestly).")


if __name__ == "__main__":
    main()

"""
MARK6 — FINAL question: do we need the factor system at all, or is buy-and-hold enough?
=======================================================================================
Apples-to-apples: every contender sits inside the SAME deployed 3-sleeve wrapper
(50% equity sleeve / 25% gold / 25% US-Nasdaq, annual rebalance, net of tax).
The ONLY thing that varies is what the equity sleeve holds:

  A. mom_factor   : deployed momentum-heavy 12-name factor book (the "system")
  B. ew_universe  : equal-weight buy-and-hold of the WHOLE eligible universe
  C. nifty        : Nifty50 (cap-weighted index) as the equity sleeve
  D. pure Nifty50 B&H, no wrapper (the do-nothing benchmark)

If A does not clearly beat B and C, the factor ranking earns nothing and the
honest deliverable is just the wrapper + indexing.

  python3 scripts/system_vs_buyhold_final.py
"""
import os, sys
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, load_ohlcv, metrics)

END = "2026-06-09"
TAX = 0.15
TD = 252
MOM = {"momentum": .45, "low_vol": .15, "trend": .25, "stability": .15}


def sleeve_ret(name, cal):
    df = load_ohlcv(name)
    s = df["close"].astype(float).reindex(cal).ffill().bfill()
    return s.pct_change(fill_method=None).fillna(0.0)


def nifty_ret(cal):
    df = pd.read_parquet(os.path.join(_ROOT, "data", "cache", "sector_NSEI.parquet"))
    df.columns = [c.lower() for c in df.columns]
    idx = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
    s = pd.Series(df["close"].astype(float).values, index=idx).sort_index()
    return s.reindex(cal).ffill().bfill().pct_change(fill_method=None).fillna(0.0)


def book_ret(panel, cfg, start, end, cal):
    nav = Backtester(panel, PortfolioConstructor(cfg)).run(start, end)["nav_net"]
    return nav.pct_change(fill_method=None).fillna(0.0).reindex(cal).fillna(0.0)


def wrap(eq_ret, gold, us, w=(0.50, 0.25, 0.25)):
    rets = {"eq": eq_ret, "gold": gold, "us": us}
    weights = dict(zip(("eq", "gold", "us"), w))
    cal = eq_ret.index
    cur = dict(weights); nav, out = 1.0, {}
    for i, d in enumerate(cal):
        if i > 0:
            prev = sum(cur.values())
            for s in rets:
                cur[s] *= (1 + rets[s].iloc[i])
            nav *= sum(cur.values()) / prev
        out[d] = nav
        if i > 0 and i % TD == 0:
            tot = sum(cur.values())
            for s in cur:
                cur[s] = tot * weights[s]
    nav_s = pd.Series(out)
    n = nav_s.copy(); g = nav_s.iloc[-1] - 1
    n.iloc[-1] = nav_s.iloc[-1] - max(0, g) * TAX
    return n


def report(label, nav):
    m = metrics(nav)
    print(f"  {label:<26}{m['cagr']*100:>+7.1f}%{m['sharpe']:>8.2f}"
          f"{m['max_dd']*100:>+7.1f}%{m['calmar']:>8.2f}")
    return m


def main():
    panel = DataPanel(discover_tickers(), END)
    mom_cfg = ConstructionConfig(mode="factor_tilt", n_hold=12, base_weighting="inverse_vol",
                                 tilt_strength=1.5, max_weight=0.125, factor_weights=MOM)
    ew_cfg = ConstructionConfig(mode="equal_weight", base_weighting="equal")

    for start, lab in [("2016-01-01", "FULL 2016-2026"), ("2022-01-01", "RECENT 2022-2026")]:
        cal = panel.trading_calendar(start, END)
        gold, us, nf = sleeve_ret("GOLDBEES", cal), sleeve_ret("MON100", cal), nifty_ret(cal)
        print(f"\n{lab}  (same 50/25/25 wrapper; only equity sleeve varies)")
        print(f"  {'equity sleeve':<26}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'Calmar':>8}")
        report("A mom-factor book (system)", wrap(book_ret(panel, mom_cfg, start, END, cal), gold, us))
        report("B equal-weight universe", wrap(book_ret(panel, ew_cfg, start, END, cal), gold, us))
        report("C Nifty50 index sleeve", wrap(nf, gold, us))
        nf_nav = (1 + nf).cumprod()
        g = nf_nav.iloc[-1] - 1; nf_net = nf_nav.copy()
        nf_net.iloc[-1] = nf_nav.iloc[-1] - max(0, g) * 0.125
        report("D pure Nifty50 B&H (no wrap)", nf_net)

    # rolling 3-yr walk-forward: A vs B inside the wrapper (the decisive robustness view)
    print("\nROLLING 3-YR WALK-FORWARD — system(A) minus EW-universe(B), inside wrapper:")
    years = list(range(2016, 2024))
    diffs = []
    for y in years:
        s, e = f"{y}-01-01", min(f"{y+2}-12-31", END)
        cal = panel.trading_calendar(s, e)
        gold, us = sleeve_ret("GOLDBEES", cal), sleeve_ret("MON100", cal)
        a = metrics(wrap(book_ret(panel, mom_cfg, s, e, cal), gold, us))["cagr"]
        b = metrics(wrap(book_ret(panel, ew_cfg, s, e, cal), gold, us))["cagr"]
        diffs.append(a - b)
        print(f"  {y}-{y+2}: A {a*100:+.1f}%  B {b*100:+.1f}%  Δ {100*(a-b):+.1f}pp")
    beats = sum(1 for d in diffs if d > 0.002)
    print(f"  => system beats EW-universe {beats}/{len(years)} windows, avg {np.mean(diffs)*100:+.1f}pp")


if __name__ == "__main__":
    main()

"""
P1.1b — is the bond sleeve real, or an artefact of the proxy?  (run me)
======================================================================
PRE-REGISTERED in docs/RESEARCH_PLAN_2026-08.md.

  HYPOTHESIS   The 19-year bond-sleeve result (Sharpe 1.25->1.43, MaxDD
               -41.8%->-22.7%) survives when TLT x USDINR is replaced by the
               instrument an Indian investor can actually buy: LTGILTBEES.
  FALSIFIED IF the real instrument fails to hedge equity — i.e. its correlation
               to the equity sleeve is materially less negative than the proxy's,
               or swapping it reverses the Sharpe/drawdown improvement over the
               overlapping window.

WHY THIS MATTERS. The 19-year test used TLT (US 20y+ Treasuries) converted to INR.
That is TWO bets, not one: US duration AND rupee depreciation. In a crisis the INR
typically falls, so the proxy collects an FX tailwind that a domestic gilt ETF
does NOT. If the hedge is mostly FX rather than duration, the sleeve is a currency
trade wearing a bond costume and the 19-year numbers overstate it.

LTGILTBEES starts 2018-05, so the overlap is ~8 years. That window contains two
genuine stress events — the 2020 COVID crash and the 2022 inflation rate shock —
which is enough to test the hedging claim even though it cannot test 2008.

  MARK5_CACHE=data/pit_cache_2007 python3 scripts/bond_sleeve_validation.py
"""
import os, sys, json

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import (DataPanel, discover_tickers, PortfolioConstructor,
                            ConstructionConfig, Backtester, BacktestConfig,
                            load_sector_map, load_delivery_factors, metrics)

REPORTS = os.path.join(_ROOT, "reports")
END = os.environ.get("MARK5_END", "2026-07-21")
EQ_CACHE = os.path.join(_ROOT, "reports", "_eq_nav_2007.csv")


def _naive(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated()]


def px(sym: str, start="2006-11-01") -> pd.Series:
    h = yf.download(sym, start=start, end="2026-07-22", auto_adjust=True,
                    progress=False)["Close"]
    if hasattr(h, "columns"):
        h = h.iloc[:, 0]
    return _naive(h.dropna())


def equity_nav() -> pd.Series:
    """The factor book's NAV. Cached — it is the slow part and does not change."""
    if os.path.exists(EQ_CACHE):
        return _naive(pd.read_csv(EQ_CACHE, index_col=0, parse_dates=True).iloc[:, 0])
    print("  running the equity book (slow, cached afterwards)...", flush=True)
    panel = DataPanel(discover_tickers(), END)
    cfg = ConstructionConfig(mode="factor_tilt", n_hold=20, base_weighting="inverse_vol",
                             tilt_strength=1.5, max_weight=0.08,
                             factor_weights={"momentum": 0.45, "low_vol": 0.15,
                                             "trend": 0.25, "stability": 0.15})
    dfac = load_delivery_factors(universe=panel.tickers)
    if dfac:
        cfg.factor_weights = {**cfg.factor_weights, "deliv_chg": 0.10}
    nav = Backtester(panel, PortfolioConstructor(cfg, sector_map=load_sector_map()),
                     BacktestConfig(rebal_bars=126, top_n_liquid=300),
                     extra_factors=dfac).run("2007-01-01", END)["nav_gross"].dropna()
    nav.to_csv(EQ_CACHE)
    return _naive(nav)


def blend(rets: pd.DataFrame, weights: dict, rebal=252) -> pd.Series:
    cols = [c for c in rets.columns if weights.get(c, 0) > 0]
    w = np.array([weights[c] for c in cols], float)
    w /= w.sum()
    r = rets[cols].to_numpy()
    nav = np.empty(len(r))
    nav[0] = 1.0
    cur = w.copy()
    for i in range(1, len(r)):
        cur = cur * (1 + r[i])
        tot = cur.sum()
        nav[i] = nav[i - 1] * tot
        cur /= tot
        if i % rebal == 0:
            cur = w.copy()
    return pd.Series(nav, index=rets.index)


def main():
    print("Loading sleeves...", flush=True)
    eq = equity_nav()
    fx = px("USDINR=X").reindex(eq.index).ffill().bfill()
    gold = (px("GC=F").reindex(eq.index).ffill() * fx).dropna()
    us = (px("^NDX").reindex(eq.index).ffill() * fx).dropna()
    tlt = (px("TLT").reindex(eq.index).ffill() * fx).dropna()
    gilt = px("LTGILTBEES.NS", start="2018-01-01")

    ov = gilt.index.intersection(eq.index)
    for v in (fx, gold, us, tlt):
        ov = ov.intersection(v.index)
    print(f"  overlap window: {ov[0].date()} -> {ov[-1].date()} ({len(ov)} bars, "
          f"{len(ov)/252:.1f}y)\n", flush=True)

    S = {"eq": eq[ov], "gold": gold[ov], "us": us[ov],
         "tlt_proxy": tlt[ov], "gilt_real": gilt[ov]}
    rets = pd.DataFrame({k: v.pct_change(fill_method=None).fillna(0.0) for k, v in S.items()})

    print("=== 1. DO THE TWO BOND SERIES BEHAVE THE SAME? ===")
    c = rets["tlt_proxy"].corr(rets["gilt_real"])
    print(f"  correlation proxy vs real instrument: {c*100:.0f}%")
    for k in ("tlt_proxy", "gilt_real"):
        m = metrics(S[k] / S[k].iloc[0])
        print(f"  {k:<12} CAGR {m['cagr']*100:+6.2f}%  vol {m['vol']*100:5.1f}%  "
              f"MaxDD {m['max_dd']*100:+7.2f}%")

    print("\n=== 2. DOES THE REAL INSTRUMENT ACTUALLY HEDGE EQUITY? ===")
    print(f"  {'':<12}{'vs equity':>12}{'vs US sleeve':>14}")
    for k in ("tlt_proxy", "gilt_real", "gold"):
        print(f"  {k:<12}{rets[k].corr(rets['eq'])*100:>11.0f}%"
              f"{rets[k].corr(rets['us'])*100:>13.0f}%")
    covid = rets.loc["2020-02-01":"2020-04-30"]
    print(f"\n  During the COVID crash (Feb-Apr 2020):")
    for k in ("tlt_proxy", "gilt_real", "gold"):
        cum = (1 + covid[k]).prod() - 1
        print(f"    {k:<12} return {cum*100:+6.2f}%   corr to equity "
              f"{covid[k].corr(covid['eq'])*100:+4.0f}%")

    print("\n=== 3. DOES SWAPPING THE INSTRUMENT CHANGE THE CONCLUSION? ===")
    out = []
    books = [({"eq": .50, "gold": .25, "us": .25}, "no bonds (deployed)"),
             ({"eq": .35, "gold": .20, "us": .20, "tlt_proxy": .25}, "25% bonds — PROXY"),
             ({"eq": .35, "gold": .20, "us": .20, "gilt_real": .25}, "25% bonds — REAL")]
    print(f"  {'book':<24}{'CAGR':>9}{'Sharpe':>8}{'vol':>7}{'MaxDD':>9}{'Calmar':>8}")
    print("  " + "-" * 66)
    for w, lab in books:
        m = metrics(blend(rets, w))
        out.append({"book": lab, **{k: m[k] for k in
                                    ("cagr", "sharpe", "vol", "max_dd", "calmar")}})
        print(f"  {lab:<24}{m['cagr']*100:>+8.2f}%{m['sharpe']:>8.2f}"
              f"{m['vol']*100:>6.1f}%{m['max_dd']*100:>+8.2f}%{m['calmar']:>8.2f}")

    base, proxy, real = out[0], out[1], out[2]
    verdict = ("SUPPORTED — the real instrument reproduces the proxy's improvement"
               if real["sharpe"] > base["sharpe"] and real["max_dd"] > base["max_dd"]
               else "FALSIFIED — the real instrument does not deliver the hedge")
    print(f"\n  PRE-REGISTERED VERDICT: {verdict}")
    print(f"  proxy overstatement: Sharpe {proxy['sharpe']-real['sharpe']:+.2f}, "
          f"MaxDD {(proxy['max_dd']-real['max_dd'])*100:+.2f}pp")

    res = {"window": [str(ov[0].date()), str(ov[-1].date())], "years": len(ov) / 252,
           "proxy_vs_real_corr": float(c), "books": out, "verdict": verdict,
           "corr_to_equity": {k: float(rets[k].corr(rets["eq"]))
                              for k in ("tlt_proxy", "gilt_real", "gold")}}
    json.dump(res, open(os.path.join(REPORTS, "bond_sleeve_validation.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/bond_sleeve_validation.json")
    return res


if __name__ == "__main__":
    main()

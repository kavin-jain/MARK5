"""
Is the 5-sleeve book real diversification, or just a bigger short-INR bet?
=========================================================================
R2 found that adding short-duration US Treasuries (SHY) cut drawdown from -17.09%
to -13.09% and lifted Calmar 0.95 -> 1.09. But SHY is a USD asset, so the
5-sleeve book is FOUR of five sleeves USD-denominated — about 80% of the book,
against 75% for the 4-sleeve version.

A1 established that ~90% of this book's crisis protection came from the rupee
falling, not from assets failing to fall together. So the apparent improvement
may simply be more short-INR exposure wearing a diversification costume.

  HYPOTHESIS   The 5-sleeve drawdown advantage survives with the exchange rate
               frozen — i.e. it is diversification, not currency.
  FALSIFIED IF with FX frozen the 5-sleeve book no longer beats the 4-sleeve one
               on drawdown, or the gap collapses to near nothing.

A third book is included as the decisive control: the 4-sleeve book re-weighted to
carry the SAME 80% USD exposure as the 5-sleeve one. If that control matches the
5-sleeve book, then the fifth sleeve added nothing and only the currency weight
mattered.

  python3 scripts/five_sleeve_fx_test.py
"""
import os, sys, json, time

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import metrics

REPORTS = os.path.join(_ROOT, "reports")
EQ_CACHE = os.path.join(REPORTS, "_eq_nav_2007.csv")
USD_SLEEVES = ("gold", "us", "bond", "shortbond")

BOOKS = {
    "4-sleeve  (75% USD)": {"eq": .25, "gold": .25, "us": .25, "bond": .25},
    "5-sleeve  (80% USD)": {"eq": .20, "gold": .20, "us": .20, "bond": .20,
                            "shortbond": .20},
    "4-sleeve CONTROL (80% USD)": {"eq": .20, "gold": .2667, "us": .2667,
                                   "bond": .2666},
}
CRISES = [("2008-01-01", "2009-03-31", "2008 GFC"),
          ("2020-02-01", "2020-04-30", "2020 COVID"),
          ("2022-01-01", "2022-10-31", "2022 rate shock")]


def _nz(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated()]


def px(sym, tries=4):
    last = 0
    for a in range(tries):
        try:
            h = yf.download(sym, start="2006-11-01", end="2026-07-22",
                            auto_adjust=True, progress=False)["Close"]
            if hasattr(h, "columns"):
                h = h.iloc[:, 0]
            h = h.dropna()
            last = len(h)
            if last >= 1000:
                return _nz(h)
        except Exception:
            pass
        time.sleep(3 * (a + 1))
    sys.exit(f"ABORT: {sym} returned {last} rows after {tries} attempts.")


def blend(rets, w, rebal=252):
    cols = [c for c in rets.columns if w.get(c, 0) > 0]
    wt = np.array([w[c] for c in cols], float)
    wt /= wt.sum()
    r = rets[cols].to_numpy()
    nav = np.empty(len(r))
    nav[0] = 1.0
    cur = wt.copy()
    for i in range(1, len(r)):
        cur = cur * (1 + r[i])
        tot = cur.sum()
        nav[i] = nav[i - 1] * tot
        cur /= tot
        if i % rebal == 0:
            cur = wt.copy()
    return pd.Series(nav, index=rets.index)


def main():
    eq = _nz(pd.read_csv(EQ_CACHE, index_col=0, parse_dates=True).iloc[:, 0])
    fx = px("USDINR=X").reindex(eq.index).ffill().bfill()
    usd = {"gold": px("GC=F"), "us": px("^NDX"), "bond": px("TLT"),
           "shortbond": px("SHY")}
    idx = eq.index
    for v in usd.values():
        idx = idx.intersection(v.index)
    eq, fx = eq[idx], fx[idx]
    usd = {k: v.reindex(idx).ffill() for k, v in usd.items()}
    print(f"aligned {idx[0].date()} -> {idx[-1].date()} ({len(idx)} bars)\n")

    def build(fx_series):
        d = {"eq": eq}
        for k, v in usd.items():
            d[k] = (v * fx_series).dropna()
        return pd.DataFrame({k: v.reindex(idx).pct_change(fill_method=None).fillna(0.0)
                             for k, v in d.items()})

    live = build(fx)
    # Frozen for the WHOLE history: the pure no-currency-tailwind counterfactual.
    flat = pd.Series(fx.iloc[0], index=idx)
    froz = build(flat)
    res = {"window": [str(idx[0].date()), str(idx[-1].date())], "books": {}}

    print("=" * 96)
    print("  FULL PERIOD — live FX vs FX frozen for the entire history")
    print("=" * 96)
    print(f"  {'book':<30}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}"
          f"{'| frozen CAGR':>15}{'Sharpe':>8}{'MaxDD':>9}")
    print("  " + "-" * 90)
    for lab, w in BOOKS.items():
        ml = metrics(blend(live, w))
        mf = metrics(blend(froz, w))
        res["books"][lab] = {"live": {k: ml[k] for k in ("cagr", "sharpe", "max_dd", "calmar")},
                             "frozen": {k: mf[k] for k in ("cagr", "sharpe", "max_dd", "calmar")},
                             "usd_pct": sum(v for k, v in w.items() if k in USD_SLEEVES)}
        print(f"  {lab:<30}{ml['cagr']*100:>+8.2f}%{ml['sharpe']:>8.2f}"
              f"{ml['max_dd']*100:>+8.2f}%{ml['calmar']:>8.2f}"
              f"{mf['cagr']*100:>+14.2f}%{mf['sharpe']:>8.2f}{mf['max_dd']*100:>+8.2f}%")

    print("\n" + "=" * 96)
    print("  CRISES with FX frozen at the pre-crisis level")
    print("=" * 96)
    print(f"  {'episode':<18}" + "".join(f"{l.split()[0]:>26}" for l in BOOKS))
    res["crises"] = {}
    for a, b, lab in CRISES:
        win = idx[(idx >= a) & (idx <= b)]
        if len(win) < 20:
            continue
        pre = fx.loc[:a]
        f2 = fx.copy()
        f2.loc[win] = pre.iloc[-1] if len(pre) else fx.loc[win].iloc[0]
        r2 = build(f2)
        cells, rec = "", {}
        for bl, w in BOOKS.items():
            n = blend(r2, w).loc[a:b]
            ret = (n.iloc[-1] / n.iloc[0] - 1) * 100
            dd = float((n / n.cummax() - 1).min()) * 100
            rec[bl] = {"return": ret, "dd": dd}
            cells += f"{ret:>+13.1f}% dd{dd:>+9.1f}%"
        res["crises"][lab] = rec
        print(f"  {lab:<18}{cells}")

    b4 = res["books"]["4-sleeve  (75% USD)"]
    b5 = res["books"]["5-sleeve  (80% USD)"]
    ctl = res["books"]["4-sleeve CONTROL (80% USD)"]
    live_gap = (b5["live"]["max_dd"] - b4["live"]["max_dd"]) * 100
    froz_gap = (b5["frozen"]["max_dd"] - b4["frozen"]["max_dd"]) * 100
    ctl_gap = (b5["frozen"]["max_dd"] - ctl["frozen"]["max_dd"]) * 100

    print("\n" + "=" * 96)
    print("  VERDICT")
    print("=" * 96)
    print(f"  drawdown advantage of 5-sleeve over 4-sleeve, LIVE FX   : {live_gap:+.2f}pp")
    print(f"  same advantage with FX FROZEN                          : {froz_gap:+.2f}pp")
    print(f"  vs the 80%-USD CONTROL (isolates the extra sleeve)     : {ctl_gap:+.2f}pp")
    verdict = ("SUPPORTED — the fifth sleeve is real diversification"
               if froz_gap > 1.0 and ctl_gap > 0.5 else
               "FALSIFIED — the gain is currency weight, not the extra sleeve")
    print(f"\n  {verdict}")
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(REPORTS, "five_sleeve_fx_test.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/five_sleeve_fx_test.json")


if __name__ == "__main__":
    main()

"""
A1/A2 — how much of this book is a currency trade, and is the naive book better?
================================================================================
The 2008 decomposition showed the USD sleeves returned +3.6% in USD terms and
+26.9% after a 29.4% rupee collapse. So what carried the book through the GFC was
being SHORT THE RUPEE, not assets failing to fall together. That is a real and
legitimate exposure for an INR investor, but it must be measured rather than
described as diversification.

Two questions, both pre-registered:

  A1  HYPOTHESIS   The book survives a crisis even if the rupee does NOT fall.
      FALSIFIED IF freezing the exchange rate through each crisis turns the
                   book's drawdown into something comparable to Nifty's.

  A2  HYPOTHESIS   The searched 30/30/10/30 allocation beats naive equal weights
                   out-of-sample.
      FALSIFIED IF naive 25/25/25/25 matches or beats it on rolling windows.
                   (The full-period audit already suggests it does — this checks
                   whether that holds window by window rather than on one number.)

FX-FREEZE METHOD. For each crisis window the USDINR series is pinned at its value
on the day before the window opens, so the USD sleeves deliver their local-currency
return with no translation gain. This is deliberately harsher than a real hedge: a
genuine INR-hedged position would also EARN the forward premium (INR rates exceed
USD rates, historically ~3-5%/yr), so the true hedged outcome sits between the
frozen and unhedged cases. Frozen is the pessimistic bound, which is the useful one.

  python3 scripts/currency_stress.py
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
SEARCHED = {"eq": .30, "gold": .30, "us": .10, "bond": .30}
NAIVE = {"eq": .25, "gold": .25, "us": .25, "bond": .25}
CRISES = [
    ("2008-01-01", "2009-03-31", "2008 GFC"),
    ("2010-11-01", "2011-12-31", "2011 Euro debt"),
    ("2013-05-01", "2013-09-30", "2013 Taper"),
    ("2015-04-01", "2016-02-29", "2015-16 China"),
    ("2018-01-01", "2019-02-28", "2018 IL&FS"),
    ("2020-02-01", "2020-04-30", "2020 COVID"),
    ("2022-01-01", "2022-10-31", "2022 rate shock"),
]


def _naive_idx(s):
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
                return _naive_idx(h)
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


def dd(nav):
    return float((nav / nav.cummax() - 1).min())


def main():
    eq = _naive_idx(pd.read_csv(EQ_CACHE, index_col=0, parse_dates=True).iloc[:, 0])
    fx = px("USDINR=X").reindex(eq.index).ffill().bfill()
    usd = {"gold": px("GC=F"), "us": px("^NDX"), "bond": px("TLT")}
    nifty = px("^NSEI")
    idx = eq.index
    for v in usd.values():
        idx = idx.intersection(v.index)
    idx = idx.intersection(nifty.index)
    eq, fx, nifty = eq[idx], fx[idx], nifty[idx]
    usd = {k: v.reindex(idx).ffill() for k, v in usd.items()}

    def build(fx_series):
        d = {"eq": eq}
        for k, v in usd.items():
            d[k] = (v * fx_series).dropna()
        return pd.DataFrame({k: v.reindex(idx).pct_change(fill_method=None).fillna(0.0)
                             for k, v in d.items()})

    rets_live = build(fx)
    res = {"crises": []}

    print("=" * 92)
    print("  A1. WHAT IF THE RUPEE DOESN'T FALL? (FX frozen at the pre-crisis level)")
    print("=" * 92)
    print(f"  {'episode':<20}{'USDINR':>9}{'BOOK live':>12}{'BOOK frozen':>13}"
          f"{'cost':>9}{'NIFTY':>9}")
    print("  " + "-" * 82)
    for a, b, lab in CRISES:
        win = idx[(idx >= a) & (idx <= b)]
        if len(win) < 20:
            continue
        pre = fx.loc[:a]
        frozen = fx.copy()
        frozen.loc[win] = pre.iloc[-1] if len(pre) else fx.loc[win].iloc[0]
        rl = blend(rets_live, SEARCHED).loc[a:b]
        rf = blend(build(frozen), SEARCHED).loc[a:b]
        n = nifty.loc[a:b]
        fx_move = (fx.loc[win].iloc[-1] / fx.loc[win].iloc[0] - 1) * 100
        live_r = (rl.iloc[-1] / rl.iloc[0] - 1) * 100
        froz_r = (rf.iloc[-1] / rf.iloc[0] - 1) * 100
        nif_r = (n.iloc[-1] / n.iloc[0] - 1) * 100
        res["crises"].append({"episode": lab, "fx_move_pct": fx_move,
                              "book_live_pct": live_r, "book_frozen_pct": froz_r,
                              "fx_contribution_pp": live_r - froz_r,
                              "book_frozen_dd": dd(rf / rf.iloc[0]),
                              "nifty_pct": nif_r})
        print(f"  {lab:<20}{fx_move:>+8.1f}%{live_r:>+11.1f}%{froz_r:>+12.1f}%"
              f"{live_r-froz_r:>+8.1f}{nif_r:>+8.1f}%")

    worst = min(res["crises"], key=lambda c: c["book_frozen_pct"])
    beat = sum(1 for c in res["crises"] if c["book_frozen_pct"] > c["nifty_pct"])
    a1 = ("SUPPORTED — survives without the currency tailwind"
          if worst["book_frozen_pct"] > -25 and beat >= 5 else
          "FALSIFIED — the crisis protection is mostly the rupee")
    print(f"\n  worst frozen-FX crisis: {worst['episode']} {worst['book_frozen_pct']:+.1f}% "
          f"(drawdown {worst['book_frozen_dd']*100:+.1f}%)")
    print(f"  beats Nifty with FX frozen: {beat}/{len(res['crises'])}")
    print(f"  A1 VERDICT: {a1}")
    res["a1_verdict"] = a1

    print("\n" + "=" * 92)
    print("  A2. SEARCHED 30/30/10/30 vs NAIVE 25/25/25/25 — rolling windows")
    print("=" * 92)
    res["a2"] = {}
    for yrs in (3, 5):
        span = int(252 * yrs)
        out = {}
        for lab, w in (("searched", SEARCHED), ("naive", NAIVE)):
            nav = blend(rets_live, w)
            v = [(nav.iloc[i + span] / nav.iloc[i]) ** (1 / yrs) - 1
                 for i in range(0, len(nav) - span, 21)]
            d = [dd(nav.iloc[i:i + span] / nav.iloc[i])
                 for i in range(0, len(nav) - span, 21)]
            out[lab] = {"worst": min(v), "median": float(np.median(v)),
                        "worst_dd": min(d), "n": len(v)}
        res["a2"][f"{yrs}y"] = out
        s, n = out["searched"], out["naive"]
        print(f"  {yrs}-year windows (n={s['n']}):")
        print(f"    searched  worst {s['worst']*100:+6.2f}%  median {s['median']*100:+6.2f}%"
              f"  worst DD {s['worst_dd']*100:+7.2f}%")
        print(f"    naive     worst {n['worst']*100:+6.2f}%  median {n['median']*100:+6.2f}%"
              f"  worst DD {n['worst_dd']*100:+7.2f}%")
    m5 = res["a2"]["5y"]
    a2 = ("SUPPORTED — searching beat naive" if
          m5["searched"]["median"] > m5["naive"]["median"] and
          m5["searched"]["worst_dd"] > m5["naive"]["worst_dd"] else
          "FALSIFIED — naive equal weights are as good or better; drop the search")
    print(f"\n  A2 VERDICT: {a2}")
    res["a2_verdict"] = a2

    json.dump(res, open(os.path.join(REPORTS, "currency_stress.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/currency_stress.json")


if __name__ == "__main__":
    main()

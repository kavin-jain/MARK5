"""
Stress test of the 30/30/10/30 book against every crisis in the record.
======================================================================
A CAGR is an average, and averages hide the thing that actually decides whether
a portfolio survives contact with a human being: what it does on the worst days,
how deep the hole gets, and how long you sit in it.

This runs the candidate allocation (equity 30 / gold 30 / US 10 / bond 30) through
every named crisis since 2007, every calendar year, every rolling 3- and 5-year
window, and the worst possible entry date, and compares each against Nifty 50 —
the thing you would otherwise hold.

PROXY LIMITS, unchanged from the rest of this work: passive sleeves are the
underlying assets in INR (gold spot, ^NDX, TLT x USDINR), not the Indian ETF
wrappers, so ETF expense (~0.5%/yr), tracking error and the premium international
ETFs trade at are excluded. Figures are GROSS of tax; the equity sleeve's own
backtest is net, but the blend arithmetic here is not. Treat as indicative of
SHAPE and RELATIVE behaviour, which is what a stress test is for.

  python3 scripts/stress_test.py
"""
import os, sys, json

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio import metrics

REPORTS = os.path.join(_ROOT, "reports")
EQ_CACHE = os.path.join(REPORTS, "_eq_nav_2007.csv")
BOOK = {"eq": .30, "gold": .30, "us": .10, "bond": .30}

# Named episodes. Dates bracket the drawdown as it was experienced, not as it is
# labelled in hindsight.
CRISES = [
    ("2008-01-01", "2009-03-31", "2008 Global Financial Crisis"),
    ("2010-11-01", "2011-12-31", "2011 Euro debt / India inflation"),
    ("2013-05-01", "2013-09-30", "2013 Taper Tantrum (INR -20%)"),
    ("2015-04-01", "2016-02-29", "2015-16 China deval / oil crash"),
    ("2018-01-01", "2019-02-28", "2018 IL&FS / midcap collapse"),
    ("2020-02-01", "2020-04-30", "2020 COVID crash"),
    ("2022-01-01", "2022-10-31", "2022 inflation / rate shock"),
]


def _naive(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated()]


def px(sym):
    h = yf.download(sym, start="2006-11-01", end="2026-07-22",
                    auto_adjust=True, progress=False)["Close"]
    if hasattr(h, "columns"):
        h = h.iloc[:, 0]
    h = h.dropna()
    if len(h) < 1000:
        sys.exit(f"ABORT: {sym} returned {len(h)} rows — refusing to stress-test "
                 f"against an incomplete series.")
    return _naive(h)


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


def dd_series(nav):
    return nav / nav.cummax() - 1


def recovery_days(nav):
    """Longest run of days spent below a previous peak — the underwater period.

    This is the number people actually experience. A -15% drawdown that recovers
    in four months and one that takes four years are the same number and a
    completely different life.
    """
    dd = dd_series(nav)
    under, worst, cur = dd < -1e-9, 0, 0
    for flag in under:
        cur = cur + 1 if flag else 0
        worst = max(worst, cur)
    return worst


def main():
    if not os.path.exists(EQ_CACHE):
        sys.exit("Missing reports/_eq_nav_2007.csv — run bond_sleeve_validation.py first.")
    eq = _naive(pd.read_csv(EQ_CACHE, index_col=0, parse_dates=True).iloc[:, 0])
    fx = px("USDINR=X").reindex(eq.index).ffill().bfill()
    sl = {"eq": eq}
    for k, s in (("gold", "GC=F"), ("us", "^NDX"), ("bond", "TLT")):
        sl[k] = (px(s).reindex(eq.index).ffill() * fx).dropna()
    nifty = px("^NSEI")
    idx = eq.index
    for v in sl.values():
        idx = idx.intersection(v.index)
    idx = idx.intersection(nifty.index)
    rets = pd.DataFrame({k: v[idx].pct_change(fill_method=None).fillna(0.0)
                         for k, v in sl.items()})
    book, nif = blend(rets, BOOK), nifty[idx]
    res = {"allocation": BOOK, "window": [str(idx[0].date()), str(idx[-1].date())]}

    print("=" * 84)
    print("  1. EVERY CRISIS SINCE 2007")
    print("=" * 84)
    print(f"  {'episode':<36}{'BOOK ret':>10}{'BOOK DD':>10}{'NIFTY ret':>11}{'NIFTY DD':>10}")
    print("  " + "-" * 78)
    res["crises"] = []
    for a, b, lab in CRISES:
        w, n = book.loc[a:b], nif.loc[a:b]
        if len(w) < 20:
            continue
        wr, nr = w.iloc[-1] / w.iloc[0] - 1, n.iloc[-1] / n.iloc[0] - 1
        wd, nd = dd_series(w / w.iloc[0]).min(), dd_series(n / n.iloc[0]).min()
        res["crises"].append({"episode": lab, "book_return": wr, "book_dd": wd,
                              "nifty_return": nr, "nifty_dd": nd})
        print(f"  {lab:<36}{wr*100:>+9.1f}%{wd*100:>+9.1f}%{nr*100:>+10.1f}%{nd*100:>+9.1f}%")

    print("\n" + "=" * 84)
    print("  2. EVERY CALENDAR YEAR")
    print("=" * 84)
    yrs = sorted({d.year for d in idx})
    res["years"] = {}
    line1, line2 = "  ", "  "
    for y in yrs:
        w = book.loc[f"{y}-01-01":f"{y}-12-31"]
        n = nif.loc[f"{y}-01-01":f"{y}-12-31"]
        if len(w) < 60:
            continue
        wr, nr = w.iloc[-1] / w.iloc[0] - 1, n.iloc[-1] / n.iloc[0] - 1
        res["years"][y] = {"book": wr, "nifty": nr}
        line1 += f"{y:>8}"
        line2 += f"{wr*100:>+7.1f}%"
    print(line1 + "\n" + line2)
    down = [y for y, v in res["years"].items() if v["book"] < 0]
    ndown = [y for y, v in res["years"].items() if v["nifty"] < 0]
    print(f"\n  losing years: BOOK {len(down)}/{len(res['years'])} {down}")
    print(f"                NIFTY {len(ndown)}/{len(res['years'])} {ndown}")

    print("\n" + "=" * 84)
    print("  3. ROLLING WINDOWS — could a bad entry date ruin you?")
    print("=" * 84)
    res["rolling"] = {}
    for yrs_n in (1, 3, 5):
        span = int(252 * yrs_n)
        vals = [(book.iloc[i + span] / book.iloc[i]) ** (1 / yrs_n) - 1
                for i in range(0, len(book) - span, 21)]
        nvals = [(nif.iloc[i + span] / nif.iloc[i]) ** (1 / yrs_n) - 1
                 for i in range(0, len(nif) - span, 21)]
        pos = sum(1 for v in vals if v > 0) / len(vals)
        res["rolling"][f"{yrs_n}y"] = {"worst": min(vals), "median": float(np.median(vals)),
                                       "best": max(vals), "pct_positive": pos,
                                       "nifty_worst": min(nvals)}
        print(f"  {yrs_n}-year windows (n={len(vals)}):  worst {min(vals)*100:+6.1f}%   "
              f"median {np.median(vals)*100:+6.1f}%   best {max(vals)*100:+6.1f}%   "
              f"positive {pos*100:.0f}%   [Nifty worst {min(nvals)*100:+.1f}%]")

    print("\n" + "=" * 84)
    print("  4. THE HOLE — depth and how long you sit in it")
    print("=" * 84)
    bd, nd = dd_series(book), dd_series(nif / nif.iloc[0])
    print(f"  worst drawdown       BOOK {bd.min()*100:+.2f}%     NIFTY {nd.min()*100:+.2f}%")
    print(f"  longest underwater   BOOK {recovery_days(book)} trading days "
          f"({recovery_days(book)/252:.1f}y)     "
          f"NIFTY {recovery_days(nif)} ({recovery_days(nif)/252:.1f}y)")
    print(f"  days below -10%      BOOK {int((bd < -0.10).sum())}     "
          f"NIFTY {int((nd < -0.10).sum())}")
    print(f"  worst single day     BOOK {rets.mul(pd.Series(BOOK)).sum(axis=1).min()*100:+.2f}%"
          f"     NIFTY {nif.pct_change().min()*100:+.2f}%")
    res["drawdown"] = {"book_max_dd": float(bd.min()), "nifty_max_dd": float(nd.min()),
                       "book_underwater_days": recovery_days(book),
                       "nifty_underwater_days": recovery_days(nif)}

    print("\n" + "=" * 84)
    print("  5. WORST POSSIBLE ENTRY — buy at the single worst date")
    print("=" * 84)
    fwd5 = int(252 * 5)
    entries = [(book.index[i], (book.iloc[i + fwd5] / book.iloc[i]) ** 0.2 - 1)
               for i in range(0, len(book) - fwd5, 5)]
    worst_d, worst_r = min(entries, key=lambda x: x[1])
    print(f"  worst 5-year outcome from any entry: {worst_r*100:+.2f}%/yr "
          f"(entered {worst_d.date()})")
    nentries = [(nif.index[i], (nif.iloc[i + fwd5] / nif.iloc[i]) ** 0.2 - 1)
                for i in range(0, len(nif) - fwd5, 5)]
    nw_d, nw_r = min(nentries, key=lambda x: x[1])
    print(f"  same for NIFTY:                      {nw_r*100:+.2f}%/yr "
          f"(entered {nw_d.date()})")
    res["worst_entry"] = {"book_5y_cagr": worst_r, "book_date": str(worst_d.date()),
                          "nifty_5y_cagr": nw_r, "nifty_date": str(nw_d.date())}

    full = metrics(book)
    res["full"] = {k: full[k] for k in ("cagr", "sharpe", "vol", "max_dd", "calmar")}
    print("\n" + "=" * 84)
    print(f"  FULL PERIOD: CAGR {full['cagr']*100:+.2f}%  Sharpe {full['sharpe']:.2f}  "
          f"vol {full['vol']*100:.1f}%  MaxDD {full['max_dd']*100:+.2f}%  "
          f"Calmar {full['calmar']:.2f}")
    print("=" * 84)

    json.dump(res, open(os.path.join(REPORTS, "stress_test.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/stress_test.json")


if __name__ == "__main__":
    main()

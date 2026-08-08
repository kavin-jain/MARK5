"""
Audit: was the 30/30/10/30 allocation chosen on the data it was then tested on?
==============================================================================
Yes, it was. The min-drawdown search ran over the full 2007-2026 window and the
stress test then reported results on that same window. That is in-sample
selection — the precise error this project's PBO work exists to detect, committed
by the author while writing the tools that detect it.

This measures the damage honestly:

  A. Pick the allocation on 2007-2015 ONLY. Test on 2016-2026. Never-seen data.
  B. Pick on 2016-2026 ONLY. Test on 2007-2015.
  C. Compare both against the full-sample-optimised choice and against a naive
     equal-weight-across-sleeves book that required no choosing at all.

If the honestly-chosen allocations land near the full-sample one, the selection
cost is small and the headline stands with a caveat. If they scatter, the
30/30/10/30 result is an artefact and must be withdrawn.

  python3 scripts/audit_allocation.py
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
SPLIT = "2015-12-31"


def _naive(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated()]


def px(sym, tries=4):
    """Fetch with retry. yfinance fails transiently often enough that a single
    attempt turns a network blip into an aborted run — but the length guard stays,
    because silently proceeding on a short series is far worse than stopping."""
    import time
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
                return _naive(h)
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


def search(rets, lo, hi, min_cagr=0.14):
    """Same search as before: minimise drawdown subject to a return floor,
    no sleeve above 30% except equity, equity at least 30%."""
    sub = rets.loc[lo:hi]
    best = None
    g = [i / 20 for i in range(7)]
    for e in (.30, .35, .40):
        for go in g:
            for u in g:
                bd = 1 - e - go - u
                if bd < 0 or bd > .30:
                    continue
                w = {"eq": e, "gold": go, "us": u, "bond": bd}
                m = metrics(blend(sub, w))
                if m["cagr"] < min_cagr:
                    continue
                if best is None or m["max_dd"] > best[1]["max_dd"]:
                    best = (w, m)
    return best


def show(rets, w, lo, hi, label):
    m = metrics(blend(rets.loc[lo:hi], w))
    print(f"  {label:<34} CAGR {m['cagr']*100:+6.2f}%  Sharpe {m['sharpe']:5.2f}  "
          f"MaxDD {m['max_dd']*100:+7.2f}%  Calmar {m['calmar']:.2f}")
    return {k: m[k] for k in ("cagr", "sharpe", "max_dd", "calmar")}


def main():
    eq = _naive(pd.read_csv(EQ_CACHE, index_col=0, parse_dates=True).iloc[:, 0])
    fx = px("USDINR=X").reindex(eq.index).ffill().bfill()
    sl = {"eq": eq}
    for k, s in (("gold", "GC=F"), ("us", "^NDX"), ("bond", "TLT")):
        sl[k] = (px(s).reindex(eq.index).ffill() * fx).dropna()
    idx = eq.index
    for v in sl.values():
        idx = idx.intersection(v.index)
    rets = pd.DataFrame({k: v[idx].pct_change(fill_method=None).fillna(0.0)
                         for k, v in sl.items()})
    lo, hi = str(idx[0].date()), str(idx[-1].date())
    res = {"window": [lo, hi]}

    full = search(rets, lo, hi)
    early = search(rets, lo, SPLIT)
    late = search(rets, "2016-01-01", hi)
    naive_w = {"eq": .25, "gold": .25, "us": .25, "bond": .25}

    print("=" * 88)
    print("  WHICH ALLOCATION DOES EACH PERIOD CHOOSE?")
    print("=" * 88)
    for lab, b in (("full sample 2007-2026 (what was reported)", full),
                   ("first half 2007-2015 only", early),
                   ("second half 2016-2026 only", late)):
        w = b[0]
        print(f"  {lab:<42} " + "  ".join(f"{k} {v*100:.0f}%" for k, v in w.items() if v > 0))
    res["choices"] = {"full": full[0], "early": early[0], "late": late[0]}

    print("\n" + "=" * 88)
    print("  HONEST OUT-OF-SAMPLE TEST — chosen on one half, measured on the other")
    print("=" * 88)
    print("\n  Tested on 2016-2026 (unseen by the 'early' choice):")
    res["oos_late"] = {
        "chosen_on_early": show(rets, early[0], "2016-01-01", hi, "chosen on 2007-2015"),
        "chosen_on_full": show(rets, full[0], "2016-01-01", hi, "chosen on FULL (contaminated)"),
        "naive_equal": show(rets, naive_w, "2016-01-01", hi, "naive 25/25/25/25 (no choosing)"),
    }
    print("\n  Tested on 2007-2015 (unseen by the 'late' choice):")
    res["oos_early"] = {
        "chosen_on_late": show(rets, late[0], lo, SPLIT, "chosen on 2016-2026"),
        "chosen_on_full": show(rets, full[0], lo, SPLIT, "chosen on FULL (contaminated)"),
        "naive_equal": show(rets, naive_w, lo, SPLIT, "naive 25/25/25/25 (no choosing)"),
    }

    print("\n" + "=" * 88)
    print("  FULL PERIOD, all four allocations")
    print("=" * 88)
    res["full_period"] = {
        "full_choice": show(rets, full[0], lo, hi, "30/30/10/30 (reported)"),
        "early_choice": show(rets, early[0], lo, hi, "chosen on first half"),
        "late_choice": show(rets, late[0], lo, hi, "chosen on second half"),
        "naive": show(rets, naive_w, lo, hi, "naive 25/25/25/25"),
    }

    a = res["oos_late"]["chosen_on_early"]
    b = res["oos_late"]["chosen_on_full"]
    n = res["full_period"]["naive"]
    f = res["full_period"]["full_choice"]
    print("\n" + "=" * 88)
    print("  VERDICT")
    print("=" * 88)
    print(f"  selection cost on unseen data: CAGR {(a['cagr']-b['cagr'])*100:+.2f}pp, "
          f"MaxDD {(a['max_dd']-b['max_dd'])*100:+.2f}pp")
    print(f"  full-sample choice vs NAIVE equal weights: "
          f"CAGR {(f['cagr']-n['cagr'])*100:+.2f}pp, MaxDD {(f['max_dd']-n['max_dd'])*100:+.2f}pp")
    print("\n  If the naive book is close to the optimised one, the allocation was")
    print("  never the source of the result and the in-sample search bought little.")

    json.dump(res, open(os.path.join(REPORTS, "audit_allocation.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/audit_allocation.json")


if __name__ == "__main__":
    main()

"""
Toyota tests — remove a failure mode, add redundancy, tune nothing.
===================================================================
Reliability in a Toyota engine does not come from clever engineering. It comes
from fewer parts, proven parts, and tolerances that hold when conditions go bad.
Every clever addition tested in this project has failed; every simplification has
won. So these two tests REMOVE a known failure mode and ADD redundancy, and
neither fits a parameter to the data.

  R1  REBALANCE TOLERANCE BAND
      The 2008 audit found the annual rebalance bought the collapsing equity
      sleeve in Jan-2009, days before the March bottom — worth about -7.5pp.
      A band rebalances only when a sleeve drifts more than X% from target.
      HYPOTHESIS   A band improves CAGR and/or drawdown versus the calendar.
      FALSIFIED IF no band beats annual rebalancing on both.
      NOT a timing rule (Mandate §4 Group A): it never forecasts, never moves to
      cash, and reacts only to realised weight drift. It is a turnover tolerance.

  R2  MORE REDUNDANT SLEEVES, EQUAL WEIGHT
      A2 showed naive equal weights beat a searched allocation. If that is true,
      adding genuinely uncorrelated parts should help without any tuning.
      HYPOTHESIS   A 5- or 6-sleeve equal-weight book beats the 4-sleeve one.
      FALSIFIED IF extra sleeves do not improve Sharpe or drawdown.
      Candidates are screened on CORRELATION first — F7 was killed for adding
      silver, which is 79% correlated with gold and therefore not a part at all.

  MARK5_CACHE=data/pit_cache_2007 python3 scripts/reliability_test.py
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

# Candidate extra sleeves. Each must be buyable by an Indian investor (directly,
# or via LRS) and must have history back to 2007 or it cannot be tested here.
CANDIDATES = {
    "intl_dev": "EFA",     # developed ex-US
    "emerging": "EEM",     # emerging ex-India-heavy
    "commod": "DBC",       # broad commodities, not just gold
    "shortbond": "SHY",    # 1-3y treasuries, a dampener
}


def _nz(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated()]


def px(sym, tries=4, need=1000):
    last = 0
    for a in range(tries):
        try:
            h = yf.download(sym, start="2006-11-01", end="2026-07-22",
                            auto_adjust=True, progress=False)["Close"]
            if hasattr(h, "columns"):
                h = h.iloc[:, 0]
            h = h.dropna()
            last = len(h)
            if last >= need:
                return _nz(h)
        except Exception:
            pass
        time.sleep(3 * (a + 1))
    print(f"  SKIP {sym}: {last} rows after {tries} attempts")
    return None


def blend(rets, w, rebal=252, band=None):
    """Fixed-weight book. `band` (e.g. 0.20) rebalances only when a sleeve drifts
    more than that fraction from its target; `rebal` is the calendar fallback."""
    cols = [c for c in rets.columns if w.get(c, 0) > 0]
    wt = np.array([w[c] for c in cols], float)
    wt /= wt.sum()
    r = rets[cols].to_numpy()
    nav = np.empty(len(r))
    nav[0] = 1.0
    cur = wt.copy()
    n_rebal = 0
    for i in range(1, len(r)):
        cur = cur * (1 + r[i])
        tot = cur.sum()
        nav[i] = nav[i - 1] * tot
        cur /= tot
        if band is not None:
            if np.any(np.abs(cur - wt) > band * wt):
                cur = wt.copy()
                n_rebal += 1
        elif i % rebal == 0:
            cur = wt.copy()
            n_rebal += 1
    return pd.Series(nav, index=rets.index), n_rebal


def row(nav, label, extra=""):
    m = metrics(nav)
    print(f"  {label:<30}{m['cagr']*100:>+8.2f}%{m['sharpe']:>8.2f}"
          f"{m['vol']*100:>7.1f}%{m['max_dd']*100:>+9.2f}%{m['calmar']:>8.2f}  {extra}")
    return {k: m[k] for k in ("cagr", "sharpe", "vol", "max_dd", "calmar")}


def main():
    eq = _nz(pd.read_csv(EQ_CACHE, index_col=0, parse_dates=True).iloc[:, 0])
    fx = px("USDINR=X").reindex(eq.index).ffill().bfill()
    base_syms = {"gold": "GC=F", "us": "^NDX", "bond": "TLT"}
    sl = {"eq": eq}
    for k, s in base_syms.items():
        sl[k] = (px(s).reindex(eq.index).ffill() * fx).dropna()
    print("Fetching candidate sleeves...", flush=True)
    extra = {}
    for k, s in CANDIDATES.items():
        p = px(s)
        if p is not None:
            extra[k] = (p.reindex(eq.index).ffill() * fx).dropna()
    idx = eq.index
    for v in list(sl.values()) + list(extra.values()):
        idx = idx.intersection(v.index)
    allsl = {**sl, **extra}
    rets = pd.DataFrame({k: v[idx].pct_change(fill_method=None).fillna(0.0)
                         for k, v in allsl.items()})
    print(f"  aligned {idx[0].date()} -> {idx[-1].date()} ({len(idx)} bars)\n")
    res = {"window": [str(idx[0].date()), str(idx[-1].date())]}

    BASE4 = {k: 0.25 for k in ("eq", "gold", "us", "bond")}
    hdr = f"  {'book':<30}{'CAGR':>9}{'Sharpe':>8}{'vol':>7}{'MaxDD':>9}{'Calmar':>8}"

    print("=" * 92)
    print("  R1. REBALANCE TOLERANCE BAND vs CALENDAR")
    print("=" * 92)
    print(hdr)
    print("  " + "-" * 74)
    res["r1"] = {}
    nav, n = blend(rets, BASE4, rebal=252)
    res["r1"]["annual"] = row(nav, "annual calendar (current)", f"{n} rebalances")
    for b in (0.10, 0.20, 0.30, 0.50):
        nav, n = blend(rets, BASE4, band=b)
        res["r1"][f"band_{int(b*100)}"] = row(nav, f"band +/-{int(b*100)}% of target",
                                              f"{n} rebalances")
    a = res["r1"]["annual"]
    best = max((v for k, v in res["r1"].items() if k != "annual"),
               key=lambda v: v["cagr"])
    better = [k for k, v in res["r1"].items()
              if k != "annual" and v["cagr"] > a["cagr"] and v["max_dd"] > a["max_dd"]]
    r1 = (f"SUPPORTED — {', '.join(better)} beat annual on BOTH" if better
          else "FALSIFIED — no band beats the calendar on both CAGR and drawdown")
    print(f"\n  R1 VERDICT: {r1}")
    res["r1_verdict"] = r1

    print("\n" + "=" * 92)
    print("  R2. IS AN EXTRA SLEEVE A REAL PART? (correlation screen first)")
    print("=" * 92)
    print(f"  {'candidate':<12}" + "".join(f"{c:>11}" for c in ("eq", "gold", "us", "bond")))
    keep = []
    for k in extra:
        cs = [rets[k].corr(rets[c]) for c in ("eq", "gold", "us", "bond")]
        mx = max(abs(c) for c in cs)
        flag = "REDUNDANT" if mx > 0.75 else "ok"
        print(f"  {k:<12}" + "".join(f"{c*100:>10.0f}%" for c in cs) + f"   max |r| {mx*100:.0f}%  {flag}")
        if mx <= 0.75:
            keep.append(k)
    res["r2_screen"] = {"kept": keep, "dropped": [k for k in extra if k not in keep]}
    print(f"\n  passing the screen: {keep or 'none'}")

    print("\n" + "=" * 92)
    print("  R2. EQUAL-WEIGHT BOOKS — does redundancy help?")
    print("=" * 92)
    print(hdr)
    print("  " + "-" * 74)
    res["r2"] = {}
    res["r2"]["4_sleeve"] = row(blend(rets, BASE4)[0], "4 sleeves (current)")
    for k in keep:
        names = list(BASE4) + [k]
        w = {n_: 1 / len(names) for n_ in names}
        res["r2"][f"5_with_{k}"] = row(blend(rets, w)[0], f"5 sleeves (+{k})")
    if len(keep) >= 2:
        names = list(BASE4) + keep
        w = {n_: 1 / len(names) for n_ in names}
        res["r2"][f"{len(names)}_sleeve_all"] = row(blend(rets, w)[0],
                                                    f"{len(names)} sleeves (all)")
    b4 = res["r2"]["4_sleeve"]
    wins = [k for k, v in res["r2"].items()
            if k != "4_sleeve" and (v["sharpe"] > b4["sharpe"] or v["max_dd"] > b4["max_dd"])]
    r2 = (f"SUPPORTED — {', '.join(wins)} improve Sharpe or drawdown" if wins
          else "FALSIFIED — extra sleeves add nothing")
    print(f"\n  R2 VERDICT: {r2}")
    res["r2_verdict"] = r2

    json.dump(res, open(os.path.join(REPORTS, "reliability_test.json"), "w"),
              indent=2, default=float)
    print("\n  Saved -> reports/reliability_test.json")


if __name__ == "__main__":
    main()

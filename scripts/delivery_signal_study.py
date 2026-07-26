"""
Does NSE delivery data contain alpha this system does not already have?
=======================================================================
THE BAR IS SET HERE, BEFORE ANY RESULT IS SEEN, so it cannot be moved afterwards.
A delivery-derived signal is worth pursuing only if it clears ALL THREE:

  1. PREDICTIVE   |IC| >= 0.02 with |t| >= 2.0 at a horizon we actually trade
                  (63 or 126 bars). Context: K7's institutional-flow signal was
                  killed at IC = -0.025, and the Delta-promoter signal at
                  IC ~ +0.034 was judged "too weak alone". This bar is
                  deliberately at that same level, not below it.

  2. ORTHOGONAL   |corr| < 0.30 against every existing factor (momentum, low_vol,
                  trend, stability). Grinold: IR = IC x sqrt(breadth) only pays
                  for INDEPENDENT information. A signal that merely restates
                  momentum adds cost, not edge — and every price-derived factor
                  this project has tested lives in the same span, which is the
                  structural reason they failed.

  3. MONOTONIC    the tercile spread must order correctly (top > mid > bottom).
                  A signal that "works" only via its extreme tail is a noise
                  artefact, not a factor.

Failing any one of these is a KILL, recorded as such. That is the expected
outcome given K1-K12, and a cheap falsification is the point.

Signals tested (all strictly causal — value at date t uses only data through t):
  deliv_per_z      delivery % vs its own 126d history. High = conviction
                   ownership rather than intraday churn.
  deliv_chg        21d mean delivery % minus 126d mean. RISING conviction.
  deliv_turn_z     delivered rupee value vs own history — real money committed.
  trade_size_z     average trade size (turnover / no_of_trades) vs own history.
                   Large average trades proxy institutional participation.

  MARK5_CACHE=data/pit_cache python3 scripts/delivery_signal_study.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as ss

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.portfolio import DataPanel, discover_tickers
from core.portfolio.factors import FactorLibrary

RAW = os.path.join(_ROOT, "data", "delivery", "raw")
END = "2026-07-21"
HORIZONS = [21, 63, 126]
IC_BAR, T_BAR, ORTHO_BAR = 0.02, 2.0, 0.30


def load_delivery():
    """-> wide frames {field -> DataFrame(date x symbol)} from the daily archive."""
    files = sorted(glob.glob(os.path.join(RAW, "*.parquet")))
    if not files:
        sys.exit("ERROR: no delivery archive — run scripts/fetch_delivery.py first.")
    dp, dq, nt, tv = {}, {}, {}, {}
    for f in files:
        d = pd.Timestamp(os.path.basename(f)[:10])
        df = pd.read_parquet(f)
        df = df[df["symbol"].notna()].drop_duplicates("symbol").set_index("symbol")
        dp[d] = df["deliv_per"]
        dq[d] = df.get("deliv_qty")
        nt[d] = df.get("no_of_trades")
        tv[d] = df.get("turnover_lacs")
    mk = lambda x: pd.DataFrame(x).T.sort_index()
    return mk(dp), mk(dq), mk(nt), mk(tv)


def zscore_own(df, win=126):
    """Each column z-scored against ITS OWN trailing window — strictly causal,
    and it removes the permanent cross-sectional level difference between a
    illiquid small-cap and a large-cap so the signal measures CHANGE in
    conviction rather than which sector a name is in."""
    m = df.rolling(win, min_periods=win // 2).mean()
    s = df.rolling(win, min_periods=win // 2).std()
    return (df - m) / s.replace(0, np.nan)


def main():
    print("  loading delivery archive...", flush=True)
    dp, dq, nt, tv = load_delivery()
    print(f"    {dp.shape[0]} days x {dp.shape[1]} symbols  "
          f"({dp.index.min().date()} -> {dp.index.max().date()})", flush=True)

    panel = DataPanel(discover_tickers(), END, freshness="off")
    close = panel.close
    cal = dp.index.intersection(close.index)
    syms = [s for s in dp.columns if s in close.columns]
    print(f"    overlap with price panel: {len(cal)} days x {len(syms)} symbols\n", flush=True)

    dp, dq, nt, tv = (x.reindex(index=cal, columns=syms) for x in (dp, dq, nt, tv))
    px = close.reindex(index=cal, columns=syms)

    trade_size = (tv * 1e5) / nt.replace(0, np.nan)
    deliv_turn = dq * px

    signals = {
        "deliv_per_z": zscore_own(dp),
        "deliv_chg": (dp.rolling(21, min_periods=10).mean()
                      - dp.rolling(126, min_periods=63).mean()),
        "deliv_turn_z": zscore_own(deliv_turn),
        "trade_size_z": zscore_own(trade_size),
    }

    # existing price factors, on the same grid, for the orthogonality test
    fac = {f: pd.DataFrame({t: FactorLibrary.compute_all(close[t])[f] for t in syms})
           .reindex(index=cal, columns=syms) for f in FactorLibrary.DEFAULT_FACTORS}

    # liquid names only — the deployed universe reaches 300 deep, and a signal
    # measured on untradeable microcaps is not a signal we could ever use
    liq = panel.turnover.reindex(index=cal, columns=syms)
    rank_liq = liq.rank(axis=1, ascending=False)
    tradeable = rank_liq <= 300

    print("=" * 96)
    print("  INFORMATION COEFFICIENT   (Spearman rank corr of signal_t vs forward return)")
    print("=" * 96)
    print(f"  bar set in advance: |IC| >= {IC_BAR}  AND  |t| >= {T_BAR}")
    print(f"  {'signal':<16}{'horizon':>9}{'IC':>9}{'t-stat':>9}{'n dates':>9}"
          f"{'IC>0 %':>9}   verdict")
    print("  " + "-" * 92)

    results, passed = [], []
    for name, sig in signals.items():
        for h in HORIZONS:
            fwd = px.shift(-h) / px - 1.0
            ics = []
            for d in cal[::5]:                       # every 5th day: overlapping
                s = sig.loc[d].where(tradeable.loc[d])
                r = fwd.loc[d]
                ok = s.notna() & r.notna()
                if ok.sum() < 50:
                    continue
                ics.append(ss.spearmanr(s[ok], r[ok]).correlation)
            ics = np.array([x for x in ics if np.isfinite(x)])
            if len(ics) < 20:
                continue
            ic = float(ics.mean())
            # overlapping forward windows autocorrelate the IC series, so the
            # naive t-stat is inflated. Newey-West style haircut: divide the
            # effective sample by the overlap factor (h / 5-day step).
            n_eff = max(1.0, len(ics) / max(1.0, h / 5.0))
            t = ic / (ics.std(ddof=1) / np.sqrt(n_eff)) if ics.std(ddof=1) > 0 else 0.0
            ok = abs(ic) >= IC_BAR and abs(t) >= T_BAR
            if ok:
                passed.append((name, h, ic, t))
            print(f"  {name:<16}{h:>9}{ic:>+9.4f}{t:>+9.2f}{len(ics):>9}"
                  f"{(ics > 0).mean()*100:>8.0f}%   {'PASS' if ok else 'fail'}")
            results.append({"signal": name, "horizon": h, "ic": ic, "t": float(t),
                            "n_dates": len(ics), "n_eff": n_eff,
                            "pct_positive": float((ics > 0).mean() * 100), "pass": bool(ok)})

    print("\n" + "=" * 96)
    print("  ORTHOGONALITY   (mean cross-sectional corr vs existing factors)")
    print("=" * 96)
    print(f"  bar: |corr| < {ORTHO_BAR} against ALL of them")
    print(f"  {'signal':<16}" + "".join(f"{f:>14}" for f in fac) + "   verdict")
    print("  " + "-" * 92)
    ortho = {}
    for name, sig in signals.items():
        row, worst = [], 0.0
        for f, fdf in fac.items():
            cs = []
            for d in cal[::10]:
                a, b = sig.loc[d], fdf.loc[d]
                ok = a.notna() & b.notna()
                if ok.sum() > 50:
                    cs.append(a[ok].corr(b[ok], method="spearman"))
            c = float(np.nanmean(cs)) if cs else np.nan
            row.append(c)
            worst = max(worst, abs(c) if np.isfinite(c) else 0.0)
        ortho[name] = dict(zip(fac.keys(), row))
        print(f"  {name:<16}" + "".join(f"{c:>+14.3f}" for c in row)
              + f"   {'PASS' if worst < ORTHO_BAR else 'fail'}")

    print("\n" + "=" * 96)
    print("  TERCILE SPREAD   (126-bar forward return by signal tercile; must be monotonic)")
    print("=" * 96)
    print(f"  {'signal':<16}{'bottom':>11}{'middle':>11}{'top':>11}"
          f"{'top-bottom':>13}   monotonic")
    print("  " + "-" * 92)
    terc = {}
    fwd = px.shift(-126) / px - 1.0
    for name, sig in signals.items():
        buckets = [[], [], []]
        for d in cal[::10]:
            s = sig.loc[d].where(tradeable.loc[d])
            r = fwd.loc[d]
            ok = s.notna() & r.notna()
            if ok.sum() < 60:
                continue
            s, r = s[ok], r[ok]
            q = pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop")
            for i in range(3):
                if (q == i).any():
                    buckets[i].append(r[q == i].mean())
        m = [float(np.nanmean(b)) * 100 if b else np.nan for b in buckets]
        mono = (m[0] < m[1] < m[2]) or (m[0] > m[1] > m[2])
        terc[name] = m
        print(f"  {name:<16}{m[0]:>+10.2f}%{m[1]:>+10.2f}%{m[2]:>+10.2f}%"
              f"{m[2]-m[0]:>+12.2f}pp   {'yes' if mono else 'NO'}")

    print("\n" + "=" * 96)
    print("  VERDICT")
    print("=" * 96)
    if not passed:
        print(f"  NO signal cleared the predictive bar (|IC| >= {IC_BAR}, |t| >= {T_BAR}).")
        print("  KILL. Delivery data does not carry usable stock-selection alpha for this")
        print("  book at the horizons it trades. This matches K7 (institutional flow,")
        print("  IC -0.025) and K12 — weak-IC public data does not survive costs and tax.")
    else:
        print(f"  {len(passed)} signal/horizon combination(s) cleared the predictive bar:")
        for n, h, ic, t in passed:
            w = max(abs(v) for v in ortho[n].values() if np.isfinite(v))
            print(f"    {n} @ {h}d: IC {ic:+.4f} (t {t:+.2f}), max factor corr {w:.3f}"
                  f" -> {'ORTHOGONAL' if w < ORTHO_BAR else 'NOT orthogonal, adds nothing'}")
        print("\n  Next step is NOT deployment. A passing IC earns only a walk-forward")
        print("  test as a small factor component, on the ~4 windows the short archive")
        print("  allows, against the same >=6/8-equivalent consistency bar.")

    out = os.path.join(_ROOT, "reports", "delivery_signal_study.json")
    json.dump({"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
               "archive": {"days": int(dp.shape[0]), "symbols": int(len(syms)),
                           "start": str(cal.min().date()), "end": str(cal.max().date())},
               "bars": {"ic": IC_BAR, "t": T_BAR, "ortho": ORTHO_BAR},
               "ic": results, "orthogonality": ortho, "terciles": terc,
               "passed": [{"signal": n, "horizon": h, "ic": ic, "t": t}
                          for n, h, ic, t in passed]},
              open(out, "w"), indent=1, default=float)
    print(f"\n  saved -> {out}\n")


if __name__ == "__main__":
    main()

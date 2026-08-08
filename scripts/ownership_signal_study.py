"""
Ownership-Signal Study (DEEP): does institutional ACCUMULATION predict returns?
================================================================================
Tests Kavin's "big investor moves the stock" thesis on DEEP, FREE, OFFICIAL data:
NSE corporate-filings shareholding XBRL, ~32 quarters back to mid-2018 (vs the old
screener.in free tier's ~12 quarters). This window COVERS the 2019-2024 HAL / BEL /
TRENT multibagger runs, so we can finally check the thesis on the actual winners.

Signal at each filing = QoQ change in holding. Tested separately for:
  - Institutions (FII+DII total)  <- the robust "big investor" signal
  - FIIs, DIIs, Promoters         <- decomposed
Forward returns measured from the REAL public-disclosure date (NSE broadcastDate),
so there is ZERO look-ahead — we only ever act on what was public.

Metrics: Spearman IC vs forward 1q/2q/1y returns, tercile spreads, and a winner
case study (did institutions pile in BEFORE the big runs, or chase after?).

Data: scripts/fetch_shareholding_nse.py -> data/cache/shareholding_nse/
Fallback: data/cache/shareholding/ (screener, 12q) if the deep dir is absent.
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.portfolio.universe import load_ohlcv

DEEP = os.path.join(_ROOT, "data", "cache", "shareholding_nse")
SCREENER = os.path.join(_ROOT, "data", "cache", "shareholding")
QEND = {"Mar": "03-31", "Jun": "06-30", "Sep": "09-30", "Dec": "12-31"}
DISCLOSURE_LAG = 45      # fallback only (screener schema has no real disclosure date)
HORIZONS = {"1q": 63, "2q": 126, "1y": 252}
SIGNALS = ["Institutions", "FIIs", "DIIs", "Promoters"]


def qlabel_to_disclosure(lbl):
    mon, yr = lbl.split()
    return pd.Timestamp(f"{yr}-{QEND[mon]}") + pd.Timedelta(days=DISCLOSURE_LAG)


def load_ownership(src):
    """ticker -> DataFrame indexed by REAL disclosure date, cols = SIGNALS present."""
    out = {}
    for f in glob.glob(os.path.join(src, "*.json")):
        t = os.path.basename(f).replace(".json", "")
        d = json.load(open(f))
        qs = d.get("quarters", [])
        if len(qs) < 5:
            continue
        # disclosure dates: deep schema has them; else derive from quarter label
        if d.get("disclosure"):
            idx = pd.to_datetime(d["disclosure"])
        else:
            idx = pd.to_datetime([qlabel_to_disclosure(q) for q in qs])
        cols = {}
        for s in SIGNALS:
            if s in d and len(d[s]) == len(qs):
                cols[s] = pd.Series(d[s], index=idx, dtype="float64")
        if "Institutions" not in cols and "FIIs" in cols and "DIIs" in cols:
            cols["Institutions"] = cols["FIIs"] + cols["DIIs"]
        if not cols:
            continue
        df = pd.DataFrame(cols).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if len(df) >= 5:
            out[t] = df
    return out


def collect(own):
    """Pool (signal_change, forward_return) across all tickers/quarters/horizons."""
    rows = {s: {h: {"sig": [], "fwd": []} for h in HORIZONS} for s in SIGNALS}
    for t, df in own.items():
        px = load_ohlcv(t)
        if px is None:
            continue
        close = px["close"].astype(float)
        chg = df.diff()
        for i in range(1, len(df)):
            d = df.index[i]
            ps = close.loc[:d]
            if len(ps) == 0:
                continue
            p0 = ps.iloc[-1]
            for h, bars in HORIZONS.items():
                pf = close.loc[d:d + pd.Timedelta(days=int(bars * 1.45))]
                if len(pf) <= bars * 0.6:
                    continue
                fwd = pf.iloc[-1] / p0 - 1
                for s in SIGNALS:
                    if s in chg.columns and np.isfinite(chg[s].iloc[i]):
                        rows[s][h]["sig"].append(chg[s].iloc[i])
                        rows[s][h]["fwd"].append(fwd)
    return rows


def report_ic(rows):
    print("=" * 78)
    print("  INFORMATION COEFFICIENT — Δ holding (accumulation) vs forward return")
    print("  Spearman rank corr. |IC|<0.03 = noise | >0.05 = weak | >0.10 = useful")
    print("=" * 78)
    summary = {}
    for s in SIGNALS:
        line = f"  {s:13s}"
        for h in HORIZONS:
            df = pd.DataFrame(rows[s][h]).dropna()
            if len(df) < 40:
                line += f"  {h}: n/a"
                continue
            ic = df["sig"].corr(df["fwd"], method="spearman")
            summary[(s, h)] = (ic, len(df))
            line += f"  {h}:{ic:+.3f}(n={len(df)})"
        print(line)
    return summary


def report_terciles(rows, signal="Institutions", horizon="1y"):
    print("\n" + "=" * 78)
    print(f"  TERCILE SPREAD — forward {horizon} return by Δ{signal} (accumulators vs sellers)")
    print("=" * 78)
    df = pd.DataFrame(rows[signal][horizon]).dropna()
    if len(df) < 60:
        print(f"  insufficient data (n={len(df)})")
        return
    # rank first so ties (many Δ=0) don't collapse bin edges
    df["b"] = pd.qcut(df["sig"].rank(method="first"), 3, labels=["selling", "neutral", "buying"])
    m = df.groupby("b")["fwd"].agg(["mean", "median", "count"])
    for b in m.index:
        print(f"  {b:9s}: mean {m.loc[b,'mean']*100:+6.1f}%   median {m.loc[b,'median']*100:+6.1f}%   n={int(m.loc[b,'count'])}")
    spread = (m.loc["buying", "mean"] - m.loc["selling", "mean"]) * 100 if {"buying", "selling"} <= set(m.index) else float("nan")
    print(f"  buying − selling spread: {spread:+.1f} pp  "
          f"({'edge exists' if abs(spread) > 5 else 'no usable edge'})")


def report_winners(own):
    print("\n" + "=" * 78)
    print("  WINNER CASE STUDY — did institutions accumulate BEFORE the run, or chase?")
    print("  (Δ institutions over the 4 quarters PRECEDING each name's best 1y move)")
    print("=" * 78)
    rows = []
    for t, df in own.items():
        if "Institutions" not in df.columns or len(df) < 8:
            continue
        px = load_ohlcv(t)
        if px is None:
            continue
        close = px["close"].astype(float)
        best_ret, best_d = -9, None
        for d in df.index:
            pf = close.loc[d:d + pd.Timedelta(days=370)]
            ps = close.loc[:d]
            if len(pf) > 150 and len(ps):
                r = pf.iloc[-1] / ps.iloc[-1] - 1
                if r > best_ret:
                    best_ret, best_d = r, d
        if best_d is None:
            continue
        prior = df["Institutions"].loc[:best_d]
        if len(prior) < 5:
            continue
        delta_before = prior.iloc[-1] - prior.iloc[max(0, len(prior) - 5)]
        rows.append((t, best_ret, delta_before, prior.iloc[-1]))
    wdf = pd.DataFrame(rows, columns=["ticker", "best_1y", "inst_chg_before", "inst_at_run"]).dropna()
    if wdf.empty:
        print("  no data")
        return
    top = wdf.sort_values("best_1y", ascending=False).head(15)
    print(f"  {'ticker':12s}{'best 1y':>9}{'Δinst before(pp)':>18}{'inst% at run':>14}")
    for _, r in top.iterrows():
        print(f"  {r.ticker:12s}{r.best_1y*100:>+8.0f}%{r.inst_chg_before:>+17.1f}{r.inst_at_run:>13.1f}")
    # correlation across the universe: did pre-run accumulation predict run size?
    ic = wdf["inst_chg_before"].corr(wdf["best_1y"], method="spearman")
    pos = (wdf["inst_chg_before"] > 0).mean() * 100
    print(f"\n  Across {len(wdf)} names: corr(pre-run Δinst, run size) = {ic:+.3f}")
    print(f"  Winners where institutions were NET BUYING in the prior year: {pos:.0f}%")
    print("  (If institutions don't accumulate before the run -> they chase, can't be front-run)")


def main():
    src = DEEP if (os.path.isdir(DEEP) and glob.glob(os.path.join(DEEP, "*.json"))) else SCREENER
    own = load_ownership(src)
    print(f"Source: {os.path.relpath(src, _ROOT)}")
    print(f"Stocks with usable ownership history: {len(own)}")
    if len(own) < 15:
        print("Insufficient data — run scripts/fetch_shareholding_nse.py first.")
        return
    spans = [len(df) for df in own.values()]
    print(f"Median quarters/stock: {int(np.median(spans))}  (range {min(spans)}-{max(spans)})\n")
    rows = collect(own)
    summary = report_ic(rows)
    report_terciles(rows, "Institutions", "1y")
    report_terciles(rows, "FIIs", "1y")
    report_winners(own)

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    ic_inst = summary.get(("Institutions", "1y"), (0, 0))[0]
    if abs(ic_inst) < 0.05:
        print(f"  Institutional-accumulation IC (1y) = {ic_inst:+.3f}: NO usable edge even on")
        print("  deep data covering the multibagger runs. The thesis is real but UNEXPLOITABLE")
        print("  with public filings — by disclosure it is priced in. Confirms MARK6 verdict.")
    else:
        print(f"  Institutional-accumulation IC (1y) = {ic_inst:+.3f}: a signal worth integrating")
        print("  into MARK6 ranking. Validate with a proper walk-forward backtest next.")


if __name__ == "__main__":
    main()

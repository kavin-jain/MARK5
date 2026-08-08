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


# ══════════════════════════════════════════════════════════════════════════
#  B1 — the test the original K7 verdict never ran
#
#  Two things are wrong with everything above, and they push in the same
#  direction (toward a flattering number that is then read as "no edge"):
#
#  1. collect() POOLS every ticker and every date into one list and takes a
#     single Spearman over the pool. That is not an information coefficient.
#     IC in the Fundamental Law is CROSS-SECTIONAL — computed within a date,
#     then averaged across dates. Pooling mixes the market's own drift into
#     the statistic: in a quarter where everything rose, every name has a
#     positive forward return regardless of who was accumulating it.
#
#  2. It scores RAW IC. Mandate §4 Group B: "each was judged on raw IC, never
#     on IC orthogonal to momentum. A weak signal correlated with a strong one
#     adds no information and dilutes the strong one. The method was wrong, so
#     some of these verdicts may be wrong."
#
#  HYPOTHESIS   Institutional accumulation carries information the existing
#               momentum composite does not already contain.
#  FALSIFIED IF mean cross-sectional residual IC < 0.03, or its t-stat < 3.0
#               (Harvey, Liu & Zhu 2016 — the profession has already tried
#               thousands of factors, so 2.0 is not a hurdle any more).
#
#  Both bars are pre-registered here, before the fetch finished, and both must
#  clear. A pass on IC with t < 3.0 is a pass on noise.
# ══════════════════════════════════════════════════════════════════════════
ORTHO_IC_BAR = 0.03
ORTHO_T_BAR = 3.0


def _spearman(a, b):
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 20:
        return np.nan
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def _residualise(sig, base):
    """sig with the part explained by `base` projected out (OLS residual).
    What survives is information the existing composite does NOT already hold."""
    d = pd.concat([sig, base], axis=1).dropna()
    d.columns = ["s", "b"]
    if len(d) < 20 or d["b"].std() == 0:
        return pd.Series(dtype=float)
    return d["s"] - (d["s"].cov(d["b"]) / d["b"].var()) * d["b"]


def report_orthogonal_ic(own, signal="Institutions", fwd_bars=126):
    """Cross-sectional IC of the ownership signal, raw and residual-to-momentum."""
    import sys
    sys.path.insert(0, _ROOT)
    from core.portfolio import (DataPanel, discover_tickers, load_sector_map)
    from core.portfolio.factors import FactorLibrary, composite_score
    from core.portfolio import (PortfolioConstructor, ConstructionConfig,
                                Backtester, BacktestConfig)

    print("\n" + "=" * 78)
    print(f"  B1. CROSS-SECTIONAL IC — Δ{signal}, raw AND orthogonal to momentum")
    print(f"  falsified if residual IC < {ORTHO_IC_BAR} or t < {ORTHO_T_BAR}")
    print("=" * 78)

    end = os.environ.get("MARK5_END", "2026-07-21")
    panel = DataPanel(discover_tickers(), end)
    base_w = {"momentum": 0.45, "low_vol": 0.15, "trend": 0.25, "stability": 0.15}
    bt = Backtester(panel, PortfolioConstructor(
        ConstructionConfig(mode="factor_tilt", n_hold=20, factor_weights=base_w),
        sector_map=load_sector_map()),
        BacktestConfig(rebal_bars=fwd_bars, top_n_liquid=300))
    close = panel.close

    # signal as a per-ticker step series of QoQ change, read as-of each date so
    # only what was actually disclosed by then is ever used
    chg = {t: df[signal].diff().dropna() for t, df in own.items() if signal in df.columns}
    if not chg:
        print(f"  no {signal} data")
        return None

    all_disc = sorted({d for s in chg.values() for d in s.index})
    if len(all_disc) < 8:
        print(f"  only {len(all_disc)} disclosure dates — cannot form a panel")
        return None
    # evaluate on a quarterly grid spanning the disclosure history
    grid = pd.DatetimeIndex(all_disc)
    cal = panel.trading_calendar(str(grid[0].date()), end)
    dates = [d for d in pd.DatetimeIndex(cal) if d >= grid[0]][::63]

    raw, res, corr, base_ics, ns = [], [], [], [], []
    for d in dates:
        fi = close.index.searchsorted(d)
        if fi + fwd_bars >= len(close.index):
            continue
        elig = [t for t in panel.eligible(d, 252, 0.0, top_n=300) if t in close.columns]
        if len(elig) < 50:
            continue
        sig = pd.Series({t: chg[t].loc[:d].iloc[-1] for t in elig
                         if t in chg and len(chg[t].loc[:d])}).dropna()
        if len(sig) < 30:
            continue
        fwd = (close.iloc[fi + fwd_bars] / close.iloc[fi] - 1).reindex(elig).dropna()
        panels = {f: pd.Series({t: bt._factors[t].loc[:d].iloc[-1].get(f, np.nan)
                                for t in elig if not bt._factors[t].loc[:d].empty})
                  for f in FactorLibrary.DEFAULT_FACTORS}
        base = composite_score(panels, base_w, rank_transform=True)
        base_ics.append(_spearman(base, fwd))
        raw.append(_spearman(sig, fwd))
        corr.append(_spearman(sig, base.reindex(sig.index)))
        r = _residualise(sig, base.reindex(sig.index))
        if len(r):
            res.append(_spearman(r, fwd))
        ns.append(len(sig))

    if len(res) < 5:
        print(f"  only {len(res)} usable cross-sections — not enough to conclude")
        return None

    def stat(v):
        v = np.array([x for x in v if np.isfinite(x)])
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        return m, (m / se if se else np.nan), len(v)

    bm, bt_, _ = stat(base_ics)
    rm, rt, _ = stat(raw)
    om, ot, n = stat(res)
    cm = np.nanmean(corr)

    print(f"  {len(dates)} candidate dates -> {n} usable cross-sections, "
          f"median {int(np.median(ns))} names each\n")
    print(f"  {'':28}{'mean IC':>10}{'t-stat':>9}")
    print(f"  {'existing momentum composite':28}{bm:>+10.4f}{bt_:>9.2f}")
    print(f"  {'Δ' + signal + ' (raw)':28}{rm:>+10.4f}{rt:>9.2f}")
    print(f"  {'Δ' + signal + ' (residual)':28}{om:>+10.4f}{ot:>9.2f}")
    print(f"\n  correlation of the signal to the existing composite: {cm:+.3f}")

    ok = abs(om) >= ORTHO_IC_BAR and abs(ot) >= ORTHO_T_BAR
    verdict = (f"SUPPORTED — Δ{signal} carries residual IC {om:+.4f} (t={ot:.2f}) "
               f"that momentum does not already own"
               if ok else
               f"FALSIFIED — residual IC {om:+.4f} (t={ot:.2f}) is below the "
               f"pre-registered bar ({ORTHO_IC_BAR}, t {ORTHO_T_BAR})")
    print(f"\n  B1 VERDICT: {verdict}")
    return {"base_ic": bm, "raw_ic": rm, "residual_ic": om, "t_residual": ot,
            "corr_to_base": cm, "n_cross_sections": n, "verdict": verdict,
            "bars": {"ic": ORTHO_IC_BAR, "t": ORTHO_T_BAR}}


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
    ortho = report_orthogonal_ic(own, "Institutions")
    if ortho:
        os.makedirs(os.path.join(_ROOT, "reports"), exist_ok=True)
        json.dump(ortho, open(os.path.join(_ROOT, "reports",
                  "ownership_orthogonal_ic.json"), "w"), indent=1, default=float)

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

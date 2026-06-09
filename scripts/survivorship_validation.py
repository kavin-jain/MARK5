"""
MARK5 — Survivorship-Bias Validation of Equal-Weight Buy-and-Hold
=================================================================
Pins the HONEST forward return of "own an equal-weight midcap basket and hold",
correcting for the survivorship bias in using today's constituents on old data.

A perfect survivorship-free test needs point-in-time index membership + prices
for delisted names (not in free data). So we do the best rigorous approximation
AND bound the residual bias four ways:

  1. POINT-IN-TIME MEMBERSHIP — at each annual rebalance only hold names that
     (a) had >=252 trading days of history before that date (no IPO look-ahead),
     (b) cleared a liquidity floor (trailing 126d median turnover) AS OF that date.
     -> removes the worst look-ahead (holding today's names from day one).
  2. FAILURE INJECTION — add k synthetic constituents/yr that go to -85% (the
     delisted/blown-up names absent from our survivor data), at NSE-plausible
     base rates 0/2/4/6% per year. -> bounds the bias we cannot remove.
  3. DROP-THE-WINNERS — recompute after removing the top 10/20/30% contributors.
     -> shows dependence on a few hindsight winners.
  4. EXTERNAL ANCHOR — the Nifty Midcap 50 index is survivorship-free by
     construction. The single cleanest reality check.

All returns GROSS (for apples-to-apples with the price indices); known tax+cost
drag (~2-3pp/yr for a low-turnover annual-rebalance hold) noted at the end.
RESEARCH ONLY.
"""
import os, sys, glob, re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_ROOT, "data", "cache")
LARGE = {'ASIANPAINT','BAJFINANCE','BEL','BHARTIARTL','COFORGE','HDFCBANK','HINDUNILVR',
         'ICICIBANK','IDEA','IDFCFIRSTB','INFY','ITC','KOTAKBANK','LT','LUPIN','MARUTI',
         'MOTHERSON','PERSISTENT','PNB','RELIANCE','SBIN','SUNPHARMA','TATAELXSI',
         'TATASTEEL','TCS','TITAN','TRENT','VOLTAS','YESBANK','AUBANK'}


def load_close_vol(t):
    for s in ["_daily.parquet", "_NS_1d.parquet"]:
        p = os.path.join(CACHE, f"{t}{s}")
        if os.path.exists(p):
            df = pd.read_parquet(p); df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            df.index = pd.to_datetime(df.index)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index(); df = df[~df.index.duplicated(keep="last")]
            if "close" in df.columns and "volume" in df.columns:
                return df["close"].astype(float), df["volume"].astype(float)
    return None, None


def discover_universe():
    names = set()
    for f in glob.glob(os.path.join(CACHE, "*.parquet")):
        b = os.path.basename(f).replace(".parquet", "")
        if b.startswith("sector_") or "NIFTY" in b.upper() or "index" in b.lower() or "60m" in b:
            continue
        names.add(re.sub(r"(_NS_1d|_daily|_NS|_1d)$", "", b))
    return sorted(names)


def index_cagr(fname, start, end):
    p = os.path.join(CACHE, fname)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p); df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df.index = pd.to_datetime(df["date"])
    s = df["close"].astype(float).sort_index().loc[start:end]
    if len(s) < 50:
        return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1


def annual_periods(cal):
    """Yield (t0, t1) annual sub-period index slices over the calendar."""
    yrs = sorted(set(cal.year))
    bounds = []
    for y in yrs:
        ys = cal[cal.year == y]
        if len(ys):
            bounds.append(ys[0])
    bounds.append(cal[-1])
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def build(start, end, universe):
    pcol, vcol = {}, {}
    for t in universe:
        c, v = load_close_vol(t)
        if c is None:
            continue
        c = c.loc[:end]
        if len(c.loc[start:end]) < 60:
            continue
        pcol[t] = c; vcol[t] = v
    prices = pd.DataFrame(pcol)
    cal = prices.loc[start:end].index
    full = prices.reindex(prices.index.union(cal)).sort_index()
    turn = (prices * pd.DataFrame(vcol)).rolling(126, min_periods=40).median()
    return prices, full, turn, cal


def eligible_at(t0, prices, turn, min_hist=252, liq_pct=0.40):
    """Point-in-time eligibility: seasoned (>=min_hist bars before t0) + liquid as-of t0."""
    elig = []
    tvals = {}
    for t in prices.columns:
        s = prices[t].loc[:t0].dropna()
        if len(s) < min_hist:
            continue
        tv = turn[t].loc[:t0].dropna()
        if len(tv) == 0:
            continue
        tvals[t] = tv.iloc[-1]
        elig.append(t)
    if not elig:
        return []
    # liquidity floor = liq_pct quantile of eligible names' turnover as-of t0
    floor = np.nanquantile([tvals[t] for t in elig], liq_pct)
    return [t for t in elig if tvals[t] >= floor]


def ew_period_return(t0, t1, prices, names):
    """Equal-weight (annual-rebalanced, intra-year drift) GROSS return over [t0,t1].
    ffill interior gaps; hold names that have a valid price at both ends of the
    window (i.e. were actually trading at t0 and t1)."""
    cols = [n for n in names if n in prices.columns]
    if not cols:
        return 0.0, {}
    sub = prices.loc[t0:t1, cols].ffill()
    valid = [c for c in sub.columns if pd.notna(sub[c].iloc[0]) and pd.notna(sub[c].iloc[-1])]
    if not valid:
        return 0.0, {}
    sub = sub[valid]
    rel = sub / sub.iloc[0]                     # each name's growth, start=1
    port = rel.mean(axis=1)                     # equal-weight, buy-and-hold within year
    contrib = {t: float(rel[t].iloc[-1] - 1) for t in valid}
    return float(port.iloc[-1] - 1), contrib


def run(start, end, universe, label, fail_rates=(0.0, 0.02, 0.04, 0.06), ghost=-0.85):
    prices, full, turn, cal = build(start, end, universe)
    periods = annual_periods(cal)
    yrs = (cal[-1] - cal[0]).days / 365.25

    # base point-in-time chain + per-name aggregate contribution
    chain, agg_contrib, sizes = 1.0, {}, []
    period_rets = []
    for (t0, t1) in periods:
        names = eligible_at(t0, prices, turn)
        r, contrib = ew_period_return(t0, t1, prices, names)
        chain *= (1 + r); period_rets.append((t0, t1, r, len(contrib)))
        sizes.append(len(contrib))
        for k, v in contrib.items():
            agg_contrib[k] = agg_contrib.get(k, 0.0) + v
    base_cagr = chain ** (1 / yrs) - 1
    avg_n = int(np.mean(sizes))

    def cagr_with_ghosts(fr):
        ch = 1.0
        for (t0, t1, r, n) in period_rets:
            k = int(round(fr * n))
            blended = (n * r + k * ghost) / (n + k) if (n + k) else r
            ch *= (1 + blended)
        return ch ** (1 / yrs) - 1

    def cagr_drop_top(pct):
        drop = set(sorted(agg_contrib, key=agg_contrib.get, reverse=True)[:int(len(agg_contrib) * pct)])
        ch = 1.0
        for (t0, t1) in periods:
            names = [t for t in eligible_at(t0, prices, turn) if t not in drop]
            r, _ = ew_period_return(t0, t1, prices, names)
            ch *= (1 + r)
        return ch ** (1 / yrs) - 1

    mid = index_cagr("MIDCAP50_index.parquet", start, end)
    crs = index_cagr("CRSMID_index.parquet", start, end)

    print(f"\n{'='*78}\n  {label}   (point-in-time EW-B&H, avg {avg_n} eligible names, GROSS)\n{'='*78}")
    print(f"  {'Survivor-universe EW-B&H (naive)':<44}{'(see prior runs)':>20}")
    print(f"  {'POINT-IN-TIME EW-B&H (existence+liquidity)':<44}{base_cagr*100:>+18.1f}%")
    print(f"  {'-'*72}")
    print(f"  Failure injection (synthetic delisted names @ {int(ghost*-100)}% loss):")
    for fr in fail_rates:
        print(f"     {'fail rate '+format(fr,'.0%')+'/yr':<41}{cagr_with_ghosts(fr)*100:>+18.1f}%")
    print(f"  {'-'*72}")
    print(f"  Drop top contributors (hindsight-winner sensitivity):")
    for pct in (0.10, 0.20, 0.30):
        print(f"     {'drop top '+format(pct,'.0%'):<41}{cagr_drop_top(pct)*100:>+18.1f}%")
    print(f"  {'-'*72}")
    print(f"  SURVIVORSHIP-FREE ANCHORS (price indices):")
    if mid is not None: print(f"     {'Nifty Midcap 50 index':<41}{mid*100:>+18.1f}%")
    if crs is not None: print(f"     {'CRISIL/Nifty midcap (CRSMID)':<41}{crs*100:>+18.1f}%")
    print(f"     {'Nifty 50 index (your old benchmark)':<41}{index_cagr('nse/NIFTY50_20150101_20260521.parquet', start, end)*100 if index_cagr('nse/NIFTY50_20150101_20260521.parquet', start, end) else float('nan'):>+18.1f}%")
    return base_cagr, mid


if __name__ == "__main__":
    uni = sorted(set(discover_universe()) | LARGE)
    print(f"Discovered universe: {len(uni)} names with local data")
    run("2016-01-01", "2021-12-31", uni, "HONEST OOS — 2016-2021 (UNTOUCHED)")
    run("2022-01-01", "2026-05-21", uni, "RECENT — 2022-2026")
    print(f"\n{'='*78}")
    print("  Net-of-tax note: annual-rebalance EW hold has low turnover; LTCG-dominated")
    print("  tax + costs shave ~2-3pp/yr. Subtract that from the gross figures above.")
    print('='*78)

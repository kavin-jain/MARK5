"""
MARK5 — Factor-Tilt vs Buy-and-Hold (HONEST OOS, local data)
============================================================
Answers ONE question with the data already on disk:

  Does a clean momentum factor-tilt (hold top-N by momentum score,
  rebalance quarterly, NO stops / cooldowns / circuit-breakers)
  beat equal-weight buy-and-hold of the SAME universe, OUT-OF-SAMPLE?

The honest benchmark is equal-weight buy-and-hold (not the NIFTY).
All strategies net of transaction costs. Momentum score reuses the
repo's existing MomentumSignalEngine (causal, no lookahead).

Windows:
  2016-2021  = genuinely UNTOUCHED (never tuned on) -> the real test
  2022-2026  = tuned window (shown for contrast only)

PAPER / RESEARCH ONLY.
"""
import os, sys, glob, re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.models.momentum_signal import MomentumSignalEngine

CACHE = os.path.join(_ROOT, "data", "cache")
COST = 0.0029          # round-trip commission
SLIP = 0.001           # one-way slippage (applied each side)
REBAL_BARS = 63        # ~quarterly
WARMUP = "2015-01-01"  # extra history for score warmup before any window


def load(t):
    for s in ["_daily.parquet", "_NS_1d.parquet"]:
        p = os.path.join(CACHE, f"{t}{s}")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            df.index = pd.to_datetime(df.index)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            return df.sort_index()[~df.sort_index().index.duplicated(keep="last")]
    return None


def metrics(nav: pd.Series):
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    ret = nav.pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    return cagr, dd, sharpe


def simulate(prices, rets, scores, dates, top_n, regime_ok=None, equal_weight_all=False):
    """Walk-forward quarterly. Returns daily NAV series (gross of tax)."""
    tickers = list(prices.columns)
    w = pd.Series(0.0, index=tickers)
    nav = 1.0
    out = {}
    last_rebal = -10**9
    for i, d in enumerate(dates):
        # drift weights with the day's returns
        if i > 0:
            r = rets.loc[d].fillna(0.0)
            port_r = float((w * r).sum())
            nav *= (1 + port_r)
            denom = (1 + port_r)
            if denom != 0:
                w = w * (1 + r) / denom
        # rebalance?
        if (i - last_rebal) >= REBAL_BARS:
            last_rebal = i
            avail = [t for t in tickers if not np.isnan(prices.loc[d, t]) and not np.isnan(scores.loc[d, t])]
            if equal_weight_all:
                target = pd.Series(0.0, index=tickers)
                if avail:
                    target[avail] = 1.0 / len(avail)
            else:
                go = True if regime_ok is None else bool(regime_ok.get(d, True))
                target = pd.Series(0.0, index=tickers)
                if go and avail:
                    ranked = scores.loc[d, avail].sort_values(ascending=False)
                    picks = list(ranked.index[:top_n])
                    target[picks] = 1.0 / len(picks)
                # else: all-cash this quarter
            turnover = float((target - w).abs().sum()) / 2.0
            nav *= (1 - turnover * (COST + 2 * SLIP))
            w = target
        out[d] = nav
    return pd.Series(out)


def run_window(start, end, universe, nifty_df, label):
    engine = MomentumSignalEngine()
    price_cols, score_cols = {}, {}
    for t in universe:
        df = load(t)
        if df is None:
            continue
        df = df.loc[:end]
        if len(df.loc[start:end]) < 60:
            continue
        sc = engine.precompute_scores(df, nifty_df=nifty_df)
        price_cols[t] = df["close"].astype(float)
        score_cols[t] = sc
    prices = pd.DataFrame(price_cols)
    scores = pd.DataFrame(score_cols).reindex(prices.index)
    # calendar = window slice on the union index
    cal = prices.loc[start:end].index
    prices = prices.reindex(cal).ffill(limit=5)
    scores = scores.reindex(cal).ffill(limit=5)
    rets = prices.pct_change()

    # regime: NIFTY > 200d SMA on each date
    regime_ok = None
    if nifty_df is not None:
        nclose = nifty_df["close"].astype(float).sort_index()
        sma200 = nclose.rolling(200, min_periods=60).mean()
        bull = (nclose > sma200).reindex(cal).ffill()
        regime_ok = {d: bool(bull.get(d, True)) for d in cal}

    n_active = prices.notna().any().sum()
    results = {}
    results["EW Buy&Hold (BAR)"] = simulate(prices, rets, scores, cal, None, equal_weight_all=True)
    for n in (5, 10):
        results[f"Momentum top-{n} (always-in)"] = simulate(prices, rets, scores, cal, n)
        results[f"Momentum top-{n} (regime-cash)"] = simulate(prices, rets, scores, cal, n, regime_ok=regime_ok)

    # NIFTY B&H
    nb = nifty_df["close"].astype(float).loc[start:end] if nifty_df is not None else None

    print(f"\n{'='*82}\n  {label}   (universe: {n_active} names, quarterly rebalance, net of costs)\n{'='*82}")
    print(f"  {'Strategy':<32}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>9}   vs BAR")
    print(f"  {'-'*78}")
    bar_cagr = metrics(results['EW Buy&Hold (BAR)'])[0]
    for name, nav in results.items():
        c, dd, sh = metrics(nav)
        delta = (c - bar_cagr) * 100
        flag = "" if name.startswith("EW Buy") else (f"{delta:+.1f}pp" if delta < 0 else f"**{delta:+.1f}pp**")
        print(f"  {name:<32}{c*100:>+8.1f}%{dd*100:>+8.1f}%{sh:>9.2f}   {flag}")
    if nb is not None and len(nb) > 2:
        yrs = (nb.index[-1] - nb.index[0]).days / 365.25
        ncagr = (nb.iloc[-1] / nb.iloc[0]) ** (1 / yrs) - 1
        print(f"  {'-'*78}")
        print(f"  {'NIFTY50 (passive index)':<32}{ncagr*100:>+8.1f}%{'':>9}{'':>9}   {(ncagr-bar_cagr)*100:+.1f}pp")
    return results


if __name__ == "__main__":
    universe = ['ASIANPAINT','BAJFINANCE','BEL','BHARTIARTL','COFORGE','HDFCBANK',
                'HINDUNILVR','ICICIBANK','IDEA','IDFCFIRSTB','INFY','ITC','KOTAKBANK',
                'LT','LUPIN','MARUTI','MOTHERSON','PERSISTENT','PNB','RELIANCE','SBIN',
                'SUNPHARMA','TATAELXSI','TATASTEEL','TCS','TITAN','TRENT','VOLTAS','YESBANK']
    nifty = None
    for p in [os.path.join(CACHE, "nse", "NIFTY50_20150101_20260521.parquet"),
              os.path.join(CACHE, "sector_NSEI.parquet")]:
        if os.path.exists(p):
            nifty = pd.read_parquet(p); nifty.columns = [c.lower() for c in nifty.columns]
            if "date" in nifty.columns:
                nifty.index = pd.to_datetime(nifty["date"])
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.sort_index()
            break
    run_window("2016-01-01", "2021-12-31", universe, nifty, "HONEST OOS — 2016-2021 (UNTOUCHED)")
    run_window("2022-01-01", "2026-05-21", universe, nifty, "TUNED WINDOW — 2022-2026 (contrast)")

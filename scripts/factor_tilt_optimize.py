"""
MARK5 — Factor-Tilt Optimization (NET of tax, holdable version)
===============================================================
Goal: keep the momentum factor-tilt's excess return over buy-and-hold
while cutting drawdown to something you can actually stay invested through.

Honest method:
  - Lot-tracked, TAX-AWARE NAV (LTCG 12.5% >365d, STCG 20% <=365d) applied
    on realized gains + terminal liquidation, for BOTH the tilt AND the
    buy-and-hold bar -> apples-to-apples NET comparison.
  - No stops / cooldowns / circuit-breakers / regime gate (all proven to
    destroy the edge). Always fully invested.
  - Sweep: top-N in {5,7,10,12} x weighting {equal, inverse-vol}.
  - Validate on 2016-2021 (UNTOUCHED), cross-check on 2022-2026.
  - Reuses the repo's MomentumSignalEngine (causal, no lookahead).

PAPER / RESEARCH ONLY.
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from core.models.momentum_signal import MomentumSignalEngine

CACHE = os.path.join(_ROOT, "data", "cache")
COST, SLIP = 0.0029, 0.001          # round-trip commission + one-way slippage each side
REBAL_BARS = 63                      # ~quarterly
LTCG, STCG, LTCG_DAYS = 0.125, 0.20, 365


def load(t):
    for s in ["_daily.parquet", "_NS_1d.parquet"]:
        p = os.path.join(CACHE, f"{t}{s}")
        if os.path.exists(p):
            df = pd.read_parquet(p); df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            df.index = pd.to_datetime(df.index)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            return df[~df.index.duplicated(keep="last")]
    return None


def metrics(nav):
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    ret = nav.pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    calmar = cagr / abs(dd) if dd != 0 else 0.0
    return cagr, dd, sharpe, calmar


def simulate(prices, rets, scores, vol, dates, day_dates, top_n=None, weighting="equal",
             buy_hold=False, rebal_bars=REBAL_BARS, exit_rank=None):
    """
    Lot-tracked, tax-aware NAV. Returns (nav_series, turnover_per_yr, tax_drag_pp).
    Per-ticker average-cost lot; holding period via value-weighted entry date.
    """
    tickers = list(prices.columns)
    pos = {t: 0.0 for t in tickers}        # market value of holding
    basis = {t: 0.0 for t in tickers}      # cost basis of holding
    entry_day = {t: None for t in tickers} # value-weighted entry calendar date
    cash, tax_paid, traded_val = 1.0, 0.0, 0.0
    nav_out = {}
    last_rebal = -10**9
    n_days = len(dates)

    def realize(t, sell_val, today):
        """Realize gain on selling `sell_val` of ticker t; deduct tax+cost; return net cash added."""
        nonlocal tax_paid, traded_val
        if pos[t] <= 0:
            return 0.0
        frac = min(1.0, sell_val / pos[t])
        gain = sell_val - basis[t] * frac
        held_days = (today - entry_day[t]).days if entry_day[t] is not None else 0
        rate = LTCG if held_days > LTCG_DAYS else STCG
        tax = max(0.0, gain) * rate
        tax_paid += tax
        cost = sell_val * (COST / 2 + SLIP)
        traded_val += sell_val
        pos[t] -= sell_val
        basis[t] -= basis[t] * frac
        if pos[t] < 1e-12:
            pos[t], basis[t], entry_day[t] = 0.0, 0.0, None
        return sell_val - tax - cost

    def buy(t, buy_val, today):
        nonlocal cash, traded_val
        cost = buy_val * (COST / 2 + SLIP)
        cash -= (buy_val + cost)
        traded_val += buy_val
        # value-weighted entry date
        if entry_day[t] is None or pos[t] <= 0:
            entry_day[t] = today
        else:
            wold = pos[t] / (pos[t] + buy_val)
            entry_day[t] = pd.Timestamp(int(wold * entry_day[t].value + (1 - wold) * today.value))
        pos[t] += buy_val
        basis[t] += buy_val

    for i, d in enumerate(dates):
        today = day_dates[i]
        if i > 0:
            r = rets.loc[d]
            for t in tickers:
                if pos[t] > 0 and not np.isnan(r[t]):
                    pos[t] *= (1 + r[t])
        nav = cash + sum(pos.values())

        do_rebal = (i - last_rebal) >= rebal_bars
        if buy_hold:
            do_rebal = (i == 0)            # buy once, then hold
        if do_rebal:
            last_rebal = i
            avail = [t for t in tickers
                     if not np.isnan(prices.loc[d, t]) and not np.isnan(scores.loc[d, t])]
            if buy_hold:
                target = {t: nav / len(avail) for t in avail} if avail else {}
            else:
                ranked = scores.loc[d, avail].sort_values(ascending=False)
                if exit_rank is not None:
                    # Buffered momentum: keep held names while they stay within
                    # top `exit_rank`, only swap out those that fell past it.
                    # Cuts turnover -> more holds cross 365d LTCG line.
                    rank_of = {t: r for r, t in enumerate(ranked.index)}
                    held = [t for t in tickers if pos[t] > 0]
                    keep = [t for t in held if rank_of.get(t, 10**9) < exit_rank][:top_n]
                    adds = [t for t in ranked.index if t not in keep][:max(0, top_n - len(keep))]
                    picks = (keep + adds)[:top_n]
                else:
                    picks = list(ranked.index[:top_n])
                if weighting == "invvol":
                    iv = {t: 1.0 / max(vol.loc[d, t], 1e-4) for t in picks
                          if not np.isnan(vol.loc[d, t])}
                    s = sum(iv.values()) or 1.0
                    target = {t: nav * iv.get(t, 0.0) / s for t in picks}
                else:
                    target = {t: nav / len(picks) for t in picks} if picks else {}
            # sells first (free up cash), then buys
            for t in tickers:
                tgt = target.get(t, 0.0)
                if pos[t] > tgt + 1e-9:
                    cash += realize(t, pos[t] - tgt, today)
            for t, tgt in target.items():
                if tgt > pos[t] + 1e-9:
                    buy(t, tgt - pos[t], today)
        nav_out[d] = cash + sum(pos.values())

    # terminal liquidation tax (both strategies pay it -> fair vs buy&hold)
    last = day_dates[-1]
    term_tax = 0.0
    for t in tickers:
        if pos[t] > 0:
            gain = pos[t] - basis[t]
            held = (last - entry_day[t]).days if entry_day[t] else 0
            term_tax += max(0.0, gain) * (LTCG if held > LTCG_DAYS else STCG)
    nav_net_final = (cash + sum(pos.values())) - term_tax
    ser = pd.Series(nav_out)
    ser.iloc[-1] = nav_net_final
    yrs = (day_dates[-1] - day_dates[0]).days / 365.25
    turnover_yr = traded_val / yrs / ser.mean()     # gross traded / yr vs avg NAV
    return ser, turnover_yr


def build(universe, nifty_df, start, end):
    engine = MomentumSignalEngine()
    pcol, scol = {}, {}
    for t in universe:
        df = load(t)
        if df is None:
            continue
        df = df.loc[:end]
        if len(df.loc[start:end]) < 60:
            continue
        pcol[t] = df["close"].astype(float)
        scol[t] = engine.precompute_scores(df, nifty_df=nifty_df)
    prices = pd.DataFrame(pcol)
    scores = pd.DataFrame(scol).reindex(prices.index)
    cal = prices.loc[start:end].index
    prices = prices.reindex(cal).ffill(limit=5)
    scores = scores.reindex(cal).ffill(limit=5)
    rets = prices.pct_change()
    vol = rets.rolling(63, min_periods=20).std()
    return prices, rets, scores, vol, cal


def run(start, end, universe, nifty_df, label):
    prices, rets, scores, vol, cal = build(universe, nifty_df, start, end)
    dd = list(cal)
    Q, A = 63, 252   # quarterly, annual
    cfgs = [
        ("Buy&Hold EW (BAR)",        dict(buy_hold=True)),
        # quarterly, no buffer (high turnover baseline)
        ("top-10 Q equal",           dict(top_n=10, weighting="equal", rebal_bars=Q)),
        # annual rebalance (lets holds cross 365d LTCG line)
        ("top-10 ANNUAL equal",      dict(top_n=10, weighting="equal",  rebal_bars=A)),
        ("top-10 ANNUAL inv-vol",    dict(top_n=10, weighting="invvol", rebal_bars=A)),
        ("top-12 ANNUAL inv-vol",    dict(top_n=12, weighting="invvol", rebal_bars=A)),
        # annual + buffer (lowest turnover)
        ("top-10 ANN+buf equal",     dict(top_n=10, weighting="equal",  rebal_bars=A, exit_rank=20)),
        ("top-10 ANN+buf inv-vol",   dict(top_n=10, weighting="invvol", rebal_bars=A, exit_rank=20)),
        ("top-12 ANN+buf inv-vol",   dict(top_n=12, weighting="invvol", rebal_bars=A, exit_rank=24)),
        # quarterly + buffer (buffer alone, keeps responsiveness)
        ("top-10 Q+buf inv-vol",     dict(top_n=10, weighting="invvol", rebal_bars=Q, exit_rank=20)),
    ]

    print(f"\n{'='*88}\n  {label}   ({prices.notna().any().sum()} names, quarterly, NET of tax)\n{'='*88}")
    print(f"  {'Config':<20}{'NET CAGR':>10}{'MaxDD':>9}{'Sharpe':>8}{'Calmar':>8}{'Turn/yr':>9}   vs BAR")
    print(f"  {'-'*84}")
    bar_cagr = None
    rows = []
    for name, kw in cfgs:
        nav, turn = simulate(prices, rets, scores, vol, dd, list(cal), **kw)
        c, mdd, sh, cal_r = metrics(nav)
        if bar_cagr is None:
            bar_cagr = c
        delta = (c - bar_cagr) * 100
        flag = "" if name.startswith("Buy&Hold") else (f"**{delta:+.1f}pp**" if delta > 0 else f"{delta:+.1f}pp")
        print(f"  {name:<20}{c*100:>+9.1f}%{mdd*100:>+8.1f}%{sh:>8.2f}{cal_r:>8.2f}{turn:>8.0%}   {flag}")
        rows.append((name, c, mdd, sh, cal_r, delta))
    return rows


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
            nifty.index = pd.to_datetime(nifty.index); nifty = nifty.sort_index()
            break
    r_oos = run("2016-01-01", "2021-12-31", universe, nifty, "HONEST OOS — 2016-2021 (UNTOUCHED)")
    r_tun = run("2022-01-01", "2026-05-21", universe, nifty, "TUNED WINDOW — 2022-2026 (cross-check)")

    # consistency: configs that beat BAR in BOTH windows
    print(f"\n{'='*88}\n  ROBUST CONFIGS — beat Buy&Hold NET in BOTH windows\n{'='*88}")
    tun = {n: d for (n, *_ , d) in r_tun}
    for (name, c, mdd, sh, cal_r, delta) in r_oos:
        if name.startswith("Buy&Hold"):
            continue
        td = tun.get(name, -99)
        if delta > 0 and td > 0:
            print(f"  {name:<20} OOS {delta:+.1f}pp | tuned {td:+.1f}pp   (OOS: Sharpe {sh:.2f}, Calmar {cal_r:.2f}, MaxDD {mdd*100:.0f}%)")

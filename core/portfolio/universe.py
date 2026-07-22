"""
MARK6 — Point-in-Time Universe & Data Layer
===========================================
Loads cached OHLCV and selects an investable universe AS OF a given date using
only information available then — the structural defence against survivorship and
look-ahead bias.

Eligibility at date T:
  1. Seasoned    : >= `min_history` trading days of data strictly before T
                   (no IPO-pop look-ahead; the name was actually listed & tradable)
  2. Liquid      : trailing-126d median turnover (close*volume) as-of T is at or
                   above the `liquidity_pct` quantile of all seasoned names
  3. Priced      : has a real (non-stale) print within `max_stale_days` of T

This makes universe membership a point-in-time decision that naturally adds names
as they list/grow liquid. HONEST LIMIT: the CANDIDATE list is whatever the local
cache holds — in practice today's surviving index constituents — so fully-delisted
names are absent and headline returns carry residual survivorship bias (estimated
~1-2pp/yr; see README and scripts/survivorship_validation.py, which bounds it via
failure injection on the equal-weight basket). The backtester adds a stale-print
force-exit so suspended names cannot silently compound at 0%.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# MARK5_CACHE lets the whole engine run against an alternative price cache without
# code changes — used to compare the survivor-biased yfinance cache (data/cache)
# against the true point-in-time bhavcopy cache (data/pit_cache, built by
# scripts/build_pit_cache.py). Relative paths resolve from the repo root.
CACHE = os.environ.get("MARK5_CACHE") or os.path.join(_ROOT, "data", "cache")
if not os.path.isabs(CACHE):
    CACHE = os.path.join(_ROOT, CACHE)
# Benchmark series (Nifty) is NOT part of the investable universe, so it must not
# move when MARK5_CACHE swaps the universe — otherwise switching to the PIT cache
# silently drops the benchmark and every "vs Nifty" figure becomes n/a.
BENCH_CACHE = os.path.join(_ROOT, "data", "cache")

# Names that are structurally inappropriate for an equity-quality basket
# (a-priori exclusions, NOT performance-based — documented to avoid snooping).
STRUCTURAL_EXCLUDE = {
    "YESBANK",   # RBI-administered bailout, permanent capital impairment
    "IDEA",      # Vodafone-Idea: going-concern / AGR overhang, perennial dilution
    # ETFs cached for multi-asset tests — NOT equity single names, must never enter
    # the equity selection (LIQUIDBEES is ~cash: lowest vol -> inverse-vol would
    # massively overweight it and corrupt the book).
    "GOLDBEES", "LIQUIDBEES", "NIFTYBEES", "BANKBEES", "JUNIORBEES",
}


def _is_etf(name: str) -> bool:
    return name.upper().endswith("BEES") or name.upper().endswith("ETF")


def _norm(name: str) -> str:
    return re.sub(r"(_NS_1d|_daily|_NS|_1d)$", "", name)


def load_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Load one instrument's OHLCV (lowercase cols, tz-naive DatetimeIndex).

    Falls back to the default cache: the multi-asset sleeve ETFs (GOLDBEES, MON100)
    and benchmark series are deliberately NOT in the equity PIT universe, but every
    wrapper still needs them, so they must survive a MARK5_CACHE swap.
    """
    for suffix in ("_daily.parquet", "_NS_1d.parquet"):
        for root in dict.fromkeys((CACHE, BENCH_CACHE)):
            path = os.path.join(root, f"{ticker}{suffix}")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                raise RuntimeError(f"Corrupt cache file {path}: {e}. "
                                   f"Delete it and re-run scripts/refetch_all.py.") from e
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            df.index = pd.to_datetime(df.index)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if "close" in df.columns:
                keep = despike(df["close"].astype(float)).index
                dropped = len(df) - len(keep)
                if dropped:
                    print(f"  data hygiene: dropped {dropped} corrupt print(s) from {ticker}")
                df = df.loc[keep]
            return df
    return None


def discover_tickers() -> list[str]:
    """All cached single-name instruments (excludes indices/sector series).
    Falls back to the version-pinned list in config/universe_tickers.json when
    the cache is empty, so a fresh clone knows what to fetch (reproducibility)."""
    names = set()
    for f in glob.glob(os.path.join(CACHE, "*.parquet")):
        b = os.path.basename(f).replace(".parquet", "")
        low = b.lower()
        if b.startswith("sector_") or "nifty" in low or "index" in low or "60m" in low:
            continue
        nm = _norm(b)
        if _is_etf(nm) or nm in ("block_deals", "bulk_deals"):
            continue
        names.add(nm)
    if not names:
        pinned = os.path.join(_ROOT, "config", "universe_tickers.json")
        if os.path.exists(pinned):
            import json
            with open(pinned) as f:
                names = set(json.load(f)["tickers"])
    return sorted(names - STRUCTURAL_EXCLUDE)


NIFTY_DIV_YIELD = 0.013   # long-run Nifty 50 dividend yield used to approximate TRI
                          # when no true total-return series is cached (~1.2-1.4%/yr
                          # historically; NSE publishes the exact TRI but with no
                          # free machine-readable history).


def despike(s: pd.Series, tol: float = 0.5, window: int = 11) -> pd.Series:
    """Drop single-day price prints that are absurdly far from their local median.

    yfinance mis-applies some corporate actions, leaving a one-day crash that fully
    reverses: GOLDBEES 2019-12-19 printed 0.34 against a 33.60 neighbourhood (-99%,
    recovered next session); MON100 2021-06-17 printed 10.09 against 101.56 (-90%).
    These are not market moves. Left in, each one injects a fake crash-and-recover
    into any portfolio holding the series - which INFLATES volatility and drawdown
    and therefore UNDERSTATES Sharpe. Gold + US are half the deployed book, so the
    effect was material.

    A `tol` of 0.5 against an 11-day centred median only fires on data errors: no
    real instrument sits 50% away from its own two-week median for a single print
    and then returns.
    """
    if len(s) < window:
        return s
    med = s.rolling(window, center=True, min_periods=3).median()
    ok = (s / med - 1).abs() <= tol
    return s[ok.fillna(True)]


def _first_existing(fname: str) -> str | None:
    """Look in the active cache, then the default one — benchmark files live with
    the primary cache and are shared by every universe."""
    for d in (CACHE, BENCH_CACHE):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None


def load_nifty(total_return: bool = True) -> pd.Series | None:
    """Nifty 50 benchmark series (close). The strategy book runs on dividend-
    adjusted (total-return) stock prices, so a fair benchmark must be total-return
    too. Prefers a real TRI series (data/cache/NIFTY_TRI.parquet) if present;
    otherwise approximates TRI = price index compounded by NIFTY_DIV_YIELD.
    total_return=False returns the raw price index (unfair vs this book —
    kept only for explicit price-index comparisons)."""
    if total_return:
        tri = _first_existing("NIFTY_TRI.parquet")
        if tri is not None:
            df = pd.read_parquet(tri)
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            s = df["close"].astype(float).sort_index()
            # drop corporate-action mis-adjustments (e.g. the Dec-2019 NIFTYBEES
            # 1:10 split glitch in yfinance): an index ETF cannot really print
            # 25% away from its local median for a day or two
            return despike(s, tol=0.25)
    path = _first_existing("sector_NSEI.parquet")
    if path is None:
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df.index = pd.to_datetime(df["date"])
    s = df["close"].astype(float).sort_index()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    if total_return:
        yrs = (s.index - s.index[0]).days / 365.25
        s = s * (1.0 + NIFTY_DIV_YIELD) ** yrs
    return s


class DataPanel:
    """Aligned close/volume/turnover matrices for a set of tickers."""

    # A name whose cache ends > this many calendar days before the panel's true end
    # is STALE: it silently loses price-eligibility and drops out of the point-in-time
    # universe near the end of the window, distorting recent-window results
    # (2026-06 audit: 137/345 files frozen ~2 months back corrupted the 2026 numbers).
    STALENESS_TOLERANCE_DAYS = 7

    def __init__(self, tickers: list[str], end: str, *, freshness: str = "warn"):
        """freshness: 'warn' (default) prints stale names, 'raise' aborts, 'off' skips."""
        closes, vols = {}, {}
        last_dates = {}
        for t in tickers:
            df = load_ohlcv(t)
            if df is None or "close" not in df.columns or "volume" not in df.columns:
                continue
            df = df.loc[:end]
            if len(df) < 60:
                continue
            closes[t] = df["close"].astype(float)
            vols[t] = df["volume"].astype(float)
            last_dates[t] = df.index[-1]
        if not closes:
            raise RuntimeError(
                f"No cached price data found for any of {len(tickers)} tickers in {CACHE}. "
                f"Run scripts/refetch_all.py to build the cache from the pinned universe "
                f"(config/universe_tickers.json). Knowing the ticker NAMES is not the same "
                f"as having their DATA.")
        self.close = pd.DataFrame(closes).sort_index()
        self.volume = pd.DataFrame(vols).reindex_like(self.close)
        self.turnover = (self.close * self.volume).rolling(126, min_periods=40).median()
        self.tickers = list(self.close.columns)
        self.stale_tickers = self._check_freshness(last_dates, freshness)

    def _check_freshness(self, last_dates: dict, mode: str) -> list[str]:
        if mode == "off" or not last_dates:
            return []
        panel_end = max(last_dates.values())
        cutoff = panel_end - pd.Timedelta(days=self.STALENESS_TOLERANCE_DAYS)
        stale = sorted(t for t, d in last_dates.items() if d < cutoff)
        if stale:
            msg = (f"DATA FRESHNESS: {len(stale)}/{len(last_dates)} tickers end >"
                   f"{self.STALENESS_TOLERANCE_DAYS}d before panel end {panel_end.date()} "
                   f"(e.g. {stale[:5]}). Recent-window results are NOT trustworthy — "
                   f"run scripts/refetch_all.py.")
            if mode == "raise":
                raise RuntimeError(msg)
            print(f"  ⚠ {msg}")
        return stale

    def trading_calendar(self, start: str, end: str) -> pd.DatetimeIndex:
        """Common calendar = union of trading days, restricted to [start, end]."""
        return self.close.loc[start:end].index

    def eligible(self, asof: pd.Timestamp, min_history: int = 252,
                 liquidity_pct: float = 0.40, max_stale_days: int = 14,
                 min_turnover: float = 0.0, top_n: int = 0) -> list[str]:
        """Point-in-time investable universe as-of `asof`.

        min_turnover (absolute rupee 126d-median daily turnover) is the preferred
        liquidity control: `liquidity_pct` is a percentile of whatever is cached, so
        it silently tightens as the universe grows and cannot express a real capacity
        constraint. When min_turnover > 0 it is applied INSTEAD of the percentile.
        """
        seasoned, turn_now = [], {}
        for t in self.tickers:
            hist = self.close[t].loc[:asof].dropna()
            if len(hist) < min_history:
                continue
            # Priced: must have a real print near `asof` — a name that stopped
            # trading (suspension/delisting) is not investable, whatever ffill says.
            if (asof - hist.index[-1]).days > max_stale_days:
                continue
            tv = self.turnover[t].loc[:asof].dropna()
            if len(tv) == 0:
                continue
            seasoned.append(t)
            turn_now[t] = float(tv.iloc[-1])
        if not seasoned:
            return []
        # top_n is the preferred screen: it is TIME-INVARIANT (adapts as the market's
        # rupee turnover grows) and capacity-meaningful (you know exactly how deep
        # into the liquidity ranking you are reaching). It mirrors NSE's own Nifty 500
        # rule, which ranks by turnover rather than using a fixed rupee threshold.
        # A fixed rupee floor is NOT time-invariant: Rs 20cr/day admitted 0 names in
        # 2016 but 436 in 2026, so it silently disables itself early in a backtest.
        ranked = sorted(seasoned, key=lambda t: turn_now[t], reverse=True)
        if top_n and top_n > 0:
            ranked = ranked[:top_n]
        if min_turnover > 0:
            keep = [t for t in ranked if turn_now[t] >= min_turnover]
            # Falling back to the FULL seasoned list here would swap a strict screen
            # for no screen at all — the opposite of intent, and invisible in results.
            # Degrade to the most-liquid decile of the ranked set instead.
            return keep if keep else ranked[:max(1, len(ranked) // 10)]
        if top_n and top_n > 0:
            return ranked
        floor = np.nanquantile(list(turn_now.values()), liquidity_pct)
        return [t for t in seasoned if turn_now[t] >= floor]

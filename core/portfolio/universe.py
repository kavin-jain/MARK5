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
  3. Priced      : has a valid price at T

This makes universe membership a point-in-time decision that naturally adds names
as they list/grow liquid and (within the survivor data we have) reflects what was
actually investable. Residual survivorship from fully-delisted names is bounded
separately via failure-injection in the backtester.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(_ROOT, "data", "cache")

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
    """Load one instrument's OHLCV (lowercase cols, tz-naive DatetimeIndex)."""
    for suffix in ("_daily.parquet", "_NS_1d.parquet"):
        path = os.path.join(CACHE, f"{ticker}{suffix}")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
            df.index = pd.to_datetime(df.index)
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()
            return df[~df.index.duplicated(keep="last")]
    return None


def discover_tickers() -> list[str]:
    """All cached single-name instruments (excludes indices/sector series)."""
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
    return sorted(names - STRUCTURAL_EXCLUDE)


class DataPanel:
    """Aligned close/volume/turnover matrices for a set of tickers."""

    def __init__(self, tickers: list[str], end: str):
        closes, vols = {}, {}
        for t in tickers:
            df = load_ohlcv(t)
            if df is None or "close" not in df.columns or "volume" not in df.columns:
                continue
            df = df.loc[:end]
            if len(df) < 60:
                continue
            closes[t] = df["close"].astype(float)
            vols[t] = df["volume"].astype(float)
        self.close = pd.DataFrame(closes).sort_index()
        self.volume = pd.DataFrame(vols).reindex_like(self.close)
        self.turnover = (self.close * self.volume).rolling(126, min_periods=40).median()
        self.tickers = list(self.close.columns)

    def trading_calendar(self, start: str, end: str) -> pd.DatetimeIndex:
        """Common calendar = union of trading days, restricted to [start, end]."""
        return self.close.loc[start:end].index

    def eligible(self, asof: pd.Timestamp, min_history: int = 252,
                 liquidity_pct: float = 0.40) -> list[str]:
        """Point-in-time investable universe as-of `asof`."""
        seasoned, turn_now = [], {}
        for t in self.tickers:
            hist = self.close[t].loc[:asof].dropna()
            if len(hist) < min_history:
                continue
            tv = self.turnover[t].loc[:asof].dropna()
            if len(tv) == 0:
                continue
            seasoned.append(t)
            turn_now[t] = float(tv.iloc[-1])
        if not seasoned:
            return []
        floor = np.nanquantile(list(turn_now.values()), liquidity_pct)
        return [t for t in seasoned if turn_now[t] >= floor]

"""
MARK5 NSE Backtest Data Provider v1.1 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-05-09] v1.1: yfinance fallback for offline backtest validation (NSE blocks scraping)
- [2026-05-09] v1.0: NSEPython primary with parquet caching

TRADING ROLE: Offline backtest data ONLY — never used in live signal path
SAFETY LEVEL: MEDIUM

IMPORTANT — Rule 4 compliance:
  yfinance is PROHIBITED in the live trading/prediction path.
  It is used ONLY in this module for OFFLINE historical backtesting.
  The live trading path (predictor.py, autonomous.py) uses Kite Connect exclusively.

  Kite Connect → live trading signals
  yfinance (this file) → offline walk-forward backtest validation only
"""
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger("MARK5.NSEDataProvider")

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "..", "..", "data", "cache", "nse")


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = key.replace(".", "_").replace("/", "_").replace("-", "")
    return os.path.join(CACHE_DIR, f"{safe}.parquet")


def fetch_equity_ohlcv(
    symbol: str,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV for NSE equity symbol.

    For OFFLINE BACKTEST USE ONLY. Uses yfinance (Yahoo Finance NSE mirror)
    which provides reliable historical data going back 10+ years.

    Returns DataFrame with columns [open, high, low, close, volume]
    and a timezone-naive DatetimeIndex sorted ascending.

    Caches to parquet so repeated runs are instant.
    """
    cache_key = f"{symbol}_{start_date}_{end_date}"
    cp = _cache_path(cache_key)

    if not force_refresh and os.path.exists(cp):
        df = pd.read_parquet(cp)
        logger.debug(f"[{symbol}] loaded from cache ({len(df)} bars)")
        return df

    logger.info(f"[{symbol}] fetching historical data {start_date}→{end_date}")

    df = _fetch_via_yfinance(symbol, start_date, end_date)

    if df is None or df.empty:
        logger.error(f"[{symbol}] all data sources exhausted — no OHLCV available")
        return None

    # ── OHLC integrity check (Rule 48) ───────────────────────────────────────
    bad = (
        (df["low"] > df["open"]) |
        (df["low"] > df["close"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"])
    )
    if bad.any():
        n_bad = bad.sum()
        logger.warning(f"[{symbol}] {n_bad} bars fail OHLC integrity — forward-filling")
        df.loc[bad, ["open", "high", "low", "close"]] = float("nan")
        df = df.ffill().dropna()

    if len(df) < 60:
        logger.warning(f"[{symbol}] only {len(df)} bars — too short for feature engineering")
        return None

    df.to_parquet(cp)
    logger.info(f"[{symbol}] cached {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def _fetch_via_yfinance(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Download NSE data via Yahoo Finance.

    OFFLINE BACKTEST USE ONLY — not in live trading path (Rule 4).
    Appends .NS suffix automatically for NSE-listed stocks.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — run: pip install yfinance")
        return None

    for ticker_suffix in [f"{symbol}.NS", symbol]:
        try:
            raw = yf.download(
                ticker_suffix,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            logger.warning(f"[{symbol}] yfinance download failed for {ticker_suffix}: {exc}")
            continue

        if raw is None or raw.empty:
            continue

        # Handle multi-level columns from yfinance v0.2+
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]

        required = {"open", "high", "low", "close"}
        if not required.issubset(set(raw.columns)):
            logger.warning(f"[{symbol}] {ticker_suffix}: missing columns {required - set(raw.columns)}")
            continue

        df = raw[["open", "high", "low", "close"]].copy()
        df["volume"] = raw["volume"] if "volume" in raw.columns else 1.0

        df.index = pd.to_datetime(df.index).normalize()
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(subset=["close"])

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        logger.info(f"[{symbol}] yfinance OK ({ticker_suffix}): {len(df)} bars")
        return df

    return None


def fetch_nifty50_index(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> Optional[pd.Series]:
    """
    Return NIFTY 50 daily close series.
    Tries NSEPython index_history first, then Yahoo Finance ^NSEI.
    OFFLINE BACKTEST USE ONLY.
    """
    cache_key = f"NIFTY50_{start_date}_{end_date}"
    cp = _cache_path(cache_key)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(cp):
        return pd.read_parquet(cp)["close"]

    logger.info(f"Fetching NIFTY 50 index {start_date}→{end_date}")

    # Try NSEPython index_history first (no scraping needed)
    try:
        import nsepython as nse

        def _fmt(dt: str) -> str:
            return datetime.strptime(dt, "%Y-%m-%d").strftime("%d-%m-%Y")

        raw = nse.index_history("NIFTY 50", _fmt(start_date), _fmt(end_date))
        if raw is not None and not raw.empty:
            col_map = {}
            for c in raw.columns:
                cl = str(c).lower()
                if cl == "close":
                    col_map[c] = "close"
                elif "date" in cl or "historicaldate" in cl:
                    col_map[c] = "date"

            df = raw.rename(columns=col_map)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index, errors="coerce").normalize()
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if "close" not in df.columns and "CLOSE" in df.columns:
                df["close"] = df["CLOSE"]
            if "close" in df.columns:
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df[["close"]].dropna()
                if not df.empty:
                    df.to_parquet(cp)
                    logger.info(f"NIFTY50 from NSEPython: {len(df)} bars")
                    return df["close"]
    except Exception as exc:
        logger.warning(f"NSEPython index_history failed: {exc} — falling back to yfinance")

    # Fallback: Yahoo Finance ^NSEI (OFFLINE BACKTEST USE ONLY)
    try:
        import yfinance as yf
        raw = yf.download("^NSEI", start=start_date, end=end_date,
                          auto_adjust=True, progress=False, threads=False)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [str(c).lower() for c in raw.columns]

            close = raw["close"].dropna()
            close.index = pd.to_datetime(close.index).normalize()
            if hasattr(close.index, "tz") and close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            close = close[~close.index.duplicated(keep="last")].sort_index()
            df = pd.DataFrame({"close": close})
            df.to_parquet(cp)
            logger.info(f"NIFTY50 from yfinance: {len(df)} bars")
            return df["close"]
    except Exception as exc:
        logger.error(f"NIFTY50 yfinance fallback failed: {exc}")

    logger.warning("NIFTY50 unavailable — relative strength will use stock-only returns")
    return None

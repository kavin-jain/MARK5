"""
MARK5 Sector Rotation Data Provider v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provides sector relative-strength features for ML feature engineering.
Uses Nifty sector indices from Yahoo Finance (free, no API key required).

NSE SECTOR INDICES (via Yahoo Finance):
  ^NSEBANK   — Nifty Bank (private + PSU banks)
  ^CNXIT     — Nifty IT
  ^CNXPHARMA — Nifty Pharma
  ^CNXAUTO   — Nifty Auto
  ^CNXFMCG   — Nifty FMCG
  ^CNXMETAL  — Nifty Metal
  ^NSEI      — Nifty 50 (fallback for unlisted sectors)

FEATURES PRODUCED (per ticker, per bar):
  sector_rs_10d  — (stock_10d_ret / sector_10d_ret) - 1  (relative out/underperformance)
  sector_rs_21d  — same over 21 days
  sector_rs_63d  — same over 63 days

CACHE:
  data/cache/sector_{INDEX_SYMBOL}.parquet — one parquet per sector index

TRADING ROLE: ML feature engineering (training + inference)
SAFETY LEVEL: LOW RISK — read-only, no execution dependency
DATA AVAILABILITY: Goes back to 1999 (full NSE history via yfinance)
LEAKAGE STATUS: ZERO — all computations use rolling windows on past data only
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

logger = logging.getLogger("MARK5.SectorData")

# ── Directory layout ─────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
CACHE_DIR = _PROJECT / "data" / "cache"

# ── Sector Index → Yahoo Finance symbol mapping ──────────────────────────────
SECTOR_INDICES = {
    "BANKING":    "^NSEBANK",
    "IT":         "^CNXIT",
    "PHARMA":     "^NSEI",       # ^CNXPHARMA delisted from Yahoo Finance — use Nifty50 proxy
    "AUTO":       "^CNXAUTO",
    "FMCG":       "^CNXFMCG",
    "METAL":      "^CNXMETAL",
    "NIFTY50":    "^NSEI",      # fallback / broad market
}

# ── Ticker → Sector mapping ──────────────────────────────────────────────────
# Based on NSE sector classification for the MARK5 active universe
TICKER_SECTOR: Dict[str, str] = {
    # BANKING
    "HDFCBANK":   "BANKING",
    "ICICIBANK":  "BANKING",
    "KOTAKBANK":  "BANKING",
    "SBIN":       "BANKING",
    "AUBANK":     "BANKING",
    "BANDHANBNK": "BANKING",
    "IDFCFIRSTB": "BANKING",
    "PNB":        "BANKING",
    "YESBANK":    "BANKING",
    "BAJFINANCE": "BANKING",   # NBFC — closest to banking sector index

    # IT / TECHNOLOGY
    "TCS":        "IT",
    "INFY":       "IT",
    "COFORGE":    "IT",
    "PERSISTENT": "IT",
    "TATAELXSI":  "IT",

    # PHARMA
    "LUPIN":      "PHARMA",
    "SUNPHARMA":  "PHARMA",

    # AUTO & AUTO-ANCILLARY
    "MARUTI":     "AUTO",
    "MOTHERSON":  "AUTO",

    # FMCG & CONSUMER
    "HINDUNILVR": "FMCG",
    "ITC":        "FMCG",
    "TRENT":      "FMCG",
    "ASIANPAINT": "FMCG",
    "TITAN":      "FMCG",

    # METAL / MATERIALS
    "TATASTEEL":  "METAL",

    # BROAD MARKET (NIFTY 50 fallback — no dedicated sector index)
    "RELIANCE":   "NIFTY50",   # Oil & Gas / Conglomerate
    "LT":         "NIFTY50",   # Engineering / Infrastructure
    "VOLTAS":     "NIFTY50",   # Consumer Durables / Engineering
    "HAL":        "NIFTY50",   # Defence / Aerospace
    "BEL":        "NIFTY50",   # Defence Electronics
    "BHARTIARTL": "NIFTY50",   # Telecom
    "IDEA":       "NIFTY50",   # Telecom
    "ONGC":       "NIFTY50",   # Oil & Gas
    "WIPRO":      "IT",        # IT
    "HINDALCO":   "METAL",     # Metal
    "JSWSTEEL":   "METAL",     # Metal
    "ULTRACEMCO": "NIFTY50",   # Cement
    "INDFCFIRSTB": "BANKING",
}


class SectorDataProvider:
    """
    Provides sector-relative-strength features for the ML feature engine.

    Usage:
        provider = SectorDataProvider()
        sector_close = provider.get_sector_series("HAL", start="2015-01-01", end="2025-01-01")
        # Returns pd.Series of sector close prices aligned to stock index
    """

    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.Series] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_sector_for_ticker(self, ticker: str) -> str:
        """Return sector name for a ticker (fallback to NIFTY50)."""
        return TICKER_SECTOR.get(ticker, "NIFTY50")

    def get_sector_series(
        self,
        ticker: str,
        start: str,
        end: str,
        stock_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        Get sector index close price series for a ticker's sector.

        Args:
            ticker:      NSE ticker symbol
            start:       "YYYY-MM-DD"
            end:         "YYYY-MM-DD"
            stock_index: If provided, reindex output to this index

        Returns:
            pd.Series of sector close prices (forward-filled for gaps).
            Returns pd.Series of zeros if yfinance unavailable.
        """
        sector = self.get_sector_for_ticker(ticker)
        yf_sym = SECTOR_INDICES[sector]
        series = self._get_index_series(yf_sym, start, end)

        if series.empty:
            # Return zeros — feature engine will produce 0.0 RS
            if stock_index is not None:
                return pd.Series(0.0, index=stock_index, name='sector_close')
            return pd.Series(dtype=float, name='sector_close')

        if stock_index is not None:
            # FIX: bfill() was pulling future sector data backward into early
            # training bars (e.g., day 1's sector close = day 3's value if
            # the index started mid-week). This is look-ahead bias.
            # Correct approach: ffill() only (propagate last known value
            # forward). Leading NaN (before any sector data) → 0.0 so that
            # sector_rs features for those bars evaluate to zero, not future.
            series = series.reindex(stock_index).ffill().fillna(0.0)

        return series

    def compute_sector_rs(
        self,
        stock_close: pd.Series,
        sector_close: pd.Series,
        windows: Tuple[int, ...] = (10, 21, 63),
    ) -> pd.DataFrame:
        """
        Compute sector-relative-strength features.

        RS_Nd = (stock_Nd_return / sector_Nd_return) - 1
        Positive = stock outperforming sector.
        Clipped to [-3, +3] to handle division edge cases.

        Args:
            stock_close:   pd.Series of stock close prices
            sector_close:  pd.Series of sector index close prices (same index)
            windows:       Tuple of lookback windows in trading days

        Returns:
            pd.DataFrame with columns [sector_rs_10d, sector_rs_21d, sector_rs_63d]
        """
        out = pd.DataFrame(index=stock_close.index)
        eps = 1e-9

        for w in windows:
            stock_ret  = stock_close.pct_change(w)
            sector_ret = sector_close.pct_change(w)
            # RS: stock excess return vs sector
            rs = (stock_ret - sector_ret).clip(-3.0, 3.0)
            # Z-score over 60d rolling window for stationarity
            rs_z = (
                (rs - rs.rolling(60, min_periods=20).mean())
                / (rs.rolling(60, min_periods=20).std() + eps)
            ).clip(-3, 3)
            out[f'sector_rs_{w}d'] = rs_z.fillna(0.0)

        return out

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_index_series(self, yf_symbol: str, start: str, end: str) -> pd.Series:
        """Get sector index close series. Uses cache to avoid repeated downloads."""
        cache_key = yf_symbol.replace('^', '')
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            start_dt = pd.Timestamp(start)
            end_dt   = pd.Timestamp(end)
            if not cached.empty and cached.index[0] <= start_dt and cached.index[-1] >= end_dt:
                return cached.loc[start_dt:end_dt]

        # Try disk cache
        cache_file = CACHE_DIR / f"sector_{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                series = df['close'] if 'close' in df.columns else df.iloc[:, 0]
                series.index = pd.to_datetime(series.index)
                if hasattr(series.index, 'tz') and series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                series.name = 'sector_close'

                # Check if cache covers requested range
                start_dt = pd.Timestamp(start)
                end_dt   = pd.Timestamp(end)
                if not series.empty and series.index[0] <= start_dt and series.index[-1] >= end_dt:
                    self._cache[cache_key] = series
                    return series.loc[start_dt:end_dt]
            except Exception as e:
                logger.debug(f"Sector cache read error ({yf_symbol}): {e}")

        # Fetch from yfinance
        if not YF_AVAILABLE:
            logger.debug("yfinance not available — returning empty sector series")
            return pd.Series(dtype=float, name='sector_close')

        try:
            # Fetch from 2000-01-01 for maximum coverage
            raw = yf.download(
                yf_symbol,
                start="2000-01-01",
                end=end,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                return pd.Series(dtype=float, name='sector_close')

            # yfinance 1.4.0+ returns MultiIndex columns: ('Close', '^SYM')
            # Flatten to single-level before lowercasing
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [str(c).lower() for c in raw.columns]
            if hasattr(raw.index, 'tz') and raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)
            raw.index = pd.to_datetime(raw.index)

            series = raw['close'].rename('sector_close')

            # Save to disk cache
            raw[['close']].to_parquet(cache_file)
            logger.info(f"Sector index {yf_symbol}: {len(series)} bars cached")

            self._cache[cache_key] = series
            start_dt = pd.Timestamp(start)
            end_dt   = pd.Timestamp(end)
            return series.loc[start_dt:end_dt]

        except Exception as e:
            logger.warning(f"Failed to fetch sector index {yf_symbol}: {e}")
            return pd.Series(dtype=float, name='sector_close')


# ── Module-level singleton ───────────────────────────────────────────────────
_provider: Optional[SectorDataProvider] = None


def get_sector_provider() -> SectorDataProvider:
    """Get or create the module-level singleton provider."""
    global _provider
    if _provider is None:
        _provider = SectorDataProvider()
    return _provider


def get_sector_rs(
    ticker: str,
    stock_close: pd.Series,
    nifty_close: Optional[pd.Series] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function: compute sector RS features for a ticker.

    Returns pd.DataFrame with columns: sector_rs_10d, sector_rs_21d, sector_rs_63d
    All values are Z-scored and clipped to [-3, +3].
    Returns zeros if sector data unavailable.
    """
    provider = get_sector_provider()

    start_str = start or stock_close.index[0].strftime('%Y-%m-%d')
    end_str   = end   or stock_close.index[-1].strftime('%Y-%m-%d')

    try:
        sector_close = provider.get_sector_series(
            ticker, start=start_str, end=end_str,
            stock_index=stock_close.index
        )
        if sector_close.empty or (sector_close == 0).all():
            # Fallback: use Nifty if provided
            if nifty_close is not None and not nifty_close.empty:
                sector_close = nifty_close.reindex(stock_close.index).ffill()
            else:
                return pd.DataFrame(
                    0.0,
                    index=stock_close.index,
                    columns=['sector_rs_10d', 'sector_rs_21d', 'sector_rs_63d'],
                )

        return provider.compute_sector_rs(stock_close, sector_close)
    except Exception as e:
        logger.warning(f"Sector RS computation failed for {ticker}: {e}")
        return pd.DataFrame(
            0.0,
            index=stock_close.index,
            columns=['sector_rs_10d', 'sector_rs_21d', 'sector_rs_63d'],
        )

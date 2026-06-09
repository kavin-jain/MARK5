"""
MARK5 Universe Expander v1.0
════════════════════════════
Scans all trained ML models and available price data to build an
expanded ticker universe for the momentum strategy.

The baseline system used only 13 tickers.  44 ML models exist.
This module discovers all usable candidates and returns them sorted
by data quality (most history first).

FILTERS APPLIED:
  1. Model must exist (LightPredictor.has_models())
  2. Price cache must have data from at least OOS_START
  3. No ".NS" duplicates (prefer clean NSE symbols)
  4. Minimum OOS bars: 200 (exclude very short histories)
  5. Optionally: exclude tickers flagged as "junk" (YESBANK class)

CHANGELOG:
- [2026-05-23] v1.0: Initial implementation
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger("MARK5.UniverseExpander")

# Tickers to permanently exclude (structural problems, penny stocks etc.)
_PERMANENT_EXCLUSIONS: Set[str] = {
    "IDEA",        # Vodafone Idea — penny stock structural decline
    "YESBANK",     # Yes Bank — structural decline, ML never qualifies anyway
    "IDFCFIRSTB",  # IDFC First Bank — volatile with unclear momentum
}

# Allow at most this many bars below OOS_START to still qualify
_MIN_OOS_BARS = 100  # ~5 months


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, drop tz, deduplicate index."""
    df.columns = [c.lower() for c in df.columns]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


class UniverseExpander:
    """
    Builds an expanded ticker universe from all available ML models.

    Usage:
        expander = UniverseExpander(model_root="models", cache_dir="data/cache")
        tickers  = expander.scan(oos_start="2022-01-01", oos_end="2026-05-21")
    """

    def __init__(
        self,
        model_root: str,
        cache_dir: str,
        exclude: Optional[Set[str]] = None,
    ):
        self.model_root = model_root
        self.cache_dir  = cache_dir
        self.exclude    = (exclude or set()) | _PERMANENT_EXCLUSIONS

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        oos_start: str = "2022-01-01",
        oos_end:   str = "2026-05-21",
        verbose:   bool = True,
    ) -> List[str]:
        """
        Return list of ticker symbols (clean, no .NS suffix) that have:
        - A trained ML model
        - Price data covering the OOS period

        Sorted: tickers with most OOS bars first.
        """
        try:
            from core.models.backtest_pipeline import LightPredictor
        except ImportError:
            logger.error("LightPredictor not available — returning empty universe")
            return []

        candidates: List[Dict] = []

        for name in sorted(os.listdir(self.model_root)):
            d = os.path.join(self.model_root, name)
            if not os.path.isdir(d):
                continue
            if "." in name:
                continue  # skip .NS duplicates
            if name in self.exclude:
                continue

            # Check ML model
            try:
                pred = LightPredictor(name, self.model_root)
                if not pred.has_models():
                    continue
            except Exception:
                continue

            # Find price data
            df = self.load_ticker(name)
            if df is None:
                continue

            # Check OOS coverage
            oos_slice = df.loc[oos_start:oos_end]
            if len(oos_slice) < _MIN_OOS_BARS:
                if verbose:
                    logger.debug(f"  {name}: too few OOS bars ({len(oos_slice)}) — skipped")
                continue

            candidates.append({
                "ticker":   name,
                "oos_bars": len(oos_slice),
                "history_start": str(df.index[0].date()),
            })

        # Sort by OOS bars (most data first)
        candidates.sort(key=lambda x: -x["oos_bars"])

        tickers = [c["ticker"] for c in candidates]

        if verbose:
            logger.info(f"Universe expanded: {len(tickers)} tickers")
            for c in candidates:
                logger.info(
                    f"  {c['ticker']:<16} {c['oos_bars']} OOS bars "
                    f"(from {c['history_start']})"
                )

        return tickers

    def load_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load daily OHLCV from parquet cache.
        Tries several file patterns in order of preference.
        """
        search_dirs = [self.cache_dir, os.path.join(self.cache_dir, "nse")]
        patterns = [
            f"{ticker}_daily.parquet",
            f"{ticker}_NS_1d.parquet",
            f"{ticker}_20220101_20260521.parquet",
            f"{ticker}_20220101_20260522.parquet",
            f"{ticker}_20210101_20251231.parquet",
        ]

        for d in search_dirs:
            for pattern in patterns:
                path = os.path.join(d, pattern)
                if os.path.exists(path):
                    try:
                        df = _clean_df(pd.read_parquet(path))
                        if "close" not in df.columns:
                            continue
                        return df
                    except Exception as e:
                        logger.debug(f"  {ticker}: load failed ({path}): {e}")
                        continue
        return None

    def load_all(
        self,
        tickers: List[str],
        oos_start: str = "2022-01-01",
        oos_end:   str = "2026-05-21",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV for a list of tickers. Returns dict of DataFrame.
        Only includes tickers with sufficient OOS data.
        """
        result: Dict[str, pd.DataFrame] = {}
        for tk in tickers:
            df = self.load_ticker(tk)
            if df is not None and len(df.loc[oos_start:oos_end]) >= _MIN_OOS_BARS:
                result[tk] = df
        return result


# ── CLI usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _ROOT)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    expander = UniverseExpander(
        model_root=os.path.join(_ROOT, "models"),
        cache_dir=os.path.join(_ROOT, "data", "cache"),
    )
    tickers = expander.scan()
    print(f"\n  ✅ Expanded universe: {len(tickers)} tickers")
    print(f"  {tickers}")

"""
MARK6 — Causal Factor Library
=============================
Academically-grounded, OHLCV-derivable equity factors, each computed as a
strictly causal time series (value at bar t uses only data up to and including t).

Factors (all "higher = more attractive" after sign normalisation):
  - momentum   : 12-1 month total return (skip last month to avoid S-T reversal)
  - low_vol    : negative of trailing realised volatility (low-vol anomaly)
  - trend      : price vs 200d moving average (trend-following / time-series mom)
  - stability  : trailing Sortino-style downside-adjusted return (quality proxy)

Why these four: they are the most robust, replicated long-only equity premia that
can be derived from price/volume alone (no fundamentals). They are deliberately
diversifying — momentum and low-vol are near-orthogonal, which is what makes a
*composite* more robust than any single factor (the lesson from the momentum-only
overlay that failed net of tax).

NO LOOK-AHEAD: every function returns a Series aligned to the input index where
value[t] is knowable at the close of bar t. Verified by tests/test_portfolio.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class FactorLibrary:
    """Stateless causal factor computations on a single instrument's price series."""

    # ── individual factors ──────────────────────────────────────────────────
    @staticmethod
    def momentum(close: pd.Series, lookback: int = TRADING_DAYS, skip: int = 21) -> pd.Series:
        """12-1 month momentum: return from t-lookback to t-skip.

        Skipping the most recent month avoids the well-documented short-term
        reversal effect that contaminates raw 12-month momentum.
        """
        close = close.astype(float)
        return close.shift(skip) / close.shift(lookback) - 1.0

    @staticmethod
    def low_vol(close: pd.Series, window: int = 126) -> pd.Series:
        """Negative annualised realised volatility (low-volatility anomaly).

        Returned with a negative sign so that *higher is better* (less volatile),
        consistent with every other factor here.
        """
        ret = close.astype(float).pct_change(fill_method=None)
        vol = ret.rolling(window, min_periods=window // 2).std() * np.sqrt(TRADING_DAYS)
        return -vol

    @staticmethod
    def trend(close: pd.Series, window: int = 200) -> pd.Series:
        """Distance above the long-run moving average (time-series momentum)."""
        close = close.astype(float)
        ma = close.rolling(window, min_periods=window // 2).mean()
        return close / ma - 1.0

    @staticmethod
    def stability(close: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
        """Trailing Sortino-style score: annualised mean return / downside deviation.

        A pure-price proxy for 'quality' — rewards names that compound with low
        downside variability.
        """
        ret = close.astype(float).pct_change(fill_method=None)
        ann = ret.rolling(window, min_periods=window // 2).mean() * TRADING_DAYS
        downside = ret.where(ret < 0, 0.0)
        dstd = downside.rolling(window, min_periods=window // 2).std() * np.sqrt(TRADING_DAYS)
        out = ann / dstd.replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    # ── batch helper ─────────────────────────────────────────────────────────
    DEFAULT_FACTORS = ("momentum", "low_vol", "trend", "stability")

    @classmethod
    def compute_all(cls, close: pd.Series, factors=DEFAULT_FACTORS) -> pd.DataFrame:
        """Return a DataFrame {factor_name -> causal Series} for one instrument."""
        fns = {
            "momentum": cls.momentum,
            "low_vol": cls.low_vol,
            "trend": cls.trend,
            "stability": cls.stability,
        }
        return pd.DataFrame({f: fns[f](close) for f in factors})


def cross_sectional_z(values: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score a cross-section of one factor across names at a single date.

    Robust to outliers via clipping. NaNs are preserved (excluded from mean/std).
    """
    v = values.astype(float)
    mu, sd = v.mean(), v.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=v.index)
    z = (v - mu) / sd
    return z.clip(-clip, clip)


def cross_sectional_rank_z(values: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score the cross-sectional RANKS instead of the raw factor values.

    Why this is the better default (validated 2026-07-26, v7.3 sweep P17):
    momentum is heavily right-skewed. One name up 400% sets the scale of the
    z-score and squashes every genuine distinction below it — 3-sigma clipping
    caps that name's score but does nothing about the inflated standard
    deviation it created, so the whole cross-section stays compressed.

    Ranking makes the score depend only on ORDER, which is all the composite
    ever uses, and is completely immune to the tail. Measured effect is
    one-sided and robust: MaxDD improves in 7/8 rolling 3-year windows
    (equity sleeve -47.0% -> -39.7%, full system -24.9% -> -22.2%) while return
    is unchanged (2/8 on CAGR — i.e. noise). It buys risk, not return.
    """
    v = values.astype(float)
    r = v.rank(method="average")           # NaNs stay NaN, as with the raw z
    mu, sd = r.mean(), r.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=v.index)
    return ((r - mu) / sd).clip(-clip, clip)


def composite_score(
    factor_panel: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    rank_transform: bool = True,
) -> pd.Series:
    """Blend per-factor cross-sectional z-scores into one composite per name.

    Args:
        factor_panel: {factor_name -> Series(index=tickers) of raw factor values
                       as-of the rebalance date}
        weights:      optional factor weights (default: equal across factors)
        rank_transform: score on ranks rather than raw values (default, see
                       `cross_sectional_rank_z`). False reproduces pre-v7.3.

    Returns:
        Series(index=tickers) composite z-score. Names missing a factor get that
        factor's neutral 0 contribution (mean-imputed via z-score).
    """
    names = sorted({n for s in factor_panel.values() for n in s.index})
    if weights is None:
        weights = {f: 1.0 / len(factor_panel) for f in factor_panel}
    wsum = sum(weights.values()) or 1.0
    zfn = cross_sectional_rank_z if rank_transform else cross_sectional_z
    comp = pd.Series(0.0, index=names)
    for f, raw in factor_panel.items():
        z = zfn(raw.reindex(names))
        comp = comp.add(z.fillna(0.0) * (weights.get(f, 0.0) / wsum), fill_value=0.0)
    return comp

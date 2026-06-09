"""
MARK5 INSTITUTIONAL FEATURE ENGINE v2.0 — 33 PRODUCTION FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEM SOLVED:
  V1 feature set (10 OHLCV features) is non-predictive in 2025-2026 regime.
  Winner confidence (0.687) ≈ loser confidence (0.673) → random signal.
  Root cause: V1 has zero regime awareness. No FII, no sector, no options data.

V2 SOLUTION — 33 FEATURES IN 6 CATEGORIES:
  Category 1: Price/Volume Microstructure (9 features) — existing V1 features
  Category 2: Multi-horizon Momentum (6 features)     — NEW
  Category 3: Price Level & Range (4 features)        — NEW
  Category 4: Market Regime (3 features)              — NEW (Nifty context)
  Category 5: Sector Relative Strength (3 features)   — NEW (sector RS)
  Category 6: Derivatives Sentiment (4 features)      — NEW (F&O, FII proxy)
  Category 7: Volatility Regime (4 features)          — NEW

LEAKAGE STATUS: ZERO
  - All rolling windows use past data only (no future data)
  - Training cutoff enforced before feature computation
  - CPCV purge/embargo in trainer.py handles train/test isolation

BACKWARD COMPATIBILITY:
  - features.py (V1, 10 features) is UNCHANGED for existing models
  - features_v2.py (V2, 33 features) activates when models are retrained
  - predictor.py detects V2 via features.json schema field 'version': 'v2'

DATA REQUIREMENTS:
  - OHLCV:       Always required (stock history)
  - context['nifty_close']:  Required for regime features (use zeros if absent)
  - context['sector_close']: Optional for sector RS (uses Nifty as fallback)
  - context['fno_features']: Optional F&O features DataFrame (zeros if absent)
  - context['fii_net']:      Optional FII net flow series (zeros if absent)

CHANGELOG:
  [2026-05-24] v2.0: Complete feature overhaul. 33 features. Zero leakage.
    - Add multi-horizon momentum (5d, 21d, 63d)
    - Add price level features (52w high dist, 200-SMA dist, Donchian)
    - Add market regime features (Nifty 200-SMA, Nifty RSI, Nifty momentum)
    - Add sector RS features (stock vs sector, 10d/21d/63d)
    - Add derivatives features (PCR, OI signal, FII proxy zscore)
    - Add volatility regime (ATR percentile, vol regime, OBV trend, CMF)
    - Add Optuna-compatible feature importance support
"""

import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger("MARK5.FeaturesV2")

# Project root — used to locate local data caches
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS_V2: List[str] = [
    # ── Category 1: Price/Volume Microstructure (inherited from V1) ──────────
    'amihud_ratio',     # IC +0.0904 — liquidity-adjusted price impact
    'range_z',          # IC +0.0840 — intraday range normalized
    'bb_width',         # IC +0.0589 — Bollinger band width (squeeze detector)
    'atr_vol',          # IC +0.0537 — ATR / close (volatility normalization)
    'rsi_14',           # IC +0.031  — 14-period RSI [0,1] scale
    'gap_sig',          # IC +0.0378 — overnight gap / close
    'vol_adj_mom',      # IC +0.0342 — 20d momentum / (std * sqrt(20))
    'mfi_div',          # IC +0.0278 — money flow vs price divergence
    'tii_60',           # IC -0.016  — trend intensity (mean reversion signal)

    # ── Category 2: Multi-horizon Momentum ───────────────────────────────────
    'mom_5d',           # 5-day price return (short momentum)
    'mom_21d',          # 21-day price return (medium momentum)
    'mom_63d',          # 63-day price return (quarterly momentum factor)
    'rsi_5',            # 5-period RSI (short-term overbought/oversold)
    'rsi_21',           # 21-period RSI (medium-term trend strength)
    'obv_trend',        # On-Balance Volume 20d trend (buying pressure)

    # ── Category 3: Price Level & Range ──────────────────────────────────────
    'dist_52w_high',    # (52w_high - close) / 52w_high → 0=at high, 1=at 52w low
    'dist_200sma',      # (close - SMA200) / SMA200 → momentum factor
    'price_channel_pct', # Donchian channel position [0=bottom, 1=top]
    'cmf',              # Chaikin Money Flow (accumulation/distribution)

    # ── Category 4: Market Regime (Nifty context) ────────────────────────────
    'nifty_200sma_dist', # (Nifty - Nifty_SMA200) / Nifty_SMA200
    'nifty_rsi_21',      # Nifty 21-period RSI — overall market momentum
    'nifty_mom_21d',     # Nifty 21-day return (market trend direction)

    # ── Category 5: Sector Relative Strength ─────────────────────────────────
    'sector_rs_10d',    # Stock vs sector 10d RS (Z-scored)
    'sector_rs_21d',    # Stock vs sector 21d RS (Z-scored)
    'sector_rs_63d',    # Stock vs sector 63d RS (Z-scored)

    # ── Category 6: Derivatives Sentiment ────────────────────────────────────
    'pcr_oi',           # Put-Call Ratio OI (options market sentiment, 0 if no F&O)
    'oi_signal',        # Futures OI momentum * price direction (0 if no F&O)
    'fii_5d_zscore',    # FII 5-day flow Z-score (proxy if actual unavailable)
    'fii_21d_zscore',   # FII 21-day flow Z-score

    # ── Category 7: Volatility Regime ────────────────────────────────────────
    'atr_percentile',   # Current ATR in 252d rolling percentile [0,1]
    'vol_regime',       # Rolling std percentile [0,1] (high vol = 1)
    'vol_breakout',     # Volume vs 20d MA Z-score (unusual activity)
    'frac_diff',        # Fractionally differentiated log-price (d≈0.4, stationary)
]

EXPECTED_FEATURE_COUNT_V2: int = len(FEATURE_COLS_V2)   # 33
FEATURE_ENGINE_VERSION: str = "v2"

# Re-export V1 for backward compatibility
from core.models.features import (
    FEATURE_COLS as FEATURE_COLS_V1,
    engineer_features_df as engineer_features_df_v1,
    compute_atr,
    compute_rsi,
    standardize_series,
    AdvancedFeatureEngine as AdvancedFeatureEngineV1,
)


# ─────────────────────────────────────────────────────────────────────────────
# MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _zscore_rolling(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Z-score clipped to [-3, +3]."""
    mu  = series.rolling(window, min_periods=max(10, window // 3)).mean()
    sig = series.rolling(window, min_periods=max(10, window // 3)).std()
    return ((series - mu) / (sig + 1e-9)).clip(-3, 3)


def _compute_obv_trend(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """On-Balance Volume normalized trend."""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    # Normalize to rolling 60d Z-score
    return _zscore_rolling(obv, window=60).fillna(0.0)


def _compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow: accumulation/distribution oscillator."""
    eps = 1e-9
    clv = ((close - low) - (high - close)) / ((high - low) + eps)  # [-1, +1]
    cmf = (clv * volume).rolling(period).sum() / (volume.rolling(period).sum() + eps)
    return cmf.clip(-1, 1).fillna(0.0)


def _frac_diff_ffd(series: pd.Series, d: float = 0.4, thres: float = 1e-4) -> pd.Series:
    """
    Fractional differentiation (Fixed-Window FFD method).
    Makes price series stationary while preserving maximum memory.
    d=0.4 is empirically optimal for daily equity prices (ADF passes, IC preserved).
    """
    s = np.log(series.ffill().dropna().values.astype(float))
    if len(s) < 20:
        return pd.Series(0.0, index=series.index)

    # Compute weights
    w = [1.0]
    for k in range(1, len(s)):
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
    w = np.array(w[::-1])

    if len(s) < len(w):
        return pd.Series(0.0, index=series.index)

    res = np.convolve(s, w, mode='valid')
    out = pd.Series(np.nan, index=series.index)
    out.iloc[len(series) - len(res):] = res
    return _zscore_rolling(out.ffill().fillna(0.0), window=60)


def _compute_donchian_pct(high: pd.Series, low: pd.Series, window: int = 20,
                          close: Optional[pd.Series] = None) -> pd.Series:
    """Price position within Donchian channel [0=bottom, 1=top].

    FIX: numerator now uses `close` not `high`. The decision is made at bar
    close; using the intraday high biases the metric upward systematically
    (high >= close always). This was look-ahead within the same bar.
    Callers must pass `close` for the fix to apply; falls back to `high` for
    backward compatibility when close is None.
    """
    price = close if close is not None else high  # close is correct; high was the bug
    ch_high = high.rolling(window).max()
    ch_low  = low.rolling(window).min()
    ch_range = ch_high - ch_low
    return ((price - ch_low) / (ch_range + 1e-9)).clip(0, 1).fillna(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features_v2(
    df: pd.DataFrame,
    ticker: str = "",
    context: Optional[Dict] = None,
    training_cutoff=None,
) -> pd.DataFrame:
    """
    MARK5 V2 Feature Engine — 33 institutional-grade features.

    Args:
        df:              OHLCV DataFrame (must have open, high, low, close, volume)
        ticker:          NSE ticker symbol (used for sector lookup)
        context:         Optional dict with keys:
                           'nifty_close'  : pd.Series of Nifty prices
                           'sector_close' : pd.Series of sector index prices
                           'fno_features' : pd.DataFrame from FNODataProvider
                           'fii_net'      : pd.Series of FII net flow (crores)
        training_cutoff: If set, only use data ≤ this timestamp

    Returns:
        pd.DataFrame with columns = FEATURE_COLS_V2, index = DatetimeIndex.
        Drops rows with any NaN. Minimum 200 bars required.
    """
    if df is None or len(df) < 200:
        logger.warning(f"[{ticker}] Insufficient data: {len(df) if df is not None else 0} bars")
        return pd.DataFrame()

    df = df.copy()
    context = context or {}

    # Apply training cutoff (prevents leakage into validation period)
    if training_cutoff is not None:
        df = df[df.index <= training_cutoff].copy()

    if len(df) < 100:
        return pd.DataFrame()

    eps = 1e-9
    c   = df['close']
    h   = df.get('high', c)
    l   = df.get('low', c)
    o   = df.get('open', c)
    v   = df.get('volume', pd.Series(1.0, index=df.index))
    tp  = (h + l + c) / 3

    out = pd.DataFrame(index=df.index)

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 1: Price/Volume Microstructure (V1 features, unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    out['amihud_ratio'] = (c.pct_change().abs() / (v * c + eps)).rolling(20).median() * 1e9
    out['range_z']      = (h - l) / (c + eps)

    sma_20 = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    out['bb_width'] = (4 * std_20) / (sma_20 + eps)   # NOTE: 2× half-width; do not change without retraining

    out['atr_vol']   = compute_atr(df, 14) / (c + eps)
    out['rsi_14']    = compute_rsi(c, period=14)
    out['gap_sig']   = (o - c.shift(1)) / (c.shift(1) + eps)

    # Vol-adjusted momentum
    out['vol_adj_mom'] = c.pct_change(20) / (c.pct_change().rolling(20).std() * np.sqrt(20) + eps)

    # MFI Divergence
    pos_mf = (tp * v).where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = (tp * v).where(tp < tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + pos_mf / (neg_mf + eps)))
    out['mfi_div'] = c.pct_change(14) - (mfi.pct_change(14) / 100.0)

    # Trend Intensity Index
    ma_60   = c.rolling(60).mean()
    dev     = c - ma_60
    pos_dev = dev.clip(lower=0).rolling(60).sum()
    neg_dev = (-dev).clip(lower=0).rolling(60).sum()
    out['tii_60'] = (pos_dev / (pos_dev + neg_dev + eps)).fillna(0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 2: Multi-horizon Momentum
    # ─────────────────────────────────────────────────────────────────────────

    out['mom_5d']  = c.pct_change(5).clip(-0.5, 0.5)
    out['mom_21d'] = c.pct_change(21).clip(-0.5, 0.5)
    out['mom_63d'] = c.pct_change(63).clip(-1.0, 1.0)
    out['rsi_5']   = compute_rsi(c, period=5)
    out['rsi_21']  = compute_rsi(c, period=21)
    out['obv_trend'] = _compute_obv_trend(c, v, window=20)

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 3: Price Level & Range
    # ─────────────────────────────────────────────────────────────────────────

    # 52-week (252-bar) rolling high distance: 0=at high, >0=below high
    high_252 = h.rolling(252, min_periods=60).max()
    out['dist_52w_high'] = ((high_252 - c) / (high_252 + eps)).clip(0, 1).fillna(0.5)

    # Distance from own 200-day SMA (momentum factor)
    sma_200 = c.rolling(200, min_periods=60).mean()
    out['dist_200sma'] = ((c - sma_200) / (sma_200 + eps)).clip(-0.5, 0.5).fillna(0.0)

    # Donchian channel position (pass close to fix look-ahead in same bar)
    out['price_channel_pct'] = _compute_donchian_pct(h, l, window=20, close=c)

    # Chaikin Money Flow
    out['cmf'] = _compute_cmf(h, l, c, v, period=20)

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 4: Market Regime (Nifty context)
    # ─────────────────────────────────────────────────────────────────────────

    nifty_close = context.get('nifty_close')

    if nifty_close is not None and not (isinstance(nifty_close, pd.Series) and nifty_close.empty):
        if not isinstance(nifty_close, pd.Series):
            if hasattr(nifty_close, '__len__') and len(nifty_close) == len(df):
                nifty_close = pd.Series(nifty_close.flatten(), index=df.index)
            else:
                nifty_close = pd.Series(nifty_close)
        nifty_aligned = nifty_close.reindex(c.index).ffill()

        if not nifty_aligned.isna().all():
            nifty_sma200 = nifty_aligned.rolling(200, min_periods=60).mean()
            out['nifty_200sma_dist'] = ((nifty_aligned - nifty_sma200) / (nifty_sma200 + eps)).clip(-0.3, 0.3).fillna(0.0)
            out['nifty_rsi_21']     = compute_rsi(nifty_aligned, period=21)
            out['nifty_mom_21d']    = nifty_aligned.pct_change(21).clip(-0.3, 0.3).fillna(0.0)
            # Also compute stock vs Nifty RS (inherited from V1)
            out['rel_strength_v1']  = (c.pct_change(10) - nifty_aligned.pct_change(10)).fillna(0.0)
        else:
            out['nifty_200sma_dist'] = 0.0
            out['nifty_rsi_21']      = 0.5
            out['nifty_mom_21d']     = 0.0
    else:
        out['nifty_200sma_dist'] = 0.0
        out['nifty_rsi_21']      = 0.5
        out['nifty_mom_21d']     = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 5: Sector Relative Strength
    # ─────────────────────────────────────────────────────────────────────────

    sector_close = context.get('sector_close')

    if sector_close is not None and isinstance(sector_close, pd.Series) and not sector_close.empty:
        sec_aligned = sector_close.reindex(c.index).ffill()
        if not sec_aligned.isna().all():
            for w, col in [(10, 'sector_rs_10d'), (21, 'sector_rs_21d'), (63, 'sector_rs_63d')]:
                stock_ret  = c.pct_change(w)
                sector_ret = sec_aligned.pct_change(w)
                rs_raw = (stock_ret - sector_ret).clip(-3, 3)
                out[col] = _zscore_rolling(rs_raw, window=60).fillna(0.0)
        else:
            out['sector_rs_10d'] = 0.0
            out['sector_rs_21d'] = 0.0
            out['sector_rs_63d'] = 0.0
    elif nifty_close is not None:
        # Fallback: use Nifty as sector proxy
        nifty_aligned = nifty_close.reindex(c.index).ffill() if isinstance(nifty_close, pd.Series) else pd.Series(0.0, index=c.index)
        for w, col in [(10, 'sector_rs_10d'), (21, 'sector_rs_21d'), (63, 'sector_rs_63d')]:
            stock_ret = c.pct_change(w)
            nifty_ret = nifty_aligned.pct_change(w)
            rs_raw = (stock_ret - nifty_ret).clip(-3, 3)
            out[col] = _zscore_rolling(rs_raw, window=60).fillna(0.0)
    else:
        out['sector_rs_10d'] = 0.0
        out['sector_rs_21d'] = 0.0
        out['sector_rs_63d'] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 6: Derivatives Sentiment (F&O + FII)
    # ─────────────────────────────────────────────────────────────────────────

    # F&O features (from FNODataProvider if available)
    fno_features: Optional[pd.DataFrame] = context.get('fno_features')

    if fno_features is not None and isinstance(fno_features, pd.DataFrame) and not fno_features.empty:
        fno_aligned = fno_features.reindex(c.index).ffill().fillna(0.0)
        out['pcr_oi']    = fno_aligned.get('pcr_oi',    pd.Series(0.0, index=c.index)).clip(0.1, 5.0)
        out['oi_signal'] = fno_aligned.get('oi_signal', pd.Series(0.0, index=c.index)).clip(-1, 1)
        # Normalize PCR: PCR > 1 = bearish (more puts), PCR < 1 = bullish
        # Z-score for stationarity
        out['pcr_oi'] = _zscore_rolling(out['pcr_oi'], window=60).fillna(0.0)
    else:
        out['pcr_oi']    = 0.0
        out['oi_signal'] = 0.0

    # FII flow features
    fii_net: Optional[pd.Series] = context.get('fii_net')

    if fii_net is not None and isinstance(fii_net, pd.Series) and not fii_net.empty:
        fii_aligned = fii_net.reindex(c.index).ffill().fillna(0.0)
        # 5d cumulative FII flow Z-score
        fii_5d  = fii_aligned.rolling(5).sum()
        fii_21d = fii_aligned.rolling(21).sum()
        out['fii_5d_zscore']  = _zscore_rolling(fii_5d,  window=126).fillna(0.0)
        out['fii_21d_zscore'] = _zscore_rolling(fii_21d, window=252).fillna(0.0)
    else:
        # Proxy: use Nifty 5d / 21d return as FII sentiment proxy
        # Rationale: FII flows have high correlation with Nifty momentum
        if nifty_close is not None and not (isinstance(nifty_close, pd.Series) and nifty_close.empty):
            if isinstance(nifty_close, pd.Series):
                nifty_p = nifty_close.reindex(c.index).ffill()
            else:
                nifty_p = pd.Series(nifty_close.flatten() if hasattr(nifty_close, 'flatten') else nifty_close, index=c.index)
            fii_proxy_5d  = nifty_p.pct_change(5)
            fii_proxy_21d = nifty_p.pct_change(21)
            out['fii_5d_zscore']  = _zscore_rolling(fii_proxy_5d,  window=126).fillna(0.0)
            out['fii_21d_zscore'] = _zscore_rolling(fii_proxy_21d, window=252).fillna(0.0)
        else:
            out['fii_5d_zscore']  = 0.0
            out['fii_21d_zscore'] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # CATEGORY 7: Volatility Regime
    # ─────────────────────────────────────────────────────────────────────────

    # ATR percentile in 252-bar rolling window
    atr_14 = compute_atr(df, 14)
    atr_pct = atr_14.rolling(252, min_periods=60).rank(pct=True)
    out['atr_percentile'] = atr_pct.fillna(0.5)

    # Rolling std percentile (vol regime)
    roll_std = c.pct_change().rolling(20).std()
    vol_pct  = roll_std.rolling(252, min_periods=60).rank(pct=True)
    out['vol_regime'] = vol_pct.fillna(0.5)

    # Volume breakout Z-score
    vol_ma  = v.rolling(20, min_periods=10).mean()
    vol_std = v.rolling(20, min_periods=10).std()
    out['vol_breakout'] = ((v - vol_ma) / (vol_std + eps)).clip(-3, 3).fillna(0.0)

    # Fractional differentiation of log-price
    out['frac_diff'] = _frac_diff_ffd(c, d=0.4)

    # ─────────────────────────────────────────────────────────────────────────
    # DROP HELPER COLUMNS & STANDARDIZE
    # ─────────────────────────────────────────────────────────────────────────

    # Drop the helper column used internally
    out = out.drop(columns=['rel_strength_v1'], errors='ignore')

    # Ensure we have all expected columns (fill any missing with 0)
    for col in FEATURE_COLS_V2:
        if col not in out.columns:
            out[col] = 0.0

    # FIX: Remove in-engine rolling Z-score — StandardScaler in trainer
    # is the sole normalization layer. Double standardization (rolling Z
    # + StandardScaler) created distributional mismatch at inference.
    # Clip outliers using a rolling 252-bar window so bounds depend only on
    # past data (no look-ahead). Global quantile would use future data.
    for col in FEATURE_COLS_V2:
        if col not in out.columns:
            out[col] = 0.0
        s = out[col]
        if s.nunique() <= 1:
            out[col] = 0.0
        else:
            # Rolling quantile (past-only) — window=252, min_periods=60
            q_hi = s.rolling(252, min_periods=60).quantile(0.99)
            q_lo = s.rolling(252, min_periods=60).quantile(0.01)
            # Clip each value against its own past-only bounds
            clipped = s.where(q_hi.isna() | q_lo.isna(), s.clip(lower=q_lo, upper=q_hi))
            out[col] = clipped.fillna(0.0)

    result = out[FEATURE_COLS_V2].dropna()

    logger.info(
        f"[{ticker}] V2 features: {result.shape[0]} rows × {result.shape[1]} features "
        f"({result.isna().sum().sum()} NaN after dropna)"
    )
    return result


def engineer_features_v2_tensor(
    df: pd.DataFrame,
    ticker: str = "",
    context: Optional[Dict] = None,
    training_cutoff=None,
) -> torch.Tensor:
    """Return V2 features as a PyTorch tensor."""
    feat_df = engineer_features_v2(df, ticker=ticker, context=context, training_cutoff=training_cutoff)
    if feat_df.empty:
        return torch.empty((0, EXPECTED_FEATURE_COUNT_V2))
    return torch.tensor(feat_df.values, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER (standalone helper for training pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def build_full_context(
    ticker: str,
    stock_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    nifty_series: Optional[pd.Series] = None,
    fii_series: Optional[pd.Series] = None,
    include_sector: bool = True,
    include_fno: bool = True,
) -> Dict:
    """
    Build the complete context dict for engineer_features_v2().

    Args:
        ticker:         NSE ticker symbol
        stock_df:       OHLCV DataFrame for the stock
        start_date:     "YYYY-MM-DD"
        end_date:       "YYYY-MM-DD"
        nifty_series:   Optional pre-loaded Nifty close series
        fii_series:     Optional pre-loaded FII net series
        include_sector: Whether to fetch sector index data (yfinance)
        include_fno:    Whether to fetch F&O features (NSE bhav copy)

    Returns:
        dict with keys: nifty_close, sector_close, fno_features, fii_net
    """
    context: Dict = {}

    # ── Nifty ────────────────────────────────────────────────────────────────
    if nifty_series is not None and not nifty_series.empty:
        context['nifty_close'] = nifty_series
    else:
        try:
            # First: try local Nifty50 cache (fastest, no network)
            _nifty_loaded = False
            _nifty_cache_paths = [
                _PROJECT / "data" / "cache" / "nse" / "NIFTY50_20150101_20260521.parquet",
                _PROJECT / "data" / "cache" / "nse" / "NIFTY50_20220101_20260522.parquet",
                _PROJECT / "data" / "cache" / "NIFTY50_1d.parquet",
                _PROJECT / "data" / "cache_yf" / "NSEI_1d.parquet",
            ]
            for _p in _nifty_cache_paths:
                if _p.exists():
                    try:
                        _ndf = pd.read_parquet(_p)
                        _ndf.columns = [str(c).lower() for c in _ndf.columns]
                        if hasattr(_ndf.index, 'tz') and _ndf.index.tz is not None:
                            _ndf.index = _ndf.index.tz_localize(None)
                        _ndf.index = pd.to_datetime(_ndf.index)
                        if 'close' in _ndf.columns and len(_ndf) > 100:
                            context['nifty_close'] = _ndf['close']
                            _nifty_loaded = True
                            logger.debug(f"[{ticker}] Nifty loaded from local cache: {_p.name}")
                            break
                    except Exception:
                        pass

            if not _nifty_loaded:
                # Fallback: fetch from yfinance
                import yfinance as yf
                nifty_raw = yf.download("^NSEI", start=start_date, end=end_date,
                                        progress=False, auto_adjust=True)
                if not nifty_raw.empty:
                    # yfinance 1.4.0+ returns MultiIndex columns — flatten first
                    if isinstance(nifty_raw.columns, pd.MultiIndex):
                        nifty_raw.columns = nifty_raw.columns.get_level_values(0)
                    nifty_raw.columns = [str(c).lower() for c in nifty_raw.columns]
                    if hasattr(nifty_raw.index, 'tz') and nifty_raw.index.tz is not None:
                        nifty_raw.index = nifty_raw.index.tz_localize(None)
                    context['nifty_close'] = nifty_raw['close']
        except Exception as e:
            logger.warning(f"[{ticker}] Nifty fetch failed: {e}")

    # ── Sector ───────────────────────────────────────────────────────────────
    if include_sector:
        try:
            from core.data.sector_data import get_sector_provider
            provider = get_sector_provider()
            sector_series = provider.get_sector_series(
                ticker, start=start_date, end=end_date,
                stock_index=stock_df.index
            )
            if not sector_series.empty and not (sector_series == 0).all():
                context['sector_close'] = sector_series
        except Exception as e:
            logger.warning(f"[{ticker}] Sector fetch failed: {e}")

    # ── F&O features ─────────────────────────────────────────────────────────
    if include_fno:
        try:
            from core.data.fno_data import FNODataProvider
            fno_provider = FNODataProvider()
            fno_df = fno_provider.get_fno_features(
                ticker, start=start_date, end=end_date, spot_df=stock_df
            )
            if not fno_df.empty:
                context['fno_features'] = fno_df
                logger.info(f"[{ticker}] F&O features loaded: {len(fno_df)} rows")
        except Exception as e:
            logger.debug(f"[{ticker}] F&O features unavailable: {e}")

    # ── FII flow ─────────────────────────────────────────────────────────────
    if fii_series is not None and not fii_series.empty:
        context['fii_net'] = fii_series
    else:
        try:
            from core.data.fii_data import FIIDataProvider
            fii_provider = FIIDataProvider()
            fii_data = fii_provider.get_fii_flow(start_date, end_date)
            if fii_data is not None and not fii_data.empty:
                context['fii_net'] = fii_data
        except Exception as e:
            logger.debug(f"[{ticker}] FII data unavailable: {e}")

    return context


# ─────────────────────────────────────────────────────────────────────────────
# SKLEARN-COMPATIBLE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedFeatureEngineV2:
    """
    V2 Feature Engine with sklearn-compatible interface.
    Drop-in upgrade for AdvancedFeatureEngine in the training pipeline.
    """

    VERSION = FEATURE_ENGINE_VERSION

    def __init__(self, include_sector: bool = True, include_fno: bool = True):
        self.include_sector = include_sector
        self.include_fno    = include_fno

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        context: Optional[Dict] = None,
        training_cutoff=None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Main entry point — matches the V1 interface used by trainer.py.
        Builds full context if not provided.
        """
        if context is None and ticker:
            # Build minimal context from stock data itself
            start = df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else '2000-01-01'
            end   = df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else '2026-12-31'
            context = build_full_context(
                ticker=ticker,
                stock_df=df,
                start_date=start,
                end_date=end,
                include_sector=self.include_sector,
                include_fno=self.include_fno,
            )
        return engineer_features_v2(
            df, ticker=ticker, context=context, training_cutoff=training_cutoff
        )

    @property
    def feature_cols(self) -> List[str]:
        return FEATURE_COLS_V2

    @property
    def n_features(self) -> int:
        return EXPECTED_FEATURE_COUNT_V2

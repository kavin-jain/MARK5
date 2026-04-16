"""
MARK5 ADVANCED FEATURE ENGINE v13.0 — DIAMOND SOLID & PURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-08] v13.0: Unified & Decoupled Feature Engineering.
  • Integrated TCN-specific features into the main engine.
  • All transformations are now pure functions (data_df -> Tensor).
  • Implemented local rolling standardization to eliminate global state.
  • Strictly decoupled from model state to ensure zero data leakage.
  • Added engineer_tcn_features_tensor for deep learning pipelines.
- [2026-04-08] v12.5: Decoupled feature engineering from model state.
"""

import numpy as np
import pandas as pd
import logging
import torch
import polars as pl
from typing import Dict, Optional, Tuple, List, Union

# ── CORE CONFIGURATION ───────────────────────────────────────────────────

EXPECTED_FEATURE_COUNT = 8

FEATURE_COLS = [
    'relative_strength_nifty',  # 1: stock vs NIFTY alpha (20d)
    'rsi_14',                   # 2: RSI(14) — Rule 31 mandated
    'dist_52w_high',            # 3: 52-week high proximity — Rule 29
    'post_earnings_drift',      # 4: post-earnings drift flag — Rule 31 mandated
    'gap_significance',         # 5: overnight gap ATR-normalised
    'sector_rel_strength',      # 6: stock vs sector ETF alpha (10d)
    'volume_zscore',            # 7: volume surge z-score vs 60d baseline
    'atr_regime',               # 8: ATR14/ATR50 volatility regime ratio
]

TCN_FEATURE_COLS = [
    'close', 'open', 'high', 'low',  # Standardized OHLC
    'log_ret', 'log_ret_5', 'dist_sma_20', 'rvol', 
    'high_low_ratio', 'natr', 'rsi_14', 'mfi_14', 'force_proxy'
]

# ─────────────────────────────────────────────────────────────────────────
# PURE TRANSFORMATION FUNCTIONS (Stateless)
# ─────────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, span: int = 14) -> pd.Series:
    """ATR(span) via Wilder's smoothing. Pure function."""
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / span, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI(period) — Wilder's smoothing.
    Returns series bounded [-1, +1]. Pure function.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)

def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (MFI). Pure function."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
    neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
    
    mfi = 100 - (100 / (1 + (pos_flow / (neg_flow + 1e-10))))
    return ((mfi - 50.0) / 50.0).clip(-1.0, 1.0)

def standardize_series(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Z-score standardization. Pure function."""
    return ((series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)).clip(-3, 3)

def compute_frac_diff_weights(d: float, thres: float = 1e-4) -> np.ndarray:
    """Compute weights for fractional differentiation using FFD."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def _frac_diff_ffd_vectorized(series: Union[pd.Series, np.ndarray], d: float, thres: float = 1e-4) -> np.ndarray:
    """Vectorized fractional differentiation using np.convolve."""
    if isinstance(series, pd.Series):
        s = series.dropna().values
    else:
        s = series
    
    w = compute_frac_diff_weights(d, thres)
    if len(s) < len(w):
        return np.full(len(s), np.nan)
    
    # Vectorized convolution
    res = np.convolve(s, w, mode='valid')
    
    # Align output with input length (pad with NaNs)
    out = np.full(len(s), np.nan)
    out[len(w)-1:] = res
    return out

def compute_volatility_clustering(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Polars-optimized volatility clustering feature.
    Measures the persistence of log-return variance.
    """
    # Convert to Polars
    pl_df = pl.from_pandas(df.reset_index())
    
    # Calculate log returns and clustering metric
    pl_df = pl_df.with_columns([
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_ret")
    ]).with_columns([
        pl.col("log_ret").abs().rolling_mean(window).alias("vol_clust")
    ])
    
    # Return as pandas Series aligned with input
    return pd.Series(pl_df["vol_clust"].to_numpy(), index=df.index)

# ─────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING ENGINES
# ─────────────────────────────────────────────────────────────────────────

def engineer_features_df(df: pd.DataFrame, context: Dict = None, is_daily: bool = False) -> pd.DataFrame:
    """
    Pure transformation: pd.DataFrame -> pd.DataFrame (8 core features).
    Strictly complies with Rule 31. No internal state.
    """
    if df is None or len(df) < 50: return pd.DataFrame()
    
    df = df.copy()
    context = context or {}

    if df.index.tz is not None:
         df.index = df.index.tz_localize(None)

    # F1: Relative Strength vs NIFTY
    stock_ret_20 = df['close'].pct_change(20)
    nifty_close  = context.get('nifty_close')
    if nifty_close is not None and len(nifty_close) > 20:
        if nifty_close.index.tz is not None:
            nifty_close = nifty_close.copy()
            nifty_close.index = nifty_close.index.tz_localize(None)
        nifty_aligned = nifty_close.reindex(df.index, method='ffill')
        nifty_ret_20  = nifty_aligned.pct_change(20).shift(1)
        df['relative_strength_nifty'] = stock_ret_20 - nifty_ret_20
    else:
        df['relative_strength_nifty'] = stock_ret_20

    atr_short = compute_atr(df, span=14)
    
    # F2: RSI(14)
    df['rsi_14'] = compute_rsi(df['close'], period=14)

    # F3: Distance from 52-Week High
    rolling_high_252 = df['high'].rolling(252, min_periods=200).max()
    df['dist_52w_high'] = ((rolling_high_252 - df['close']) / (rolling_high_252 + 1e-9)).clip(0, 1)

    # F4: Post-Earnings Drift Flag
    vol_20d_avg = df['volume'].rolling(20, min_periods=10).mean()
    volume_surge = df['volume'] / (vol_20d_avg + 1e-9)
    gap_raw  = df['open'] - df['close'].shift(1)
    gap_pct  = gap_raw / (df['close'].shift(1) + 1e-9)
    is_event = (volume_surge >= 1.5) & (gap_raw.abs() >= 0.8 * atr_short)
    event_direction = np.sign(gap_pct)
    event_signal = pd.Series(np.where(is_event, event_direction, 0.0), index=df.index)
    drift_values = np.zeros(len(df))
    for i, val in enumerate(event_signal.values):
        if val != 0:
            end_i = min(i + 5, len(df))
            drift_values[i:end_i] = val
    df['post_earnings_drift'] = pd.Series(drift_values, index=df.index).fillna(0.0)

    # F5: Gap Significance
    gap = df['open'] - df['close'].shift(1)
    df['gap_significance'] = (gap / (atr_short + 1e-9)).clip(-3, 3)

    # F6: Sector Relative Strength
    stock_ret_10  = df['close'].pct_change(10)
    sector_close  = context.get('sector_etf_close')
    if sector_close is not None and len(sector_close) > 10:
        if sector_close.index.tz is not None:
            sector_close = sector_close.copy()
            sector_close.index = sector_close.index.tz_localize(None)
        sector_aligned = sector_close.reindex(df.index, method='ffill')
        sector_ret_10  = sector_aligned.pct_change(10).shift(1)
        df['sector_rel_strength'] = stock_ret_10 - sector_ret_10
    else:
        df['sector_rel_strength'] = stock_ret_10

    # F7: Volume Z-Score
    df['volume_zscore'] = standardize_series(df['volume'], window=60)

    # F8: ATR Regime
    atr50_series = compute_atr(df, span=50)
    df['atr_regime'] = (atr_short / (atr50_series + 1e-9)).clip(0.2, 5.0)

    result = df[FEATURE_COLS].copy()
    result = result.dropna(thresh=4)
    fill_values = {
        'relative_strength_nifty': 0.0, 'rsi_14': 0.0, 'dist_52w_high': 0.5,
        'post_earnings_drift': 0.0, 'gap_significance': 0.0, 'sector_rel_strength': 0.0,
        'volume_zscore': 0.0, 'atr_regime': 1.0,
    }
    result = result.fillna(value=fill_values)
    if (~np.isfinite(result.values)).any():
        result = result.replace([np.inf, -np.inf], 0.0)
    return result

def engineer_tcn_features_df(df: pd.DataFrame, context: Dict = None) -> pd.DataFrame:
    """
    Pure transformation for TCN: pd.DataFrame -> pd.DataFrame (13 features).
    Includes standardized OHLC and deep learning optimized features.
    """
    if df is None or len(df) < 64: return pd.DataFrame()
    
    data = df.copy()
    epsilon = 1e-8

    # 1. Log Returns
    data['log_ret'] = np.log(data['close'] / (data['close'].shift(1) + epsilon))
    data['log_ret_5'] = np.log(data['close'] / (data['close'].shift(5) + epsilon))
    
    # 2. Volatility Normalized Momentum
    roll_std = data['close'].rolling(20).std()
    data['dist_sma_20'] = (data['close'] - data['close'].rolling(20).mean()) / (roll_std + epsilon)
    
    # 3. Relative Volume
    roll_vol = data['volume'].rolling(20).mean()
    data['rvol'] = data['volume'] / (roll_vol + epsilon)
    
    # 4. High-Low Ratio
    data['high_low_ratio'] = (data['high'] - data['low']) / (data['close'] + epsilon)
    
    # 5. Normalized ATR
    atr_14 = compute_atr(data, span=14)
    data['natr'] = atr_14 / (data['close'] + epsilon)

    # 6. Oscillators
    data['rsi_14'] = compute_rsi(data['close'], period=14)
    data['mfi_14'] = compute_mfi(data, period=14)
    
    # 7. Interaction
    data['force_proxy'] = data['log_ret'] * data['rvol']

    # 8. Standardize OHLC (Local normalization for stationarity)
    for col in ['close', 'open', 'high', 'low']:
        data[col] = standardize_series(data[col], window=60)
    
    # Standardization of derived features
    derived_cols = ['log_ret', 'log_ret_5', 'dist_sma_20', 'rvol', 'high_low_ratio', 'natr', 'force_proxy']
    for col in derived_cols:
        data[col] = standardize_series(data[col], window=60)

    result = data[TCN_FEATURE_COLS].dropna()
    return result

def engineer_features_tensor(df: pd.DataFrame, context: Dict = None, is_daily: bool = False) -> torch.Tensor:
    """Standard 8-feature tensor."""
    df_features = engineer_features_df(df, context, is_daily)
    if df_features.empty: return torch.empty((0, EXPECTED_FEATURE_COUNT))
    return torch.tensor(df_features.values, dtype=torch.float32)

def engineer_tcn_features_tensor(df: pd.DataFrame, context: Dict = None) -> torch.Tensor:
    """TCN 13-feature tensor."""
    df_features = engineer_tcn_features_df(df, context)
    if df_features.empty: return torch.empty((0, 13))
    return torch.tensor(df_features.values, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────
# BACKWARD COMPATIBILITY WRAPPER
# ─────────────────────────────────────────────────────────────────────────

class AdvancedFeatureEngine:
    def __init__(self, is_daily: bool = False):
        self.logger = logging.getLogger("MARK5.Features")
        self.is_daily = is_daily
        self._warned_features = set()

    def _frac_diff_ffd(self, series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
        """Wrapper for vectorized fractional differentiation."""
        result = _frac_diff_ffd_vectorized(series, d, thres)
        return pd.Series(result, index=series.index)

    def engineer_all_features(self, df: pd.DataFrame, ticker: str = "", context: Dict = None, is_daily: bool = None, training_cutoff: pd.Timestamp = None) -> pd.DataFrame:
        _is_daily = is_daily if is_daily is not None else self.is_daily
        result = engineer_features_df(df, context, _is_daily)
        if result.empty: return result
        
        nan_fracs = result.isnull().mean()
        for col, frac in nan_fracs.items():
            if frac > 0.20 and col not in self._warned_features:
                self.logger.warning(f"Feature '{col}' has {frac:.1%} NaN pre-fillna")
                self._warned_features.add(col)

        if result.shape[1] != EXPECTED_FEATURE_COUNT:
            for missing_col in FEATURE_COLS:
                if missing_col not in result.columns: result[missing_col] = 0.0
            result = result[FEATURE_COLS]
        return result

    def get_wick_confirmed_entry(self, df: pd.DataFrame) -> pd.Series:
        if df.empty or len(df) < 11: return pd.Series(False, index=df.index)
        candle_range = df['high'] - df['low'] + 1e-9
        body = (df['close'] - df['open']).abs()
        body_bottom = df[['open', 'close']].min(axis=1)
        lower_wick = (body_bottom - df['low']).clip(lower=0)
        bullish_rejection = (lower_wick / candle_range > 0.45) & (body / candle_range < 0.30) & (df['close'] > df['open'])
        vol_10ma = df['volume'].rolling(10, min_periods=5).mean()
        wick_confirmed = bullish_rejection.shift(1).fillna(False) & (df['close'] > df['open']) & (df['volume'] > vol_10ma)
        return wick_confirmed.fillna(False)

"""
MARK5 GOLDEN FEATURE ENGINE v16.0 — L99 PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-23] v16.0: Full Module Integrity.
  • Fixed missing compute_rsi() NameError.
  • Linearized function definitions for top-down reliability.
  • Implemented Golden 8 institutional features with full Z-scoring.
"""

import numpy as np
import pandas as pd
import logging
import torch
from typing import Dict, List, Union

logger = logging.getLogger("MARK5.GoldenFeatureEngine")

# ── CORE CONFIGURATION ───────────────────────────────────────────────────

FEATURE_COLS = [
    'amihud_ratio',     # 1: IC +0.0904
    'range_z',          # 2: IC +0.0840
    'bb_width',         # 3: IC +0.0589
    'atr_vol',          # 4: IC +0.0537
    'rsi_14',           # 5: RSI 14-period
    'gap_sig',          # 6: IC +0.0378
    'vol_adj_mom',      # 7: NEW - IC +0.0342
    'mfi_div',          # 8: NEW - IC +0.0278
    'rel_strength',     # 9: NEW - IC +0.0209
    'tii_60',           # 10: NEW - IC -0.0161 (Mean reversion signal)
]

EXPECTED_FEATURE_COUNT = 10

# ─────────────────────────────────────────────────────────────────────────
# MATH UTILITIES (Pure Functions)
# ─────────────────────────────────────────────────────────────────────────

def standardize_series(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Z-score standardization."""
    return ((series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-9)).clip(-3, 3)

def compute_atr(df: pd.DataFrame, span: int = 14) -> pd.Series:
    """ATR calculation."""
    if 'high' not in df.columns: return df['close'].pct_change().rolling(span).std()
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / span, adjust=False).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Institutional RSI: [0, 1] scale."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0/period, adjust=False).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi / 100.0

def _frac_diff_ffd_vectorized(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """Vectorized fractional differentiation."""
    s = series.ffill().dropna().values
    if len(s) == 0: return series
    w = [1.0]
    for k in range(1, len(s)):
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres: break
        w.append(w_k)
    w = np.array(w[::-1])
    if len(s) < len(w): return pd.Series(0.0, index=series.index)
    res = np.convolve(s, w, mode='valid')
    out = pd.Series(np.nan, index=series.index)
    out.iloc[len(series) - len(res):] = res
    return out.ffill().fillna(0.0)

# ─────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING ENGINES
# ─────────────────────────────────────────────────────────────────────────

# ── LEAKAGE ISOLATION CONTEXT ───────────────────────────────────────────
# This global state allows the feature engine to be aware of test blocks 
# even when called from legacy tests that don't pass test_indices.
_LEAKAGE_TEST_INDICES = None

def set_leakage_isolation(indices: Union[np.ndarray, List[int], None]):
    global _LEAKAGE_TEST_INDICES
    _LEAKAGE_TEST_INDICES = indices

def engineer_features_df(df: pd.DataFrame, context: Dict = None, is_daily: bool = False, training_cutoff=None, test_indices=None) -> pd.DataFrame:
    """MARK6 Refined Feature Transformation."""
    if df is None or len(df) < 200: return pd.DataFrame()
    
    df = df.copy()
    
    # Apply training cutoff if provided (BUG-1)
    if training_cutoff is not None:
        df = df[df.index <= training_cutoff].copy()

    # Use explicitly passed test_indices or the global isolation context
    effective_test_indices = test_indices if test_indices is not None else _LEAKAGE_TEST_INDICES

    if effective_test_indices is not None:
        # Mask test indices to prevent rolling leakage into subsequent training samples.
        # By setting to NaN, any rolling window crossing these indices will correctly 
        # result in NaN, effectively isolating the training blocks.
        try:
            df.iloc[effective_test_indices] = np.nan
        except Exception as e:
            logger.warning(f"Leakage isolation failed: {e}")

    c = df['close']
    h = df.get('high', c)
    l = df.get('low', c)
    v = df.get('volume', pd.Series(1.0, index=df.index))
    tp = (h + l + c) / 3
    epsilon = 1e-9

    # 1. Amihud Ratio (Microstructure)
    df['amihud_ratio'] = (c.pct_change().abs() / (v * c + epsilon)).rolling(20).median() * 1e9

    # 2. Range-Z (Intraday Volatility)
    df['range_z'] = (h - l) / (c + epsilon)

    # 3. BB Width (Volatility Squeeze)
    sma_20 = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    df['bb_width'] = (4 * std_20) / (sma_20 + epsilon)

    # 4. ATR Vol (Institutional Normalization)
    df['atr_vol'] = compute_atr(df, 14) / (c + epsilon)

    # 5. RSI 14-period
    df['rsi_14'] = compute_rsi(c, period=14)

    # 6. Gap Significance (Overnight Footprint)
    df['gap_sig'] = (df['open'] - c.shift(1)) / (c.shift(1) + epsilon)

    # 7. Vol-Adjusted Momentum (MARK6)
    df['vol_adj_mom'] = c.pct_change(20) / (c.pct_change().rolling(20).std() * np.sqrt(20) + epsilon)

    # 8. MFI Divergence (MARK6)
    mfi_period = 14
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(mfi_period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(mfi_period).sum()
    mfi = 100 - (100 / (1 + pos_mf / (neg_mf + epsilon)))
    df['mfi_div'] = c.pct_change(mfi_period) - (mfi.pct_change(mfi_period) / 100.0)

    # 9. Relative Strength vs Nifty (MARK6)
    if context and 'nifty_close' in context and context['nifty_close'] is not None:
        n_close = context['nifty_close']
        # Robust conversion to Series (Fix for ndarray type variance)
        if not hasattr(n_close, 'reindex'):
            import pandas as _pd
            if len(n_close) == len(df):
                n_close = _pd.Series(n_close, index=df.index)
            else:
                n_close = _pd.Series(n_close)

        if not n_close.empty:
            n_close_aligned = n_close.reindex(c.index).ffill()
            # Fallback if reindex produced all NaNs (e.g. date mismatch)
            if n_close_aligned.isna().all():
                df['rel_strength'] = 0.0
            else:
                df['rel_strength'] = (c.pct_change(10) - n_close_aligned.pct_change(10)).fillna(0.0)
        else:
            df['rel_strength'] = 0.0
    else:
        df['rel_strength'] = 0.0

    # 10. Trend Intensity Index (MARK6)
    ma_60 = c.rolling(60).mean()
    dev = c - ma_60
    pos_dev = dev.clip(lower=0).rolling(60).sum()
    neg_dev = (-dev).clip(lower=0).rolling(60).sum()
    df['tii_60'] = (pos_dev / (pos_dev + neg_dev + epsilon)).fillna(0.5)

    # Final standardization for booster optimization
    for col in FEATURE_COLS:
        # Optimization: if column is constant, standardization should return 0.0
        if df[col].nunique() <= 1:
            df[col] = 0.0
        else:
            df[col] = standardize_series(df[col], window=60)

    logger.info(f"engineer_features_df: shape before dropna: {df[FEATURE_COLS].shape}")
    if not df[FEATURE_COLS].empty:
        logger.info(f"engineer_features_df: NaNs per column:\n{df[FEATURE_COLS].isna().sum()}")
    
    return df[FEATURE_COLS].dropna()

def engineer_tcn_features_df(df: pd.DataFrame, context: Dict = None) -> pd.DataFrame:
    """TCN Optimized Features (Sequence learning)."""
    if df is None or len(df) < 64: return pd.DataFrame()
    data = df.copy()
    epsilon = 1e-8
    
    data['log_ret'] = np.log(data['close'] / (data['close'].shift(1) + epsilon))
    data['log_ret_5'] = np.log(data['close'] / (data['close'].shift(5) + epsilon))
    roll_std = data['close'].rolling(20).std()
    data['dist_sma_20'] = (data['close'] - data['close'].rolling(20).mean()) / (roll_std + epsilon)
    roll_vol = data['volume'].rolling(20).mean()
    data['rvol'] = data['volume'] / (roll_vol + epsilon)
    data['high_low_ratio'] = (data['high'] - data['low']) / (data['close'] + epsilon)
    data['natr'] = compute_atr(data, 14) / (data['close'] + epsilon)
    data['rsi_14'] = compute_rsi(data['close'], period=14)
    
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    data['mfi_14'] = pos_flow / (neg_flow + 1e-10)
    data['force_proxy'] = data['log_ret'] * data['rvol']

    for col in ['close', 'open', 'high', 'low']:
        data[col] = standardize_series(data[col], window=60)
    
    for col in ['log_ret', 'log_ret_5', 'dist_sma_20', 'rvol', 'high_low_ratio', 'natr', 'force_proxy']:
        data[col] = standardize_series(data[col], window=60)

    return data[TCN_FEATURE_COLS].dropna()

def engineer_features_tensor(df: pd.DataFrame, context: Dict = None, is_daily: bool = False) -> torch.Tensor:
    df_features = engineer_features_df(df, context, is_daily)
    if df_features.empty: return torch.empty((0, EXPECTED_FEATURE_COUNT))
    return torch.tensor(df_features.values, dtype=torch.float32)

def engineer_tcn_features_tensor(df: pd.DataFrame, context: Dict = None) -> torch.Tensor:
    df_features = engineer_tcn_features_df(df, context)
    if df_features.empty: return torch.empty((0, 13))
    return torch.tensor(df_features.values, dtype=torch.float32)

class AdvancedFeatureEngine:
    def __init__(self, is_daily: bool = False):
        self.is_daily = is_daily

    def engineer_all_features(self, df: pd.DataFrame, ticker: str = "", context: Dict = None, **kwargs) -> pd.DataFrame:
        return engineer_features_df(df, context, self.is_daily, **kwargs)

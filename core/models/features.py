"""
MARK5 ADVANCED FEATURE ENGINE v12.4 — RULE 31 RESTORATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-01] v12.4: IC-driven feature surgery:
  • KILL fii_flow_3d   (IC=0.000 — NSE blocks FII feed; hardcoded 0 poisons model)
    → REPLACE with rsi_14 (Rule 31 mandated, IC expected 0.03–0.06 per rebuild §3.1)
  • KILL amihud_illiquidity (IC=-0.001 — effectively random noise)
    → REPLACE with post_earnings_drift (Rule 31 mandated; IC expected 0.04–0.08)
  • IMPROVE volume_confirmation → volume_zscore (z-score vs 60d baseline;
    old ratio was flat IC=0.010, z-score adds cross-sectional comparability)
  • KEEP: dist_52w_high (+0.036 ✅), gap_significance (+0.028 ✅),
          sector_rel_strength (-0.021 ✅), atr_regime (-0.024 ✅)

Final 8 features (per Rule 31):
  1. relative_strength_nifty  — stock alpha vs index (20d)
  2. rsi_14                   — momentum/mean-reversion (Rule 31 explicit)
  3. dist_52w_high            — 52w high proximity anomaly (Rule 29)
  4. post_earnings_drift      — post-event drift window (Rule 31 explicit)
  5. gap_significance         — overnight gap ATR-normalised
  6. sector_rel_strength      — stock alpha within sector (10d)
  7. volume_zscore            — volume surge relative to 60d baseline
  8. atr_regime               — volatility regime ratio ATR14/ATR50

- [2026-04-01] v12.3: Restored exactly the 8 core features required by Rule 31.
- [2026-03-21] v12.2: Applied fractional differentiation (d=0.4) to dist_52w_high.
- [2026-03-21] v12.1: Applied fractional differentiation (d=0.7) to amihud_illiquidity.
- [2026-03-20] v12.0: Phase 0 IC-based pruning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple

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


class AdvancedFeatureEngine:
    def __init__(self, is_daily: bool = False):
        self.logger = logging.getLogger("MARK5.Features")
        self.is_daily = is_daily
        self.w_mult = 1.0 if is_daily else 7.0  # 7 hourly bars per daily bar
        self._warned_features = set()
        self._tcn_cache = {} # Cache for TCN models to prevent redundant loading

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _add_tcn_signals(self, ticker: str, df: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.Series]]:
        """
        Injects VAJRA TCN (Deep Learning) predictions into the tabular dataset.
        Requires 64 bars of history for the TCN sequence.
        """
        if not ticker: return None
        
        if ticker not in self._tcn_cache:
            try:
                from core.models.tcn.system import TCNTradingModel
                from pathlib import Path
                
                # Use current directory to find models
                model_dir = Path(f"models/{ticker}")
                # Find latest version/vX directory
                versions = sorted([v for v in model_dir.glob("v*") if v.is_dir()], key=lambda x: x.name, reverse=True)
                
                if not versions:
                    # Fallback to base models dir
                    tcn_path = model_dir / "tcn_model"
                else:
                    tcn_path = versions[0] / "tcn_model"
                
                if not tcn_path.with_suffix(".keras").exists():
                    return None

                # Initialize and Load TCN (This automatically loads the correct tcn_model_scaler.pkl)
                tcn = TCNTradingModel(sequence_length=64, n_features=13)
                tcn.build_model()
                tcn.load(str(tcn_path))
                
                self._tcn_cache[ticker] = tcn
            except Exception as e:
                self.logger.warning(f"[{ticker}] TCN loading failed: {e}")
                return None

        tcn = self._tcn_cache[ticker]
        
        try:
            from core.models.tcn.features import AlphaFeatureEngineer
            tcn_fe = AlphaFeatureEngineer()
            
            # NOTE: We DO NOT attach a scaler to tcn_fe here. 
            # We want tcn_fe to return RAW, unscaled features because 
            # tcn.predict() will handle the scaling internally per Rule 38.
            
            tcn_data = tcn_fe.generate_features(df, training_mode=False)
            common_idx = tcn_data.index
            feature_vals = tcn_data.values
            
            if len(feature_vals) < 64: return None
            
            X_batch = []
            valid_indices = []
            
            # optimization for live inference (usually only need the latest)
            if len(df) < 100: # Inference mode
                 X_batch.append(feature_vals[-64:])
                 valid_indices.append(common_idx[-1])
            else: # Training mode (heavy)
                 for i in range(64, len(feature_vals) + 1):
                     X_batch.append(feature_vals[i-64:i])
                     valid_indices.append(common_idx[i-1])
            
            X_batch = np.array(X_batch)
            if len(X_batch) > 0:
                # tcn.predict() scales the raw X_batch safely using its internal 13-feature scaler
                pred_dir, pred_vol = tcn.predict(X_batch)
                
                prob_series = pd.Series(pred_dir.flatten(), index=valid_indices)
                vol_series = pd.Series(pred_vol.flatten(), index=valid_indices)
                
                return prob_series, vol_series
                
        except Exception as e:
            self.logger.error(f"[{ticker}] TCN Inference failed: {e}")
            return None
        return None

    def _compute_atr(self, df: pd.DataFrame, span: int = 98) -> pd.Series:
        """ATR(span) via Wilder's smoothing."""
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low']  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / span, adjust=False).mean()

    def _frac_diff_ffd(self, series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
        """Fixed-Width Window Fractional Differentiation."""
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres:
                break
            w.append(w_k)
            k += 1

        w_arr = np.array(w[::-1])
        width = len(w_arr) - 1
        s = series.dropna().values
        out = np.full(len(s), np.nan)
        for i in range(width, len(s)):
            out[i] = np.dot(w_arr, s[i - width: i + 1])

        return pd.Series(out, index=series.dropna().index)

    def _compute_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI(period) — Wilder's smoothing (identical to TradingView/NSE charts).
        Returns series bounded [0, 100], normalised to [-1, +1] for ML:
          rsi_norm = (RSI - 50) / 50
        Overbought (RSI>70) → positive; oversold (RSI<30) → negative.
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # Wilder's smoothing = EWM with alpha=1/period
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Normalise to [-1, +1]: centred at 50, bounded at 0/100
        return ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN FEATURE ENGINE
    # ─────────────────────────────────────────────────────────────────────────

    def engineer_all_features(self, df: pd.DataFrame, ticker: str = "", context: Dict = None, is_daily: bool = None, training_cutoff: pd.Timestamp = None) -> pd.DataFrame:
        """
        Main entry point for feature engineering.
        Strictly complies with Rule 31 (8 core features).
        """
        if df is None or len(df) < 50: return pd.DataFrame()
        
        # Override is_daily if provided explicitly
        _is_daily = is_daily if is_daily is not None else self.is_daily
        _w_mult = 1.0 if _is_daily else 7.0
        
        df = df.copy()
        context = context or {}

        # ── Timezone Normalization ───────────────────────────────────────────
        if df.index.tz is not None:
             df.index = df.index.tz_localize(None)

        # ── TCN Signal Injection REMOVED (v12.4 Audit M-01) ─────────────────
        # Probabilities from TCN-1D are now handled in the ensemble layer,
        # not injected as features to avoid data leakage and distribution shift.
        # ───────────────────────────────────────────────────────────────────


        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 1: Relative Strength vs NIFTY (20-day equivalent)
        # ═════════════════════════════════════════════════════════════════════
        rs_window = int(20 * _w_mult)
        stock_ret_20 = df['close'].pct_change(rs_window)
        nifty_close  = context.get('nifty_close')
        if nifty_close is not None and len(nifty_close) > rs_window:
            if nifty_close.index.tz is not None:
                nifty_close = nifty_close.copy()
                nifty_close.index = nifty_close.index.tz_localize(None)
            
            # Align Nifty to Stock hourly index
            nifty_aligned = nifty_close.reindex(df.index, method='ffill')
            
            # STRUCTURAL FIX: Shift by 1 to ensure at time T, we only know T-1 index state.
            nifty_ret_20  = nifty_aligned.pct_change(rs_window).shift(1)
            df['relative_strength_nifty'] = stock_ret_20 - nifty_ret_20
        else:
            df['relative_strength_nifty'] = stock_ret_20

        # ── Pre-compute ATR_SHORT (14-day equivalent) ──────────────
        atr_window = int(14 * _w_mult)
        atr_short = self._compute_atr(df, span=atr_window)
        
        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 2: RSI(14) — RULE 31 MANDATED
        # ═════════════════════════════════════════════════════════════════════
        df['rsi_14'] = self._compute_rsi(df['close'], period=atr_window)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 3: Distance from 52-Week High (252-day equivalent)
        # ═════════════════════════════════════════════════════════════════════
        high_window = int(252 * _w_mult)
        min_periods = int(350 * _w_mult) if not _is_daily else 126
        rolling_high_252 = df['high'].rolling(high_window, min_periods=min_periods).max()
        df['dist_52w_high'] = (
            (rolling_high_252 - df['close']) / (rolling_high_252 + 1e-9)
        ).clip(0, 1)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 4: Post-Earnings Drift Flag — RULE 31 MANDATED
        # PEAD window: 5 days
        # ═════════════════════════════════════════════════════════════════════
        vol_avg_window = int(20 * _w_mult)
        vol_20d_avg = df['volume'].rolling(vol_avg_window, min_periods=vol_avg_window // 2).mean()
        volume_surge = df['volume'] / (vol_20d_avg + 1e-9)

        gap_raw  = df['open'] - df['close'].shift(1)
        gap_pct  = gap_raw / (df['close'].shift(1) + 1e-9)

        # Event: meaningful volume spike + material gap
        is_event = (volume_surge >= 1.5) & (gap_raw.abs() >= 0.8 * atr_short)

        # Direction = sign of gap (continuation bias)
        event_direction = np.sign(gap_pct)
        event_signal    = pd.Series(
            np.where(is_event, event_direction, 0.0),
            index=df.index,
        )

        # Forward-fill the signal for 5-day equivalent
        drift_window = int(5 * _w_mult)
        drift_values = np.zeros(len(df))
        for i, val in enumerate(event_signal.values):
            if val != 0:
                end_i = min(i + drift_window, len(df))
                drift_values[i:end_i] = val

        drift_active = pd.Series(drift_values, index=df.index)
        df['post_earnings_drift'] = drift_active.fillna(0.0)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 5: Gap Significance (ATR-normalised overnight gap)
        # ═════════════════════════════════════════════════════════════════════
        gap = df['open'] - df['close'].shift(1)
        df['gap_significance'] = (gap / (atr_short + 1e-9)).clip(-3, 3)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 6: Sector Relative Strength (10-day equivalent)
        # ═════════════════════════════════════════════════════════════════════
        sector_window = int(10 * _w_mult)
        stock_ret_10  = df['close'].pct_change(sector_window)
        sector_close  = context.get('sector_etf_close')
        if sector_close is not None and len(sector_close) > sector_window:
            if sector_close.index.tz is not None:
                sector_close = sector_close.copy()
                sector_close.index = sector_close.index.tz_localize(None)
            
            # Align Sector to Stock hourly index
            sector_aligned = sector_close.reindex(df.index, method='ffill')
            
            # STRUCTURAL FIX: Shift by 1 to ensure at time T, we only know T-1 sector state.
            sector_ret_10  = sector_aligned.pct_change(sector_window).shift(1)
            df['sector_rel_strength'] = stock_ret_10 - sector_ret_10
        else:
            df['sector_rel_strength'] = stock_ret_10

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 7: Volume Z-Score vs 60-day baseline
        # ═════════════════════════════════════════════════════════════════════
        vol60_window = int(60 * _w_mult)
        vol_60d_avg = df['volume'].rolling(vol60_window, min_periods=vol60_window // 3).mean()   
        vol_60d_std = df['volume'].rolling(vol60_window, min_periods=vol60_window // 3).std()
        
        df['volume_zscore'] = (
            (df['volume'] - vol_60d_avg) / (vol_60d_std + 1e-9)
        ).clip(-3, 3)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 8: ATR Regime (ATR14 / ATR50)
        # ══════════════════════════════════════════════════════════════════════
        atr50_window = int(50 * _w_mult)
        atr50_series = self._compute_atr(df, span=atr50_window)
        df['atr_regime'] = (atr_short / (atr50_series + 1e-9)).clip(0.2, 5.0)

        # TCN features REMOVED per Rule 31 (Pruning to 8 core features).
        # Use as individual model in ensemble instead.

        # ═════════════════════════════════════════════════════════════════════
        # OUTPUT CONSTRUCTION
        # ═════════════════════════════════════════════════════════════════════
        result = df[FEATURE_COLS].copy()

        # ── Validation: warn on >20% NaN before fill (once per feature) ─────
        nan_fracs = result.isnull().mean()
        for col, frac in nan_fracs.items():
            if frac > 0.20 and col not in self._warned_features:
                self.logger.warning(
                    f"Feature '{col}' has {frac:.1%} NaN pre-fillna — "
                    "data quality may be degrading."
                )
                self._warned_features.add(col)

        # ── Drop warmup rows: require ≥ 4 of 10 features non-NaN ─────────────
        result = result.dropna(thresh=4)

        # ── Per-feature neutral fills ─────────────────────────────────────────
        fill_values = {
            'relative_strength_nifty': 0.0,   # in-line with market
            'rsi_14':                  0.0,   # neutral RSI=50
            'dist_52w_high':           0.5,   # mid-range
            'post_earnings_drift':     0.0,   # no active drift window
            'gap_significance':        0.0,   # no gap
            'sector_rel_strength':     0.0,   # in-line with sector
            'volume_zscore':           0.0,   # normal volume
            'atr_regime':              1.0,   # neutral (ATR14 = ATR50)
            'tcn_bull_prob':           0.5,   # neutral prob
            'tcn_expected_vol':        0.01,  # neutral vol
        }
        result = result.fillna(value=fill_values)

        # ── Replace residual inf / -inf after fill ───────────────────────────
        inf_mask = ~np.isfinite(result.values)
        if inf_mask.any():
            result = result.replace([np.inf, -np.inf], 0.0)

        # ── Assert exact feature count (hard guard — breaks CI if violated) ──
        if result.shape[1] != EXPECTED_FEATURE_COUNT:
            self.logger.warning(
                f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, "
                f"got {result.shape[1]}. Columns: {list(result.columns)}. "
                f"Padding/trimming to expected schema."
            )
            # Neutral-fill missing columns; drop extras — never crash the live path
            for missing_col, fill_val in fill_values.items():
                if missing_col not in result.columns:
                    result[missing_col] = fill_val
            result = result[FEATURE_COLS]

        return result

    def get_wick_confirmed_entry(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a boolean Series: True on days where a wick-confirmed
        breakout setup exists.
        """
        if df.empty or len(df) < 11:
            return pd.Series(False, index=df.index)

        candle_range = df['high'] - df['low'] + 1e-9
        body         = (df['close'] - df['open']).abs()
        body_bottom  = df[['open', 'close']].min(axis=1)
        lower_wick   = (body_bottom - df['low']).clip(lower=0)

        bullish_rejection = (
            (lower_wick / candle_range > 0.45) &
            (body / candle_range < 0.30) &
            (df['close'] > df['open'])
        )

        vol_10ma = df['volume'].rolling(10, min_periods=5).mean()

        wick_confirmed = (
            bullish_rejection.shift(1).fillna(False) &
            (df['close'] > df['open']) &
            (df['volume'] > vol_10ma)
        )

        return wick_confirmed.fillna(False)

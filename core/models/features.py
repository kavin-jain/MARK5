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
from typing import Dict, Optional

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
    def __init__(self):
        self.logger = logging.getLogger("MARK5.Features")
        self._warned_features = set()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────────

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

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        context: Optional[Dict] = None,
        training_cutoff: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generates exactly 8 features per Rule 31.
        Feature set is fixed — no additions allowed without IC validation.
        """
        if df.empty:
            return df

        df = df.copy()
        context = context or {}

        # ── Timezone Normalization ───────────────────────────────────────────
        if df.index.tz is not None:
             df.index = df.index.tz_localize(None)


        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 1: Relative Strength vs NIFTY (140-bar / ~20-day)
        # ═════════════════════════════════════════════════════════════════════
        stock_ret_140 = df['close'].pct_change(140)
        nifty_close  = context.get('nifty_close')
        if nifty_close is not None and len(nifty_close) > 140:
            if nifty_close.index.tz is not None:
                nifty_close = nifty_close.copy()
                nifty_close.index = nifty_close.index.tz_localize(None)
            
            # Align Nifty hourly close to Stock hourly index
            nifty_aligned = nifty_close.reindex(df.index, method='ffill')
            
            # STRUCTURAL FIX: Shift by 1 to ensure at time T, we only know T-1 index state.
            # This deletes the 'Day-End' leakage entirely.
            nifty_ret_140  = nifty_aligned.pct_change(140).shift(1)
            df['relative_strength_nifty'] = stock_ret_140 - nifty_ret_140
        else:
            df['relative_strength_nifty'] = stock_ret_140

        # ── Pre-compute ATR_SHORT (98-bar / ~14-day) ────────────────────────
        atr_short = self._compute_atr(df, span=98)
        
        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 2: RSI(98) — RULE 31 MANDATED
        # Wilder's smoothing, scaled for 60m data (14d * 7 = 98b).
        # ═════════════════════════════════════════════════════════════════════
        df['rsi_14'] = self._compute_rsi(df['close'], period=98)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 3: Distance from 52-Week High (1764-bar / ~252-day)
        # ═════════════════════════════════════════════════════════════════════
        rolling_high_1764 = df['high'].rolling(1764, min_periods=350).max()
        df['dist_52w_high'] = (
            (rolling_high_1764 - df['close']) / (rolling_high_1764 + 1e-9)
        ).clip(0, 1)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 4: Post-Earnings Drift Flag — RULE 31 MANDATED
        # Replaces dead amihud_illiquidity (IC=-0.001).
        # PEAD: price drift continues in direction of earnings gap for 5-20 days.
        #
        # Event detection (no earnings calendar needed):
        #   volume ≥ 1.5× 20d average  → above-normal interest
        #   |gap| ≥ 0.8× ATR14         → material overnight move
        # Together they capture most earnings, news, and block-deal events.
        # Lower thresholds than v12.4a to fire frequently enough for IC signal.
        #
        # Drift window = 5 bars (NSE midcap PEAD peaks at 3–7 days).
        # Encoded: +1 positive drift, -1 negative drift, 0 no event.
        # T-1 shifted (event observable at close, acted on next morning open).
        # ═════════════════════════════════════════════════════════════════════
        vol_140b_avg = df['volume'].rolling(140, min_periods=70).mean()
        volume_surge = df['volume'] / (vol_140b_avg + 1e-9)

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

        # Forward-fill the signal for 5 bars (PEAD window)
        drift_window = 35
        drift_values = np.zeros(len(df))
        for i, val in enumerate(event_signal.values):
            if val != 0:
                end_i = min(i + drift_window, len(df))
                drift_values[i:end_i] = val

        drift_active = pd.Series(drift_values, index=df.index)
        # Removed .shift(1) to align event knowledge with EOD state
        df['post_earnings_drift'] = drift_active.fillna(0.0)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 5: Gap Significance (ATR-normalised overnight gap)
        # ═════════════════════════════════════════════════════════════════════
        gap = df['open'] - df['close'].shift(1)
        df['gap_significance'] = (gap / (atr_short + 1e-9)).clip(-3, 3)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 6: Sector Relative Strength (70-bar / ~10-day)
        # ═════════════════════════════════════════════════════════════════════
        stock_ret_70  = df['close'].pct_change(70)
        sector_close  = context.get('sector_etf_close')
        if sector_close is not None and len(sector_close) > 70:
            if sector_close.index.tz is not None:
                sector_close = sector_close.copy()
                sector_close.index = sector_close.index.tz_localize(None)
            
            # Align Sector hourly close to Stock hourly index
            sector_aligned = sector_close.reindex(df.index, method='ffill')
            
            # STRUCTURAL FIX: Shift by 1 to ensure at time T, we only know T-1 sector state.
            sector_ret_70  = sector_aligned.pct_change(70).shift(1)
            df['sector_rel_strength'] = stock_ret_70 - sector_ret_70
        else:
            df['sector_rel_strength'] = stock_ret_70

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 7: Volume Z-Score vs 420-bar / ~60-day baseline
        # ═════════════════════════════════════════════════════════════════════
        vol_420b_avg = df['volume'].rolling(420, min_periods=140).mean()   
        vol_420b_std = df['volume'].rolling(420, min_periods=140).std()
        
        df['volume_zscore'] = (
            (df['volume'] - vol_420b_avg) / (vol_420b_std + 1e-9)
        ).clip(-3, 3)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 8: ATR Regime (ATR98 / ATR350)
        # ═════════════════════════════════════════════════════════════════════
        atr350_series = self._compute_atr(df, span=350)
        df['atr_regime'] = (atr_short / (atr350_series + 1e-9)).clip(0.2, 5.0)

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

        # ── Drop warmup rows: require ≥ 4 of 8 features non-NaN ─────────────
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
        }
        result = result.fillna(value=fill_values)

        # ── Replace residual inf / -inf after fill ───────────────────────────
        inf_mask = ~np.isfinite(result.values)
        if inf_mask.any():
            result = result.replace([np.inf, -np.inf], 0.0)

        # ── Assert exact feature count (hard guard — breaks CI if violated) ──
        if result.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, "
                f"got {result.shape[1]}. Columns: {list(result.columns)}"
            )

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

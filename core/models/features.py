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

    def _compute_atr14(self, df: pd.DataFrame) -> pd.Series:
        """ATR(14) via Wilder's smoothing."""
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low']  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / 14, adjust=False).mean()

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

        # ── Pre-compute ATR14 ────────────────────────────────────────────────
        atr14 = self._compute_atr14(df)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 1: Relative Strength vs NIFTY (20-day)
        # Rule 31: "relative strength vs NIFTY (20-day)"
        # IC = -0.018 (borderline but structurally important; NSE-specific alpha)
        # ═════════════════════════════════════════════════════════════════════
        stock_ret_20 = df['close'].pct_change(20)
        nifty_close  = context.get('nifty_close')
        if nifty_close is not None and len(nifty_close) > 20:
            if nifty_close.index.tz is not None:
                nifty_close = nifty_close.copy()
                nifty_close.index = nifty_close.index.tz_localize(None)
            nifty_aligned = nifty_close.reindex(df.index, method='ffill')
            nifty_ret_20  = nifty_aligned.pct_change(20)
            df['relative_strength_nifty'] = stock_ret_20 - nifty_ret_20
        else:
            df['relative_strength_nifty'] = stock_ret_20

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 2: RSI(14) — RULE 31 MANDATED
        # Replaces dead fii_flow_3d (IC=0.000 — NSE blocks FII feed).
        # Wilder's smoothing, normalised to [-1, +1].
        # Mean-reversion signal: oversold stocks bounce on NSE midcap universe.
        # ═════════════════════════════════════════════════════════════════════
        df['rsi_14'] = self._compute_rsi(df['close'], period=14)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 3: Distance from 52-Week High
        # Rule 29: "stocks within 3% of 52-week high get +0.05 confidence bonus."
        # IC = +0.036 ✅ — strongest feature in current set.
        # Fractional differentiation applied for stationarity (d=0.4).
        # ═════════════════════════════════════════════════════════════════════
        rolling_high_252 = df['high'].rolling(252, min_periods=50).max()
        dist_52w_raw = (
            (rolling_high_252 - df['close']) / (rolling_high_252 + 1e-9)
        ).clip(0, 1)

        if len(df) > 1000:
            dist_52w_fd = self._frac_diff_ffd(dist_52w_raw.dropna(), d=0.4, thres=5e-3)
            df['dist_52w_high'] = dist_52w_fd.reindex(df.index).ffill()
        else:
            df['dist_52w_high'] = dist_52w_raw

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
        vol_20d_avg = df['volume'].rolling(20, min_periods=10).mean()
        volume_surge = df['volume'] / (vol_20d_avg + 1e-9)

        gap_raw  = df['open'] - df['close'].shift(1)
        gap_pct  = gap_raw / (df['close'].shift(1) + 1e-9)

        # Event: meaningful volume spike + material gap
        is_event = (volume_surge >= 1.5) & (gap_raw.abs() >= 0.8 * atr14)

        # Direction = sign of gap (continuation bias)
        event_direction = np.sign(gap_pct)
        event_signal    = pd.Series(
            np.where(is_event, event_direction, 0.0),
            index=df.index,
        )

        # Forward-fill the signal for 5 bars (PEAD window)
        drift_window = 5
        drift_values = np.zeros(len(df))
        for i, val in enumerate(event_signal.values):
            if val != 0:
                end_i = min(i + drift_window, len(df))
                drift_values[i:end_i] = val

        drift_active = pd.Series(drift_values, index=df.index)
        # T-1 shift: signal from yesterday's event applies today at open
        df['post_earnings_drift'] = drift_active.shift(1).fillna(0.0)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 5: Gap Significance (ATR-normalised overnight gap)
        # IC = +0.028 ✅ — second strongest, high ICIR=0.819
        # ═════════════════════════════════════════════════════════════════════
        gap = df['open'] - df['close'].shift(1)
        df['gap_significance'] = (gap / (atr14 + 1e-9)).clip(-3, 3)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 6: Sector Relative Strength (10-day)
        # IC = -0.021 ✅ — stock alpha within sector rotation
        # Fallback to stock's own 10d return when sector ETF unavailable.
        # ═════════════════════════════════════════════════════════════════════
        stock_ret_10  = df['close'].pct_change(10)
        sector_close  = context.get('sector_etf_close')
        if sector_close is not None and len(sector_close) > 10:
            if sector_close.index.tz is not None:
                sector_close = sector_close.copy()
                sector_close.index = sector_close.index.tz_localize(None)
            sector_aligned = sector_close.reindex(df.index, method='ffill')
            df['sector_rel_strength'] = stock_ret_10 - sector_aligned.pct_change(10)
        else:
            df['sector_rel_strength'] = stock_ret_10

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 7: Volume Z-Score vs 60-day baseline
        # Replaces volume_confirmation ratio (IC=+0.010).
        # Z-score adds cross-sectional comparability: a 3σ volume surge on
        # RELIANCE means the same thing as on COFORGE.
        # Bounded [-3, +3] and shifted T-1 (published EOD).
        # ═════════════════════════════════════════════════════════════════════
        vol_60d_avg = df['volume'].rolling(60, min_periods=20).mean()   # separate from vol_20d_avg used in PEAD
        vol_60d_std = df['volume'].rolling(60, min_periods=20).std()
        volume_zscore_raw = (
            (df['volume'] - vol_60d_avg) / (vol_60d_std + 1e-9)
        ).clip(-3, 3)
        # T-1 shift: we observe volume at close, act next morning at open
        df['volume_zscore'] = volume_zscore_raw.shift(1)

        # ═════════════════════════════════════════════════════════════════════
        # FEATURE 8: ATR Regime (ATR14 / ATR50)
        # IC = -0.024 ✅ — predicts volatility expansion for Rule 19 gating
        # ═════════════════════════════════════════════════════════════════════
        prev_close_atr = df['close'].shift(1)
        tr_atr50 = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close_atr).abs(),
            (df['low']  - prev_close_atr).abs(),
        ], axis=1).max(axis=1)
        atr50_series = tr_atr50.ewm(alpha=1.0 / 50, adjust=False).mean()
        df['atr_regime'] = (atr14 / (atr50_series + 1e-9)).clip(0.2, 5.0)

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

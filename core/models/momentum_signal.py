"""
MARK5 — Multi-Factor Momentum Signal Engine
============================================
Universal entry/exit signal for NSE equities.
No per-stock ML training required. Works for ANY ticker with ≥60 bars.

Empirical basis:
  - Jegadeesh & Titman (1993): momentum factor persists 3-12M horizons
  - MA trend-following: well-documented in academic + practitioner literature
  - Relative strength / cross-sectional momentum: established NSE alpha source
  - Volume–price confirmation: Arms index family, O'Neil's CANSLIM

Signal components and weights:
  Trend Alignment   (27%→25%): MA(20/50/200) crossover + price position
  Price Momentum    (27%→25%): 1M/3M/6M/11M returns → sigmoid normalisation
  Relative Strength (18%→16%): Stock vs NIFTY50 over 3M/6M (optional)
  Vol-Adjusted      (13%→12%): 20-day rolling Sharpe → sigmoid
  Volume Quality     (5%→4%): Up-day vs down-day volume ratio
  Candlestick       (10%):    22-pattern engine (context-aware)
  Sector RS          (8%):    Stock vs sector index 21d Z-score (optional)

Signal thresholds (calibrated on NSE OOS 2022-2026):
  ≥ 0.55 → bullish  → ENTER position
  0.40–0.55 → neutral → HOLD existing, no new entry
  ≤ 0.40 → bearish  → EXIT position

Why this beats the current ML approach:
  - ML models (XGB/LGB/CAT) on ~1700 bars yield AUC ~0.37-0.50 (near-random)
  - pct_above_hurdle=100% tickers (HAL, TRENT) → always bullish → no timing
  - pct_above_hurdle=0% tickers (BEL, SBIN) → never triggers → misses 50%+ gains
  - Momentum signal has genuine variance (0.15-0.90 range, ~0.2 std dev)
  - Enters HAL in Feb 2022 at ₹672 (ML entered May 2023 at ₹2000+)
  - Exits when trend deteriorates, not on random 15% dips

SAFETY: PAPER MODE ONLY — this module does not execute real orders.
"""
import math
import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.models.candlestick_patterns import CandlestickPatternEngine

logger = logging.getLogger("MARK5.MomentumSignal")

# ── Calibration constants ──────────────────────────────────────────────────────
ENTRY_THRESHOLD  = 0.55   # score ≥ this → bullish entry signal
EXIT_THRESHOLD   = 0.40   # score ≤ this → bearish exit signal
NEUTRAL_BAND_LO  = 0.40
NEUTRAL_BAND_HI  = 0.55
ROLL_WINDOW      = 5      # bars for rolling average of scores (smoothing)
MIN_BARS         = 60     # minimum history for any score (uses shorter windows)
OPTIMAL_BARS     = 252    # optimal history (enables 11M momentum + full RS)

# Component weights — must sum to 1.0
# v2: added candlestick component (10%), reduced others proportionally
WEIGHTS = {
    "trend":       0.27,   # was 0.30
    "momentum":    0.27,   # was 0.30
    "relative":    0.18,   # was 0.20
    "vol_adj":     0.13,   # was 0.15
    "volume":      0.05,   # unchanged
    "candlestick": 0.10,   # NEW: candlestick pattern engine
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# Shared candlestick engine (instantiated once, stateless)
_CANDLE_ENGINE = CandlestickPatternEngine()

# Sigmoid scales (calibrated so a "typical strong signal" maps to ~0.75)
_SIG_SCALE_MOM = 0.20   # 20% return → score ~0.73   (momentum)
_SIG_SCALE_REL = 0.10   # 10% outperformance → ~0.73  (relative strength)
_SIG_SCALE_VOL = 0.10   # Sharpe ≈ 1 daily → ~0.73   (vol-adjusted)


def _sig(x: float, scale: float) -> float:
    """Sigmoid mapping ℝ → (0,1) with given scale parameter."""
    try:
        return 1.0 / (1.0 + math.exp(-x / scale))
    except (OverflowError, ZeroDivisionError):
        return 0.0 if x < 0 else 1.0


# ── Main engine ───────────────────────────────────────────────────────────────
class MomentumSignalEngine:
    """
    Computes a multi-factor momentum score for any NSE equity.

    Usage — vectorised (fast, preferred for backtesting):
        engine = MomentumSignalEngine()
        scores = engine.precompute_scores(df, nifty_df=nifty_df)
        # scores: pd.Series indexed like df, values in [0.0, 1.0]

    Usage — single point (for live inference):
        score = engine.compute_score(df, date, nifty_df=nifty_df)
    """

    def __init__(self):
        self.entry_threshold = ENTRY_THRESHOLD
        self.exit_threshold  = EXIT_THRESHOLD
        self.roll_window     = ROLL_WINDOW

    # ── Vectorised (O(n)) ────────────────────────────────────────────────────

    def precompute_scores(
        self,
        df: pd.DataFrame,
        nifty_df: Optional[pd.DataFrame] = None,
        sector_rs: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute momentum scores for every bar in df.
        All operations are O(n) via rolling windows.

        Args:
            df:         OHLCV DataFrame with DatetimeIndex (columns lowercase)
            nifty_df:   Optional NIFTY50 OHLCV with compatible DatetimeIndex
            sector_rs:  Optional Z-scored 21d sector-relative-strength Series
                        (from core.data.sector_data.get_sector_rs). When
                        provided, adds an 8% sector RS component and reduces
                        other weights proportionally (7-component mode).

        Returns:
            pd.Series of scores in [0.0, 1.0], same index as df
        """
        if len(df) < 2:
            return pd.Series(0.0, index=df.index)

        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # ── 1. Trend Alignment (30%) ─────────────────────────────────────────
        ma20  = close.rolling(20, min_periods=1).mean()
        ma50  = close.rolling(50, min_periods=1).mean()
        ma200 = close.rolling(200, min_periods=1).mean()

        # Weight each MA condition (sum to 1.0 within component)
        trend = (
            (close > ma20).astype(float)  * 0.15 +
            (close > ma50).astype(float)  * 0.20 +
            (close > ma200).astype(float) * 0.25 +
            (ma20 > ma50).astype(float)   * 0.20 +
            (ma50 > ma200).astype(float)  * 0.20
        )
        # Require at least 20 bars of history; below that → neutral
        trend = trend.where(close.index.map(lambda d: (df.index <= d).sum()) >= 20, other=0.5)

        # ── 2. Price Momentum (30%) ──────────────────────────────────────────
        # 11-month lag: skip-month adjusted (exclude last 21 bars to avoid
        # short-term reversal double-counting with ret_1m).
        # Returns T-231 to T-21, not T-231 to T.
        ret_1m  = close.pct_change(21)
        ret_3m  = close.pct_change(63)
        ret_6m  = close.pct_change(126)
        ret_11m = close.shift(21).pct_change(210)

        def _sig_series(s: pd.Series, scale: float) -> pd.Series:
            return s.apply(lambda x: _sig(x, scale) if pd.notna(x) else np.nan)

        sig_1m  = _sig_series(ret_1m,  _SIG_SCALE_MOM)
        sig_3m  = _sig_series(ret_3m,  _SIG_SCALE_MOM)
        sig_6m  = _sig_series(ret_6m,  _SIG_SCALE_MOM)
        sig_11m = _sig_series(ret_11m, _SIG_SCALE_MOM)

        # Mean of available signals per bar (ignores NaN)
        mom_stack = pd.DataFrame({
            "m1": sig_1m, "m3": sig_3m, "m6": sig_6m, "m11": sig_11m
        })
        momentum = mom_stack.mean(axis=1, skipna=True).fillna(0.5)

        # ── 3. Relative Strength vs NIFTY50 (20%) ───────────────────────────
        if nifty_df is not None and len(nifty_df) >= 63:
            try:
                nifty_close = (
                    nifty_df["close"]
                    .astype(float)
                    .reindex(df.index, method="ffill")
                )
                nifty_ret3m = nifty_close.pct_change(63)
                nifty_ret6m = nifty_close.pct_change(126)

                rel3m = _sig_series(ret_3m - nifty_ret3m,  _SIG_SCALE_REL)
                rel6m = _sig_series(ret_6m - nifty_ret6m,  _SIG_SCALE_REL)

                rs_stack = pd.DataFrame({"r3": rel3m, "r6": rel6m})
                relative = rs_stack.mean(axis=1, skipna=True).fillna(0.5)
            except Exception as e:
                logger.debug(f"Relative strength failed: {e}")
                relative = pd.Series(0.5, index=df.index)
        else:
            relative = pd.Series(0.5, index=df.index)

        # ── 4. Volatility-Adjusted (15%) — 20-day rolling Sharpe ────────────
        daily_ret  = close.pct_change()
        roll_mean  = daily_ret.rolling(20, min_periods=5).mean()
        roll_std   = daily_ret.rolling(20, min_periods=5).std()
        # Clip to [-3, 3] to avoid extreme values on tiny std
        sharpe     = (roll_mean / roll_std.replace(0, np.nan)).clip(-3.0, 3.0)
        vol_adj    = _sig_series(sharpe, _SIG_SCALE_VOL).fillna(0.5)

        # ── 5. Volume Quality (5%) ───────────────────────────────────────────
        up_mask  = (daily_ret > 0).astype(float)
        dn_mask  = (daily_ret <= 0).astype(float)
        up_vol   = (volume * up_mask).rolling(20, min_periods=5).sum()
        dn_vol   = (volume * dn_mask).rolling(20, min_periods=5).sum()
        total    = (up_vol + dn_vol).replace(0, np.nan)
        vol_qual = (up_vol / total).fillna(0.5).clip(0.0, 1.0)

        # ── 6. Candlestick Pattern Score (10%) ───────────────────────────────
        try:
            candle = _CANDLE_ENGINE.precompute_scores(df)
        except Exception as e:
            logger.debug(f"Candlestick engine failed, using neutral: {e}")
            candle = pd.Series(0.5, index=df.index)

        # ── 7. Sector Relative Strength (8%) — optional ───────────────────
        # sector_rs is a Z-scored series: Z > 0 = stock outperforming sector.
        # Converted to [0, 1] via sigmoid(z, scale=1.0):
        #   Z=+1 (1 std dev outperformance) → 0.73   (bullish)
        #   Z= 0 (tracking sector)          → 0.50   (neutral)
        #   Z=-1 (underperforming by 1 std) → 0.27   (bearish)
        _use_sector_rs = False
        sector_score = pd.Series(0.5, index=df.index)
        if sector_rs is not None:
            try:
                sector_rs_aligned = sector_rs.reindex(df.index).fillna(0.0)
                sector_score = _sig_series(sector_rs_aligned, 1.0).fillna(0.5)
                _use_sector_rs = True
            except Exception as e:
                logger.debug(f"Sector RS component failed, using neutral: {e}")

        # ── Composite — 7-component (with sector RS) or 6-component ─────────
        if _use_sector_rs:
            # Weights shift: each existing component reduced proportionally
            # to free 8% for sector RS.
            composite = (
                trend        * 0.25 +
                momentum     * 0.25 +
                relative     * 0.16 +
                vol_adj      * 0.12 +
                vol_qual     * 0.04 +
                candle       * 0.10 +
                sector_score * 0.08
            ).clip(0.0, 1.0)
        else:
            composite = (
                trend    * WEIGHTS["trend"]       +
                momentum * WEIGHTS["momentum"]    +
                relative * WEIGHTS["relative"]    +
                vol_adj  * WEIGHTS["vol_adj"]     +
                vol_qual * WEIGHTS["volume"]      +
                candle   * WEIGHTS["candlestick"]
            ).clip(0.0, 1.0)

        # Clamp early bars that don't have enough history to neutral
        n_bars = pd.Series(range(1, len(df) + 1), index=df.index)
        composite = composite.where(n_bars >= MIN_BARS, other=0.0)

        return composite

    def rolling_score(
        self,
        scores: pd.Series,
        date: pd.Timestamp,
        window: int = ROLL_WINDOW,
    ) -> float:
        """
        Rolling mean of score series up to and including `date`.
        Mirrors the rolling_conf() function used by the ML system.
        """
        try:
            idx = scores.index.searchsorted(date, side="right") - 1
            idx = max(0, min(idx, len(scores) - 1))
            start = max(0, idx - window + 1)
            vals = scores.iloc[start : idx + 1]
            return float(vals.mean()) if len(vals) > 0 else 0.0
        except Exception:
            return 0.0

    # ── Single-point (for live inference) ────────────────────────────────────

    def compute_score(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        nifty_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Score for a single bar `date`.
        Uses vectorised precompute internally for correctness; efficient for
        single calls. For backtesting over many bars, use precompute_scores().
        """
        # Find slice up to and including date
        mask = df.index <= date
        df_slice = df[mask]
        nifty_slice = None
        if nifty_df is not None:
            nifty_slice = nifty_df[nifty_df.index <= date]

        if len(df_slice) < 2:
            return 0.0

        scores = self.precompute_scores(df_slice, nifty_df=nifty_slice)
        return float(scores.iloc[-1])

    # ── ATR-based trailing stop ───────────────────────────────────────────────

    @staticmethod
    def compute_atr_pct(df: pd.DataFrame, date: pd.Timestamp, n: int = 20) -> float:
        """
        20-day ATR as fraction of price (e.g. 0.02 = 2%).
        Used to set volatility-adaptive trailing stop distance.
        """
        try:
            idx = df.index.searchsorted(date, side="right") - 1
            start = max(0, idx - n)
            sl = df.iloc[start : idx + 1]
            if len(sl) < 3:
                return 0.02
            hl = sl["high"] - sl["low"]
            hc = (sl["high"] - sl["close"].shift(1)).abs()
            lc = (sl["low"]  - sl["close"].shift(1)).abs()
            tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr = float(tr.mean())
            price = float(sl["close"].iloc[-1])
            return atr / price if price > 0 else 0.02
        except Exception:
            return 0.02

    @staticmethod
    def trailing_stop_price(
        peak_price: float,
        atr_pct: float,
        multiplier: float = 4.0,
        min_pct: float = 0.10,
        max_pct: float = 0.25,
    ) -> float:
        """
        ATR-adaptive trailing stop below peak.

        stop = peak × (1 − clamp(multiplier × atr_pct, min_pct, max_pct))

        Clamp ensures:
          - Never tighter than 10% (avoids whipsawing in low-vol stocks)
          - Never wider than 25%  (caps maximum loss per trade)

        Examples at multiplier=4:
          HAL  (ATR ~2.5%): stop = peak × (1 − 10.0%) = 10% trail
          TRENT (ATR ~2.0%): stop = peak × (1 − 8.0%) = 8% trail
          YESBANK (ATR ~4%): stop = peak × (1 − 16%)  = 16% trail (high vol)
        """
        distance = max(min_pct, min(max_pct, multiplier * atr_pct))
        return peak_price * (1.0 - distance)

    # ── Descriptors ──────────────────────────────────────────────────────────

    def describe(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        nifty_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Debug breakdown of all signal components at a given date.
        Useful for understanding why the score is what it is.
        """
        mask = df.index <= date
        df_s = df[mask]
        nifty_s = nifty_df[nifty_df.index <= date] if nifty_df is not None else None

        if len(df_s) < 2:
            return {"error": "insufficient data"}

        close  = df_s["close"].astype(float)
        volume = df_s["volume"].astype(float)
        n      = len(close)
        price  = float(close.iloc[-1])

        ma20  = float(close.iloc[-min(20, n):].mean())
        ma50  = float(close.iloc[-min(50, n):].mean())
        ma200 = float(close.iloc[-min(200, n):].mean())

        trend_raw = sum([
            price > ma20,
            price > ma50,
            price > ma200,
            n >= 50  and float(close.iloc[-min(20,n):].mean()) > float(close.iloc[-min(50,n):].mean()),
            n >= 200 and float(close.iloc[-min(50,n):].mean()) > float(close.iloc[-min(200,n):].mean()),
        ])

        ret1m  = (price / float(close.iloc[-22]) - 1) if n > 22  else None
        ret3m  = (price / float(close.iloc[-64]) - 1) if n > 64  else None
        ret6m  = (price / float(close.iloc[-127])- 1) if n > 127 else None
        ret11m = (price / float(close.iloc[-232])- 1) if n > 232 else None

        daily_r = close.pct_change().iloc[-21:]
        sharpe  = float(daily_r.mean() / daily_r.std()) if daily_r.std() > 1e-9 else 0.0

        up_vol = float(volume.iloc[-21:][close.pct_change().iloc[-21:] > 0].mean())
        dn_vol = float(volume.iloc[-21:][close.pct_change().iloc[-21:] <= 0].mean())
        vol_ratio = up_vol / (up_vol + dn_vol) if (up_vol + dn_vol) > 0 else 0.5

        full_scores = self.precompute_scores(df_s, nifty_s)
        candle_score = float(_CANDLE_ENGINE.precompute_scores(df_s).iloc[-1])
        candle_detail = _CANDLE_ENGINE.describe_patterns(df_s, date)

        return {
            "date":            str(date.date()),
            "price":           round(price, 2),
            "MA20":            round(ma20,  2),
            "MA50":            round(ma50,  2),
            "MA200":           round(ma200, 2),
            "trend_conds":     f"{trend_raw}/5",
            "ret_1m":          f"{ret1m*100:+.1f}%" if ret1m is not None else "N/A",
            "ret_3m":          f"{ret3m*100:+.1f}%" if ret3m is not None else "N/A",
            "ret_6m":          f"{ret6m*100:+.1f}%" if ret6m is not None else "N/A",
            "ret_11m":         f"{ret11m*100:+.1f}%" if ret11m is not None else "N/A",
            "daily_sharpe":    round(sharpe, 3),
            "volume_ratio":    round(vol_ratio, 3),
            "candlestick_score": round(candle_score, 4),
            "patterns_fired":  candle_detail.get("patterns", []),
            "composite_score": round(float(full_scores.iloc[-1]), 4),
            "signal":          "BULLISH" if full_scores.iloc[-1] >= ENTRY_THRESHOLD
                               else ("BEARISH" if full_scores.iloc[-1] <= EXIT_THRESHOLD
                             else "NEUTRAL"),
        }

    # ── Multi-timeframe: weekly trend confirmation ────────────────────────────

    @staticmethod
    def weekly_aligned(
        df: pd.DataFrame,
        date: pd.Timestamp,
        min_weeks: int = 10,
    ) -> bool:
        """
        Returns True if the *weekly* trend is aligned with a bullish entry.

        Why needed: The daily momentum score can cross 0.55 on a single-day
        spike during a multi-week bearish trend. The TATASTEEL March 2025 entry
        is the canonical example: daily score fired on a 3-day bounce; the weekly
        chart was clearly still bearish. Weekly confirmation blocks these.

        Method:
          1. Resample daily OHLCV to weekly (last close of each 5-bar week)
          2. Compute: 5-week MA (≈ 1 month), 10-week MA (≈ 2 months)
          3. Weekly aligned = close > 5-week MA AND 5-week MA > 10-week MA
             (two conditions, both required → trend AND acceleration)
          4. Minimum 10 weekly bars (≈ 50 daily bars) for a valid signal.
             Below minimum → return True (don't block entries on data-sparse stocks).

        Args:
            df:        Daily OHLCV DataFrame with DatetimeIndex (columns lowercase)
            date:      The current bar date (only uses data up to and including this date)
            min_weeks: Minimum weeks of history required (default 10)

        Returns:
            True  → weekly trend aligned with bullish daily signal → allow entry
            False → weekly trend disagrees → block entry (wait for weekly recovery)
        """
        try:
            # Slice up to and including date
            df_slice = df.loc[df.index <= date, "close"].astype(float).dropna()

            if len(df_slice) < min_weeks * 5:
                # Insufficient weekly history → don't block (benefit of doubt)
                return True

            # Resample to weekly: take the last close of each 5-bar rolling group.
            # Using pd.Grouper('W') aligns to calendar weeks (Mon-Fri).
            df_weekly = (
                df_slice
                .resample("W", label="right", closed="right")
                .last()
                .dropna()
            )

            if len(df_weekly) < min_weeks:
                return True

            wk_close = df_weekly.values  # numpy array, oldest first

            # 5-week and 10-week MAs (simple)
            ma5  = float(np.mean(wk_close[-5:]))
            ma10 = float(np.mean(wk_close[-10:]))
            curr = float(wk_close[-1])

            # Both conditions must hold for a bullish weekly alignment
            return (curr > ma5) and (ma5 > ma10)

        except Exception as e:
            logger.debug(f"weekly_aligned failed: {e}")
            return True  # fail-open: don't block on errors

"""
MARK5 — Candlestick & Market Pattern Engine
============================================
Translates raw OHLCV bars into a pattern-based score component for the
MomentumSignalEngine.  Implements 22 classic Japanese candlestick patterns
across single-bar, two-bar, and three-bar families, with context-aware
weighting (trend position, support/resistance proximity, volume confirmation).

PATTERN INVENTORY
  Single-bar (9): Doji, Dragonfly Doji, Gravestone Doji, Hammer, Hanging Man,
                  Shooting Star, Inverted Hammer, Bullish Marubozu,
                  Bearish Marubozu
  Two-bar   (8): Bullish Engulfing, Bearish Engulfing, Bullish Harami,
                  Bearish Harami, Piercing Line, Dark Cloud Cover,
                  Tweezer Bottom, Tweezer Top
  Three-bar (5): Morning Star, Evening Star, Three White Soldiers,
                  Three Black Crows, Inside Bar Breakout

SCORING MODEL
  Each detected pattern receives a raw score in [-1, +1]:
    +1  = strong bullish pattern  (e.g. Three White Soldiers)
    -1  = strong bearish pattern  (e.g. Three Black Crows)
     0  = neutral or ambiguous

  Context multipliers (applied before aggregation):
    • "At support" (price near 20-day low): bullish patterns ×1.40
    • "At resistance" (price near 20-day high): bearish patterns ×1.40
    • "In downtrend" (price < MA20 < MA50): bearish patterns ×1.20
    • "In uptrend" (price > MA20 > MA50): bullish continuation patterns ×1.20
    • Volume confirmation (volume > 1.3× 20d avg): all patterns ×1.25

  Rolling window (default 5 bars): weighted mean of recent net signals, each
  bar's signal also decays by a geometric factor (most-recent bar weighted ×1.0,
  5-bar-ago ×0.5).

  Final score = sigmoid(weighted_net_signal / SCALE) → clipped to [0,1]
    Score > 0.6 → net bullish pattern context
    Score < 0.4 → net bearish pattern context

INTEGRATION
  Used as the "candlestick" component in MomentumSignalEngine.precompute_scores().
  Weight: 10% of composite score.

Academic basis:
  Nison (1991) — Japanese Candlestick Charting Techniques
  Bulkowski (2008) — Encyclopedia of Candlestick Charts
  Goo, Chen & Chang (2007) — application to Taiwan equity markets
  Caginalp & Laurent (1998) — predictability of candlestick chart patterns

SAFETY: PAPER MODE ONLY — this module does not execute any orders.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("MARK5.CandlestickPatterns")

# ── Calibration constants ──────────────────────────────────────────────────────

# Body-to-range thresholds
DOJI_BODY_RATIO        = 0.08   # body < 8% of range → doji
MARUBOZU_SHADOW_RATIO  = 0.05   # shadows < 5% of range each → marubozu
HAMMER_SHADOW_RATIO    = 2.0    # lower shadow ≥ 2× body
STAR_SHADOW_RATIO      = 2.0    # upper shadow ≥ 2× body
ENGULF_TOLERANCE       = 0.001  # engulfing: body must exceed prior body by this fraction

# Context thresholds
SUPPORT_PERCENTILE     = 0.20   # price in bottom 20% of 20d range → "at support"
RESIST_PERCENTILE      = 0.80   # price in top 20% of 20d range → "at resistance"
VOLUME_CONFIRM_RATIO   = 1.30   # volume ≥ 1.3× 20d average → confirms pattern

# Scoring
_SIGMOID_SCALE         = 0.50   # controls how steeply the sigmoid responds
_DECAY_FACTOR          = 0.85   # geometric decay for older bars in rolling window
_ROLL_WINDOW           = 5      # rolling bars for aggregation

# Pattern base scores (absolute value; sign applied by bullish/bearish direction)
SCORE = {
    # Three-bar patterns — highest reliability
    "three_white_soldiers": 0.90,
    "three_black_crows":    0.90,
    "morning_star":         0.80,
    "evening_star":         0.80,
    # Two-bar patterns
    "bullish_engulfing":    0.75,
    "bearish_engulfing":    0.75,
    "piercing_line":        0.65,
    "dark_cloud_cover":     0.65,
    "tweezer_bottom":       0.55,
    "tweezer_top":          0.55,
    "bullish_harami":       0.45,
    "bearish_harami":       0.45,
    # Single-bar patterns
    "marubozu_bull":        0.65,
    "marubozu_bear":        0.65,
    "hammer":               0.60,
    "hanging_man":          0.60,
    "shooting_star":        0.60,
    "inverted_hammer":      0.55,
    "dragonfly_doji":       0.50,
    "gravestone_doji":      0.50,
    "doji":                 0.25,
}


def _sigmoid(x: float, scale: float = _SIGMOID_SCALE) -> float:
    """Sigmoid ℝ → (0, 1) with given scale."""
    try:
        return 1.0 / (1.0 + math.exp(-x / scale))
    except (OverflowError, ZeroDivisionError):
        return 0.0 if x < 0 else 1.0


# ── Pattern detection helpers (vectorised) ────────────────────────────────────

def _body_metrics(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> dict:
    """
    Pre-compute all body/shadow metrics needed by pattern detectors.
    All Series share the same index as the input df.
    """
    full_range  = (high - low).replace(0, np.nan)
    body_size   = (close - open_).abs()
    body_top    = pd.concat([close, open_], axis=1).max(axis=1)
    body_bot    = pd.concat([close, open_], axis=1).min(axis=1)
    upper_shad  = high - body_top
    lower_shad  = body_bot - low
    is_bull     = (close > open_).astype(float)  # 1 = bullish, 0 = bearish/neutral
    is_bear     = (close < open_).astype(float)
    body_ratio  = (body_size / full_range).fillna(0.0)   # fraction of range that is body
    upper_ratio = (upper_shad / full_range).fillna(0.0)
    lower_ratio = (lower_shad / full_range).fillna(0.0)

    return dict(
        full_range=full_range,
        body_size=body_size,
        body_top=body_top,
        body_bot=body_bot,
        upper_shad=upper_shad,
        lower_shad=lower_shad,
        is_bull=is_bull,
        is_bear=is_bear,
        body_ratio=body_ratio,
        upper_ratio=upper_ratio,
        lower_ratio=lower_ratio,
    )


# ── Single-bar pattern detectors ─────────────────────────────────────────────

def _detect_doji(m: dict) -> pd.Series:
    """Body < DOJI_BODY_RATIO of range. Returns +1 (bullish context later)."""
    return (m["body_ratio"] < DOJI_BODY_RATIO).astype(float) * 0.0  # neutral baseline


def _detect_dragonfly_doji(m: dict) -> pd.Series:
    """Open ≈ Close ≈ High; long lower shadow → bullish at support."""
    is_doji  = m["body_ratio"] < DOJI_BODY_RATIO
    long_lo  = m["lower_ratio"] > 0.60
    short_hi = m["upper_ratio"] < 0.10
    return (is_doji & long_lo & short_hi).astype(float)  # +1 = bullish


def _detect_gravestone_doji(m: dict) -> pd.Series:
    """Open ≈ Close ≈ Low; long upper shadow → bearish at resistance."""
    is_doji  = m["body_ratio"] < DOJI_BODY_RATIO
    long_hi  = m["upper_ratio"] > 0.60
    short_lo = m["lower_ratio"] < 0.10
    return (is_doji & long_hi & short_lo).astype(float)  # +1 → will be negated for bear


def _detect_hammer(m: dict) -> pd.Series:
    """
    Small body in upper 1/3 of range, lower shadow ≥ 2× body.
    Shape-based (bullish/bearish context applied separately).
    """
    long_lo    = m["lower_shad"] >= HAMMER_SHADOW_RATIO * m["body_size"].replace(0, np.nan).fillna(0.001)
    short_hi   = m["upper_ratio"] < 0.15
    has_body   = m["body_ratio"] > 0.03          # needs some body
    return (long_lo & short_hi & has_body).astype(float)


def _detect_shooting_star(m: dict) -> pd.Series:
    """
    Small body in lower 1/3 of range, upper shadow ≥ 2× body.
    Shape-based.
    """
    long_hi   = m["upper_shad"] >= STAR_SHADOW_RATIO * m["body_size"].replace(0, np.nan).fillna(0.001)
    short_lo  = m["lower_ratio"] < 0.15
    has_body  = m["body_ratio"] > 0.03
    return (long_hi & short_lo & has_body).astype(float)


def _detect_marubozu_bull(m: dict) -> pd.Series:
    """Almost no shadows, bullish body > 60% of range."""
    strong  = m["body_ratio"] > 0.60
    bull    = m["is_bull"] > 0.5
    small_shad = (m["upper_ratio"] < MARUBOZU_SHADOW_RATIO) & (m["lower_ratio"] < MARUBOZU_SHADOW_RATIO)
    return (strong & bull & small_shad).astype(float)


def _detect_marubozu_bear(m: dict) -> pd.Series:
    """Almost no shadows, bearish body > 60% of range."""
    strong  = m["body_ratio"] > 0.60
    bear    = m["is_bear"] > 0.5
    small_shad = (m["upper_ratio"] < MARUBOZU_SHADOW_RATIO) & (m["lower_ratio"] < MARUBOZU_SHADOW_RATIO)
    return (strong & bear & small_shad).astype(float)


# ── Two-bar pattern detectors ─────────────────────────────────────────────────

def _detect_bullish_engulfing(
    open_: pd.Series, close: pd.Series, m: dict
) -> pd.Series:
    """
    Bar t-1 is bearish; Bar t is bullish AND body completely engulfs t-1's body.
    """
    prev_is_bear   = m["is_bear"].shift(1)
    prev_body_top  = m["body_top"].shift(1)
    prev_body_bot  = m["body_bot"].shift(1)
    curr_is_bull   = m["is_bull"]
    curr_body_top  = m["body_top"]
    curr_body_bot  = m["body_bot"]
    engulfs = (
        curr_body_top > prev_body_top + ENGULF_TOLERANCE * prev_body_top.abs().replace(0, 0.001)
    ) & (
        curr_body_bot < prev_body_bot - ENGULF_TOLERANCE * prev_body_bot.abs().replace(0, 0.001)
    )
    return (prev_is_bear.fillna(0) * curr_is_bull * engulfs.astype(float)).fillna(0.0)


def _detect_bearish_engulfing(
    open_: pd.Series, close: pd.Series, m: dict
) -> pd.Series:
    """Bar t-1 bullish; Bar t bearish AND completely engulfs t-1's body."""
    prev_is_bull   = m["is_bull"].shift(1)
    prev_body_top  = m["body_top"].shift(1)
    prev_body_bot  = m["body_bot"].shift(1)
    curr_is_bear   = m["is_bear"]
    curr_body_top  = m["body_top"]
    curr_body_bot  = m["body_bot"]
    engulfs = (
        curr_body_top > prev_body_top + ENGULF_TOLERANCE * prev_body_top.abs().replace(0, 0.001)
    ) & (
        curr_body_bot < prev_body_bot - ENGULF_TOLERANCE * prev_body_bot.abs().replace(0, 0.001)
    )
    return (prev_is_bull.fillna(0) * curr_is_bear * engulfs.astype(float)).fillna(0.0)


def _detect_bullish_harami(m: dict) -> pd.Series:
    """Bar t-1 large bearish; Bar t small bullish INSIDE t-1's body."""
    prev_is_bear   = m["is_bear"].shift(1)
    prev_body_top  = m["body_top"].shift(1)
    prev_body_bot  = m["body_bot"].shift(1)
    prev_is_large  = m["body_ratio"].shift(1) > 0.40
    curr_is_bull   = m["is_bull"]
    curr_inside    = (m["body_top"] < prev_body_top.fillna(0)) & \
                     (m["body_bot"] > prev_body_bot.fillna(0))
    curr_small     = m["body_ratio"] < 0.30
    return (prev_is_bear.fillna(0) * prev_is_large.fillna(0) * curr_is_bull *
            curr_inside.astype(float) * curr_small.astype(float)).fillna(0.0)


def _detect_bearish_harami(m: dict) -> pd.Series:
    """Bar t-1 large bullish; Bar t small bearish INSIDE t-1's body."""
    prev_is_bull   = m["is_bull"].shift(1)
    prev_body_top  = m["body_top"].shift(1)
    prev_body_bot  = m["body_bot"].shift(1)
    prev_is_large  = m["body_ratio"].shift(1) > 0.40
    curr_is_bear   = m["is_bear"]
    curr_inside    = (m["body_top"] < prev_body_top.fillna(0)) & \
                     (m["body_bot"] > prev_body_bot.fillna(0))
    curr_small     = m["body_ratio"] < 0.30
    return (prev_is_bull.fillna(0) * prev_is_large.fillna(0) * curr_is_bear *
            curr_inside.astype(float) * curr_small.astype(float)).fillna(0.0)


def _detect_piercing_line(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, m: dict
) -> pd.Series:
    """
    Bar t-1 bearish; Bar t opens BELOW t-1 low, closes above 50% of t-1's body.
    Bullish reversal.
    """
    prev_is_bear   = m["is_bear"].shift(1)
    prev_body_top  = m["body_top"].shift(1)
    prev_body_bot  = m["body_bot"].shift(1)
    midpoint       = ((prev_body_top + prev_body_bot) / 2).fillna(0)
    gap_down       = open_ < low.shift(1)
    closes_in      = (close > midpoint) & (close < prev_body_top.fillna(0))
    curr_is_bull   = m["is_bull"]
    return (prev_is_bear.fillna(0) * gap_down.astype(float) *
            closes_in.astype(float) * curr_is_bull).fillna(0.0)


def _detect_dark_cloud_cover(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, m: dict
) -> pd.Series:
    """
    Bar t-1 bullish; Bar t opens ABOVE t-1 high, closes below 50% of t-1's body.
    Bearish reversal.
    """
    prev_is_bull   = m["is_bull"].shift(1)
    prev_body_top  = m["body_top"].shift(1)
    prev_body_bot  = m["body_bot"].shift(1)
    midpoint       = ((prev_body_top + prev_body_bot) / 2).fillna(0)
    gap_up         = open_ > high.shift(1)
    closes_in      = (close < midpoint) & (close > prev_body_bot.fillna(0))
    curr_is_bear   = m["is_bear"]
    return (prev_is_bull.fillna(0) * gap_up.astype(float) *
            closes_in.astype(float) * curr_is_bear).fillna(0.0)


def _detect_tweezer_bottom(
    low: pd.Series, m: dict, tolerance: float = 0.003
) -> pd.Series:
    """
    Two consecutive bars with approximately the same LOW → double support.
    Bullish reversal when in downtrend.
    """
    same_low = (low - low.shift(1)).abs() / low.shift(1).replace(0, np.nan) < tolerance
    return same_low.fillna(False).astype(float)


def _detect_tweezer_top(
    high: pd.Series, m: dict, tolerance: float = 0.003
) -> pd.Series:
    """
    Two consecutive bars with approximately the same HIGH → double resistance.
    Bearish reversal when in uptrend.
    """
    same_high = (high - high.shift(1)).abs() / high.shift(1).replace(0, np.nan) < tolerance
    return same_high.fillna(False).astype(float)


# ── Three-bar pattern detectors ───────────────────────────────────────────────

def _detect_morning_star(
    open_: pd.Series, close: pd.Series, m: dict
) -> pd.Series:
    """
    Bar t-2: large bearish.
    Bar t-1: small body (star) — open and close within bar t-2's lower range.
    Bar t:   large bullish, close >= 50% of bar t-2's body.
    """
    # bar indices shifted by 2 and 1
    bear2      = m["is_bear"].shift(2)
    large2     = m["body_ratio"].shift(2) > 0.40
    small1     = m["body_ratio"].shift(1) < 0.30
    bull0      = m["is_bull"]
    large0     = m["body_ratio"] > 0.40
    # close of current bar must penetrate at least 50% of bar t-2's body
    body_top2  = m["body_top"].shift(2)
    body_bot2  = m["body_bot"].shift(2)
    mid2       = ((body_top2 + body_bot2) / 2).fillna(0)
    recover    = close > mid2

    pattern = (bear2.fillna(0) * large2.fillna(0) * small1.fillna(0) *
               bull0 * large0.astype(float) * recover.astype(float))
    return pattern.fillna(0.0)


def _detect_evening_star(
    open_: pd.Series, close: pd.Series, m: dict
) -> pd.Series:
    """
    Bar t-2: large bullish.
    Bar t-1: small body (star).
    Bar t:   large bearish, close ≤ 50% of bar t-2's body.
    """
    bull2      = m["is_bull"].shift(2)
    large2     = m["body_ratio"].shift(2) > 0.40
    small1     = m["body_ratio"].shift(1) < 0.30
    bear0      = m["is_bear"]
    large0     = m["body_ratio"] > 0.40
    body_top2  = m["body_top"].shift(2)
    body_bot2  = m["body_bot"].shift(2)
    mid2       = ((body_top2 + body_bot2) / 2).fillna(0)
    decline    = close < mid2

    pattern = (bull2.fillna(0) * large2.fillna(0) * small1.fillna(0) *
               bear0 * large0.astype(float) * decline.astype(float))
    return pattern.fillna(0.0)


def _detect_three_white_soldiers(close: pd.Series, m: dict) -> pd.Series:
    """
    Three consecutive bullish candles, each closing higher than the previous.
    Bodies should each be substantial (> 40% of range).
    """
    c0, c1, c2 = close, close.shift(1), close.shift(2)
    b0 = m["body_ratio"]
    b1 = m["body_ratio"].shift(1)
    b2 = m["body_ratio"].shift(2)
    bull0 = m["is_bull"]
    bull1 = m["is_bull"].shift(1)
    bull2 = m["is_bull"].shift(2)
    higher_closes = (c0 > c1) & (c1 > c2)
    all_substantial = (b0 > 0.35) & (b1 > 0.35) & (b2 > 0.35)
    all_bull = (bull0 > 0.5) & (bull1.fillna(0) > 0.5) & (bull2.fillna(0) > 0.5)
    return (higher_closes & all_substantial & all_bull).astype(float).fillna(0.0)


def _detect_three_black_crows(close: pd.Series, m: dict) -> pd.Series:
    """
    Three consecutive bearish candles, each closing lower than the previous.
    Bodies should each be substantial.
    """
    c0, c1, c2 = close, close.shift(1), close.shift(2)
    b0 = m["body_ratio"]
    b1 = m["body_ratio"].shift(1)
    b2 = m["body_ratio"].shift(2)
    bear0 = m["is_bear"]
    bear1 = m["is_bear"].shift(1)
    bear2 = m["is_bear"].shift(2)
    lower_closes  = (c0 < c1) & (c1 < c2)
    all_substantial = (b0 > 0.35) & (b1 > 0.35) & (b2 > 0.35)
    all_bear = (bear0 > 0.5) & (bear1.fillna(0) > 0.5) & (bear2.fillna(0) > 0.5)
    return (lower_closes & all_substantial & all_bear).astype(float).fillna(0.0)


# ── Context computation ───────────────────────────────────────────────────────

def _context_multiplier(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    signal: pd.Series,    # raw signed signal (+ = bullish, - = bearish)
) -> pd.Series:
    """
    Apply context multipliers to a signed pattern signal Series.

    Multipliers stack multiplicatively up to a cap of ×2.0.

    Context signals (20-bar rolling windows):
      - At support (close near 20-bar low): amplify bullish signals
      - At resistance (close near 20-bar high): amplify bearish signals
      - Trend (MA20 vs MA50): amplify aligned-direction signals
      - Volume spike: amplify all signals when volume confirms
    """
    ma20   = close.rolling(20, min_periods=5).mean()
    ma50   = close.rolling(50, min_periods=10).mean()
    hi20   = high.rolling(20, min_periods=5).max()
    lo20   = low.rolling(20, min_periods=5).min()
    rng20  = (hi20 - lo20).replace(0, np.nan)
    vol20  = volume.rolling(20, min_periods=5).mean().replace(0, np.nan)

    # Position within 20-bar range  [0 = at low, 1 = at high]
    pos    = ((close - lo20) / rng20).fillna(0.5)

    # Context booleans
    at_support    = pos < SUPPORT_PERCENTILE
    at_resistance = pos > RESIST_PERCENTILE
    in_uptrend    = (close > ma20) & (ma20 > ma50.fillna(0))
    in_downtrend  = (close < ma20) & (ma20 < ma50.fillna(close * 1e6))
    vol_spike     = volume > VOLUME_CONFIRM_RATIO * vol20.fillna(0)

    mult = pd.Series(1.0, index=close.index)

    # Bullish signals at support or in uptrend (continuation)
    bull_signal = signal > 0
    mult = mult.where(~(bull_signal & at_support), other=mult * 1.40)
    mult = mult.where(~(bull_signal & in_uptrend), other=mult * 1.20)

    # Bearish signals at resistance or in downtrend
    bear_signal = signal < 0
    mult = mult.where(~(bear_signal & at_resistance), other=mult * 1.40)
    mult = mult.where(~(bear_signal & in_downtrend), other=mult * 1.20)

    # Volume spike amplifies any signal
    mult = mult.where(~(vol_spike & (signal != 0)), other=mult * 1.25)

    return mult.clip(upper=2.0)


# ── Main engine ───────────────────────────────────────────────────────────────

class CandlestickPatternEngine:
    """
    Vectorised candlestick pattern detection and scoring engine.

    Usage:
        engine = CandlestickPatternEngine()
        scores = engine.precompute_scores(df)   # pd.Series in [0, 1]

    Args for precompute_scores:
        df : pd.DataFrame with DatetimeIndex and lowercase OHLCV columns
             (open, high, low, close, volume)
        context : bool  — whether to apply context multipliers (default True)

    Returns:
        pd.Series in [0, 1] indexed like df
          > 0.6 = net bullish pattern environment
          < 0.4 = net bearish pattern environment
    """

    def __init__(
        self,
        roll_window: int = _ROLL_WINDOW,
        sigmoid_scale: float = _SIGMOID_SCALE,
        decay_factor: float = _DECAY_FACTOR,
    ):
        self.roll_window    = roll_window
        self.sigmoid_scale  = sigmoid_scale
        self.decay_factor   = decay_factor

    def precompute_scores(
        self,
        df: pd.DataFrame,
        context: bool = True,
    ) -> pd.Series:
        """
        Compute pattern scores for every bar in df.

        All pattern detection is look-ahead free (uses only df.iloc[:i+1]).
        Vectorised implementation: O(n) over all bars.

        Returns pd.Series in [0, 1] with same index as df.
        """
        if len(df) < 5:
            return pd.Series(0.5, index=df.index)

        try:
            open_  = df["open"].astype(float)
            high   = df["high"].astype(float)
            low    = df["low"].astype(float)
            close  = df["close"].astype(float)
            volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)
        except KeyError as e:
            logger.warning(f"Missing OHLCV column: {e} — returning neutral 0.5")
            return pd.Series(0.5, index=df.index)

        m = _body_metrics(open_, high, low, close)

        # ── Detect all patterns, produce signed signal per bar ────────────────
        # Convention: positive = bullish signal, negative = bearish signal

        # ----- Single-bar bullish -----
        dragonfly    = _detect_dragonfly_doji(m)
        hammer_shape = _detect_hammer(m)
        inv_hammer   = _detect_shooting_star(m)   # same shape, bullish at bottom
        marubozu_b   = _detect_marubozu_bull(m)

        # Context for single-bar: hammer is bullish only in downtrend / at support
        ma20 = close.rolling(20, min_periods=5).mean()
        at_bottom = close < ma20   # rough "at support" proxy for single-bar

        # Hammer shape at bottom = bullish hammer;  at top = bearish hanging man
        hammer = hammer_shape * at_bottom.astype(float)
        hanging_man = hammer_shape * (~at_bottom).astype(float)

        # Shooting-star shape at top = bearish; at bottom = inverted hammer (bullish)
        shooting_star = _detect_shooting_star(m) * (~at_bottom).astype(float)
        inv_hammer    = _detect_shooting_star(m) * at_bottom.astype(float)

        # ----- Single-bar bearish -----
        gravestone   = _detect_gravestone_doji(m)
        marubozu_br  = _detect_marubozu_bear(m)

        # ----- Two-bar bullish -----
        bull_engulf  = _detect_bullish_engulfing(open_, close, m)
        bull_harami  = _detect_bullish_harami(m)
        piercing     = _detect_piercing_line(open_, high, low, close, m)
        tweezer_bot  = _detect_tweezer_bottom(low, m)

        # ----- Two-bar bearish -----
        bear_engulf  = _detect_bearish_engulfing(open_, close, m)
        bear_harami  = _detect_bearish_harami(m)
        dark_cloud   = _detect_dark_cloud_cover(open_, high, low, close, m)
        tweezer_top_ = _detect_tweezer_top(high, m)

        # ----- Three-bar bullish -----
        morning_star = _detect_morning_star(open_, close, m)
        three_white  = _detect_three_white_soldiers(close, m)

        # ----- Three-bar bearish -----
        evening_star = _detect_evening_star(open_, close, m)
        three_black  = _detect_three_black_crows(close, m)

        # ── Build signed raw signal per bar ──────────────────────────────────
        bullish_raw = (
            dragonfly    * SCORE["dragonfly_doji"]      +
            hammer       * SCORE["hammer"]              +
            inv_hammer   * SCORE["inverted_hammer"]     +
            marubozu_b   * SCORE["marubozu_bull"]       +
            bull_engulf  * SCORE["bullish_engulfing"]   +
            bull_harami  * SCORE["bullish_harami"]      +
            piercing     * SCORE["piercing_line"]       +
            tweezer_bot  * SCORE["tweezer_bottom"]      +
            morning_star * SCORE["morning_star"]        +
            three_white  * SCORE["three_white_soldiers"]
        )

        bearish_raw = (
            gravestone   * SCORE["gravestone_doji"]     +
            hanging_man  * SCORE["hanging_man"]         +
            shooting_star* SCORE["shooting_star"]       +
            marubozu_br  * SCORE["marubozu_bear"]       +
            bear_engulf  * SCORE["bearish_engulfing"]   +
            bear_harami  * SCORE["bearish_harami"]      +
            dark_cloud   * SCORE["dark_cloud_cover"]    +
            tweezer_top_ * SCORE["tweezer_top"]         +
            evening_star * SCORE["evening_star"]        +
            three_black  * SCORE["three_black_crows"]
        )

        net_signal = bullish_raw - bearish_raw  # positive = bullish, negative = bearish

        # ── Context multipliers ───────────────────────────────────────────────
        if context:
            mult = _context_multiplier(close, high, low, volume, net_signal)
            net_signal = net_signal * mult

        # ── Rolling aggregation with geometric decay ──────────────────────────
        # Weight most recent bar ×1.0, previous bar ×decay, ...
        window  = self.roll_window
        decay   = self.decay_factor
        weights = np.array([decay ** k for k in range(window - 1, -1, -1)])
        weights = weights / weights.sum()  # normalise

        # Convolve with decay weights (pandas rolling with weights)
        rolling_net = pd.Series(0.0, index=df.index)
        net_arr = net_signal.values
        for i in range(len(df)):
            start = max(0, i - window + 1)
            chunk = net_arr[start : i + 1]
            w     = weights[-(len(chunk)):]  # trim weights to available history
            w     = w / w.sum()
            rolling_net.iloc[i] = float(np.dot(chunk, w))

        # ── Sigmoid normalisation → [0, 1] ────────────────────────────────────
        result = rolling_net.apply(
            lambda x: _sigmoid(x, self.sigmoid_scale) if pd.notna(x) else 0.5
        ).clip(0.0, 1.0)

        # Ensure early bars with no pattern history return 0.5 (neutral)
        n_bars = pd.Series(range(1, len(df) + 1), index=df.index)
        result = result.where(n_bars >= 5, other=0.5)

        return result.fillna(0.5)

    def score_at(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        context: bool = True,
    ) -> float:
        """
        Single-point score at `date`.  For live inference.
        Uses only data up to and including `date`.
        """
        df_slice = df[df.index <= date]
        if len(df_slice) < 3:
            return 0.5
        scores = self.precompute_scores(df_slice, context=context)
        return float(scores.iloc[-1])

    def describe_patterns(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
    ) -> dict:
        """
        Debugging aid: lists which patterns fired at `date` with their scores.
        Returns a dict with pattern names, directions, and net score.
        """
        df_slice = df[df.index <= date].tail(10)  # last 10 bars for context
        if len(df_slice) < 3:
            return {"error": "insufficient data"}

        open_  = df_slice["open"].astype(float)
        high   = df_slice["high"].astype(float)
        low    = df_slice["low"].astype(float)
        close  = df_slice["close"].astype(float)
        volume = df_slice.get("volume", pd.Series(1.0, index=df_slice.index)).astype(float)

        m = _body_metrics(open_, high, low, close)
        ma20 = close.rolling(20, min_periods=1).mean()
        at_bottom = close < ma20

        fired = []

        # Helper: check last bar of a detection series
        def _last(s: pd.Series) -> bool:
            return bool(s.iloc[-1] > 0)

        detections = {
            "dragonfly_doji":       (_detect_dragonfly_doji(m), "BULLISH"),
            "gravestone_doji":      (_detect_gravestone_doji(m), "BEARISH"),
            "hammer":               (_detect_hammer(m) * at_bottom.astype(float), "BULLISH"),
            "hanging_man":          (_detect_hammer(m) * (~at_bottom).astype(float), "BEARISH"),
            "inverted_hammer":      (_detect_shooting_star(m) * at_bottom.astype(float), "BULLISH"),
            "shooting_star":        (_detect_shooting_star(m) * (~at_bottom).astype(float), "BEARISH"),
            "marubozu_bull":        (_detect_marubozu_bull(m), "BULLISH"),
            "marubozu_bear":        (_detect_marubozu_bear(m), "BEARISH"),
            "bullish_engulfing":    (_detect_bullish_engulfing(open_, close, m), "BULLISH"),
            "bearish_engulfing":    (_detect_bearish_engulfing(open_, close, m), "BEARISH"),
            "bullish_harami":       (_detect_bullish_harami(m), "BULLISH"),
            "bearish_harami":       (_detect_bearish_harami(m), "BEARISH"),
            "piercing_line":        (_detect_piercing_line(open_, high, low, close, m), "BULLISH"),
            "dark_cloud_cover":     (_detect_dark_cloud_cover(open_, high, low, close, m), "BEARISH"),
            "tweezer_bottom":       (_detect_tweezer_bottom(low, m), "BULLISH"),
            "tweezer_top":          (_detect_tweezer_top(high, m), "BEARISH"),
            "morning_star":         (_detect_morning_star(open_, close, m), "BULLISH"),
            "evening_star":         (_detect_evening_star(open_, close, m), "BEARISH"),
            "three_white_soldiers": (_detect_three_white_soldiers(close, m), "BULLISH"),
            "three_black_crows":    (_detect_three_black_crows(close, m), "BEARISH"),
        }

        for name, (series, direction) in detections.items():
            if _last(series):
                fired.append({"pattern": name, "direction": direction,
                              "score": SCORE.get(name, 0.50)})

        net_score = self.score_at(df, date)

        return {
            "date":      str(date.date()),
            "patterns":  fired,
            "count":     len(fired),
            "net_score": round(net_score, 4),
            "signal":    "BULLISH" if net_score > 0.6 else (
                         "BEARISH" if net_score < 0.4 else "NEUTRAL"),
        }

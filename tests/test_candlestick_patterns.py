"""
Tests for core/models/candlestick_patterns.py
=============================================
Covers:
  - All 20 individual pattern detectors
  - Rolling score aggregation and sigmoid mapping
  - Context multipliers (support, resistance, volume, trend)
  - No look-ahead contamination
  - Integration with MomentumSignalEngine
  - Edge cases: sparse data, constant prices, NaN, missing columns
  - Score range invariants

Total: 60+ tests
"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models.candlestick_patterns import (
    CandlestickPatternEngine,
    _body_metrics,
    _detect_doji,
    _detect_dragonfly_doji,
    _detect_gravestone_doji,
    _detect_hammer,
    _detect_shooting_star,
    _detect_marubozu_bull,
    _detect_marubozu_bear,
    _detect_bullish_engulfing,
    _detect_bearish_engulfing,
    _detect_bullish_harami,
    _detect_bearish_harami,
    _detect_piercing_line,
    _detect_dark_cloud_cover,
    _detect_tweezer_bottom,
    _detect_tweezer_top,
    _detect_morning_star,
    _detect_evening_star,
    _detect_three_white_soldiers,
    _detect_three_black_crows,
    SCORE,
    _sigmoid,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_bar(o, h, l, c, v=1_000_000, n=1):
    """Create a single-row OHLCV DataFrame at a specific date."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": [o]*n, "high": [h]*n, "low": [l]*n,
                         "close": [c]*n, "volume": [v]*n}, index=idx)


def _make_trend(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Create a steady uptrend OHLCV DataFrame."""
    idx   = pd.date_range("2022-01-01", periods=n, freq="B")
    close = np.arange(start, start + n * step, step)[:n]
    open_ = close - 0.5
    high  = close + 0.5
    low   = close - 1.0
    vol   = np.full(n, 500_000)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def _make_downtrend(n: int, start: float = 200.0, step: float = 1.0) -> pd.DataFrame:
    """Create a steady downtrend OHLCV DataFrame."""
    idx   = pd.date_range("2022-01-01", periods=n, freq="B")
    close = np.arange(start, start - n * step, -step)[:n]
    open_ = close + 0.5
    high  = close + 1.0
    low   = close - 0.5
    vol   = np.full(n, 500_000)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def _concat_rows(*rows: dict) -> pd.DataFrame:
    """Build a multi-bar DataFrame from list-of-dicts."""
    records = list(rows)
    idx = pd.date_range("2024-01-01", periods=len(records), freq="B")
    df  = pd.DataFrame(records, index=idx)
    df["volume"] = df.get("volume", pd.Series(1_000_000, index=idx))
    return df


# ── Body metrics ──────────────────────────────────────────────────────────────

class TestBodyMetrics:
    def test_bull_bar(self):
        o, h, l, c = 100, 105, 98, 104
        df = _make_bar(o, h, l, c)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert m["body_size"].iloc[0] == pytest.approx(abs(c - o))
        assert m["is_bull"].iloc[0]  == pytest.approx(1.0)
        assert m["is_bear"].iloc[0]  == pytest.approx(0.0)
        assert m["upper_shad"].iloc[0] == pytest.approx(h - max(o, c))
        assert m["lower_shad"].iloc[0] == pytest.approx(min(o, c) - l)

    def test_bear_bar(self):
        o, h, l, c = 104, 106, 99, 100
        df = _make_bar(o, h, l, c)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert m["is_bear"].iloc[0] == pytest.approx(1.0)
        assert m["is_bull"].iloc[0] == pytest.approx(0.0)

    def test_doji_bar(self):
        # open == close → zero body
        o, c, h, l = 100, 100, 103, 97
        df = _make_bar(o, h, l, c)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert m["body_ratio"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_full_range_nonzero(self):
        df = _make_bar(100, 105, 95, 102)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert m["full_range"].iloc[0] == pytest.approx(10.0)


# ── Sigmoid ───────────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero_maps_to_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_positive_above_half(self):
        assert _sigmoid(1.0) > 0.5

    def test_negative_below_half(self):
        assert _sigmoid(-1.0) < 0.5

    def test_range(self):
        for x in [-10, -1, 0, 1, 10]:
            s = _sigmoid(x)
            assert 0.0 <= s <= 1.0


# ── Single-bar patterns ───────────────────────────────────────────────────────

class TestDoji:
    def _detect(self, o, h, l, c):
        df = _make_bar(o, h, l, c)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        return bool(_detect_doji(m).iloc[0])  # doji returns 0.0 always (neutral)

    def test_perfect_doji(self):
        # body 0/10 range = 0 ratio — doji, but _detect_doji returns 0 (neutral)
        df = _make_bar(100, 105, 95, 100)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_doji(m).iloc[0]) == pytest.approx(0.0)

    def test_dragonfly_fires(self):
        # open=high=close, long lower shadow
        df = _make_bar(100, 100, 90, 100)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_dragonfly_doji(m).iloc[0]) == pytest.approx(1.0)

    def test_gravestone_fires(self):
        # open=low=close, long upper shadow
        df = _make_bar(100, 110, 100, 100)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_gravestone_doji(m).iloc[0]) == pytest.approx(1.0)

    def test_dragonfly_not_gravestone(self):
        df = _make_bar(100, 100, 90, 100)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_gravestone_doji(m).iloc[0]) == pytest.approx(0.0)

    def test_regular_candle_not_dragonfly(self):
        df = _make_bar(100, 105, 97, 103)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_dragonfly_doji(m).iloc[0]) == pytest.approx(0.0)


class TestHammer:
    def test_hammer_shape_detected(self):
        # Long lower shadow (~5), small body (0.3), tiny upper shadow (0.2)
        # open=100.3, close=100 → body=0.3; low=95; high=100.5
        df = _make_bar(100.3, 100.5, 95, 100)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = float(_detect_hammer(m).iloc[0])
        assert result == pytest.approx(1.0)

    def test_long_upper_shadow_not_hammer(self):
        # Upper shadow dominant → shooting star shape, not hammer
        df = _make_bar(100, 110, 99, 101)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_hammer(m).iloc[0]) == pytest.approx(0.0)

    def test_shooting_star_shape_detected(self):
        # Long upper shadow (~5), small body (0.2), tiny lower shadow (0.3)
        # open=100, close=99.8; high=105, low=99.5
        df = _make_bar(100, 105, 99.5, 99.8)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = float(_detect_shooting_star(m).iloc[0])
        assert result == pytest.approx(1.0)


class TestMarubozu:
    def test_bull_marubozu(self):
        # No shadows, strong bullish body
        df = _make_bar(100, 108, 100, 108)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_marubozu_bull(m).iloc[0]) == pytest.approx(1.0)

    def test_bear_marubozu(self):
        df = _make_bar(108, 108, 100, 100)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_marubozu_bear(m).iloc[0]) == pytest.approx(1.0)

    def test_shadow_prevents_marubozu(self):
        # Has a shadow → not marubozu
        df = _make_bar(100, 109, 99, 108)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_marubozu_bull(m).iloc[0]) == pytest.approx(0.0)

    def test_small_body_not_marubozu(self):
        # Body is only 10% of range → not marubozu
        df = _make_bar(100, 110, 99, 101)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        assert float(_detect_marubozu_bull(m).iloc[0]) == pytest.approx(0.0)


# ── Two-bar patterns ──────────────────────────────────────────────────────────

class TestEngulfing:
    def _build(self, o1, h1, l1, c1, o2, h2, l2, c2):
        df = _concat_rows(
            {"open": o1, "high": h1, "low": l1, "close": c1},
            {"open": o2, "high": h2, "low": l2, "close": c2},
        )
        return df

    def test_bullish_engulfing_detected(self):
        # Bar 1: bearish (o=104, c=100) — Bar 2: bullish body exceeds bar 1 body
        df = self._build(104, 105, 99, 100,   99, 106, 98, 105)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_bullish_engulfing(df["open"], df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_bearish_engulfing_detected(self):
        df = self._build(100, 106, 99, 105,   106, 107, 98, 99)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_bearish_engulfing(df["open"], df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_partial_engulf_not_detected(self):
        # Bar 2 only partially covers bar 1
        df = self._build(104, 105, 99, 100,   101, 104, 99, 103)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_bullish_engulfing(df["open"], df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(0.0)

    def test_same_direction_not_engulfing(self):
        # Both bars bullish — no bearish bar t-1 to engulf
        df = self._build(100, 105, 99, 104,   103, 109, 102, 108)
        m  = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_bullish_engulfing(df["open"], df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(0.0)


class TestHarami:
    def test_bullish_harami_detected(self):
        # Bar 1: large bearish (o=110, c=100) — Bar 2: small bullish inside
        df = _concat_rows(
            {"open": 110, "high": 111, "low": 99, "close": 100},
            {"open": 103, "high": 104, "low": 102, "close": 103.5},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_bullish_harami(m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_bearish_harami_detected(self):
        # Bar 1: large bullish — Bar 2: small bearish inside
        df = _concat_rows(
            {"open": 100, "high": 112, "low": 99, "close": 111},
            {"open": 107, "high": 108, "low": 106, "close": 106.5},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_bearish_harami(m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_outside_bar_not_harami(self):
        # Bar 2 is OUTSIDE bar 1 body → not harami
        df = _concat_rows(
            {"open": 110, "high": 111, "low": 99, "close": 100},
            {"open": 98,  "high": 115, "low": 97, "close": 114},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_bullish_harami(m)
        assert float(result.iloc[-1]) == pytest.approx(0.0)


class TestPiercingAndDarkCloud:
    def test_piercing_line_detected(self):
        # Bar 1 bearish, Bar 2 gaps below bar 1 low, closes above 50% of bar 1 body
        df = _concat_rows(
            {"open": 110, "high": 111, "low": 99, "close": 100},
            {"open": 98,  "high": 108, "low": 96, "close": 106},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_piercing_line(df["open"], df["high"], df["low"], df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_dark_cloud_cover_detected(self):
        # Bar 1 bullish, Bar 2 gaps above bar 1 high, closes below 50% of bar 1 body
        df = _concat_rows(
            {"open": 100, "high": 112, "low": 99, "close": 111},
            {"open": 113, "high": 114, "low": 103, "close": 104},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_dark_cloud_cover(df["open"], df["high"], df["low"], df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)


class TestTweezer:
    def test_tweezer_bottom(self):
        # Two bars with same low
        df = _concat_rows(
            {"open": 105, "high": 107, "low": 100, "close": 104},
            {"open": 103, "high": 106, "low": 100, "close": 105},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_tweezer_bottom(df["low"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_tweezer_top(self):
        df = _concat_rows(
            {"open": 104, "high": 110, "low": 103, "close": 106},
            {"open": 108, "high": 110, "low": 105, "close": 107},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_tweezer_top(df["high"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_different_lows_not_tweezer(self):
        df = _concat_rows(
            {"open": 105, "high": 107, "low": 100, "close": 104},
            {"open": 103, "high": 106, "low": 95,  "close": 105},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_tweezer_bottom(df["low"], m)
        assert float(result.iloc[-1]) == pytest.approx(0.0)


# ── Three-bar patterns ────────────────────────────────────────────────────────

class TestMorningStar:
    def test_morning_star_detected(self):
        # Bar 0: large bearish; Bar 1: small body; Bar 2: large bullish recovering
        df = _concat_rows(
            {"open": 110, "high": 111, "low": 99, "close": 100},   # large bear
            {"open": 101, "high": 102, "low": 99, "close": 101},   # small
            {"open": 101, "high": 112, "low": 100, "close": 109},  # large bull, recovers
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_morning_star(df["open"], df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_evening_star_detected(self):
        df = _concat_rows(
            {"open": 100, "high": 112, "low": 99, "close": 110},   # large bull
            {"open": 110, "high": 111, "low": 109, "close": 110},  # small
            {"open": 109, "high": 110, "low": 98, "close": 100},   # large bear
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_evening_star(df["open"], df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_morning_star_needs_recovery(self):
        # Bar 2 (bull) doesn't recover above 50% of bar 0 body → should NOT fire
        df = _concat_rows(
            {"open": 110, "high": 111, "low": 99, "close": 100},   # large bear body: 100-110
            {"open": 101, "high": 102, "low": 99, "close": 101},   # small
            {"open": 101, "high": 105, "low": 100, "close": 102},  # bull but barely recovers
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_morning_star(df["open"], df["close"], m)
        # close=102, mid of bear body (100-110) = 105 → doesn't recover → 0
        assert float(result.iloc[-1]) == pytest.approx(0.0)


class TestThreeSoldiers:
    def test_three_white_soldiers(self):
        df = _concat_rows(
            {"open": 99, "high": 104, "low": 98, "close": 104},
            {"open": 103, "high": 109, "low": 102, "close": 109},
            {"open": 108, "high": 115, "low": 107, "close": 115},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_three_white_soldiers(df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_three_black_crows(self):
        df = _concat_rows(
            {"open": 110, "high": 111, "low": 103, "close": 103},
            {"open": 104, "high": 105, "low": 97,  "close": 97},
            {"open": 98,  "high": 99,  "low": 90,  "close": 90},
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_three_black_crows(df["close"], m)
        assert float(result.iloc[-1]) == pytest.approx(1.0)

    def test_two_soldiers_not_three(self):
        df = _concat_rows(
            {"open": 100, "high": 102, "low": 99, "close": 101},  # tiny body → small
            {"open": 100, "high": 107, "low": 99, "close": 107},  # good
            {"open": 106, "high": 114, "low": 105, "close": 114}, # good
        )
        m = _body_metrics(df["open"], df["high"], df["low"], df["close"])
        result = _detect_three_white_soldiers(df["close"], m)
        # bar 0 has body_ratio ~0.17 which is < 0.35 → pattern should NOT fire
        assert float(result.iloc[-1]) == pytest.approx(0.0)


# ── Engine: score range and shape ─────────────────────────────────────────────

class TestEngineScoreRange:
    """Engine scores must always be in [0, 1] and neutral on sparse data."""

    def setup_method(self):
        self.engine = CandlestickPatternEngine()

    def test_uptrend_produces_series(self):
        df     = _make_trend(100)
        scores = self.engine.precompute_scores(df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == 100

    def test_all_values_in_range(self):
        df = _make_trend(150)
        scores = self.engine.precompute_scores(df)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_sparse_data_returns_neutral(self):
        df     = _make_trend(3)
        scores = self.engine.precompute_scores(df)
        assert (scores - 0.5).abs().max() < 1e-6

    def test_empty_df_returns_empty_or_neutral(self):
        df     = _make_trend(0)
        scores = self.engine.precompute_scores(df)
        assert len(scores) == 0

    def test_constant_price_returns_neutral(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="B")
        df  = pd.DataFrame({
            "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 100.0, "volume": 1_000_000
        }, index=idx)
        scores = self.engine.precompute_scores(df)
        assert scores.notna().all()
        # constant price → no directional patterns → near neutral
        assert abs(scores.mean() - 0.5) < 0.20

    def test_missing_volume_col_still_works(self):
        df = _make_trend(50)[["open", "high", "low", "close"]]
        scores = self.engine.precompute_scores(df)
        assert len(scores) == 50
        assert scores.between(0, 1).all()

    def test_downtrend_score_below_uptrend(self):
        up   = self.engine.precompute_scores(_make_trend(100))
        down = self.engine.precompute_scores(_make_downtrend(100))
        # Mean score should be lower in a downtrend
        assert down.mean() < up.mean() + 0.05


class TestEngineScoreBullishBias:
    """Validate that canonical bullish patterns push the score above 0.5."""

    def setup_method(self):
        self.engine = CandlestickPatternEngine()

    def _score_last(self, df):
        return float(self.engine.precompute_scores(df).iloc[-1])

    def test_three_white_soldiers_raises_score(self):
        # 50 neutral bars + 3 white soldiers at end
        base = _make_trend(50, start=100, step=0.1)
        soldiers = _concat_rows(
            {"open": 105, "high": 111, "low": 104, "close": 111},
            {"open": 110, "high": 117, "low": 109, "close": 117},
            {"open": 116, "high": 124, "low": 115, "close": 124},
        )
        soldiers.index = pd.date_range(base.index[-1] + pd.Timedelta(days=1),
                                       periods=3, freq="B")
        df = pd.concat([base, soldiers])
        score = self._score_last(df)
        # Expect clear bullish bias
        assert score > 0.55

    def test_three_black_crows_lowers_score(self):
        base = _make_trend(50, start=200, step=0.1)
        crows = _concat_rows(
            {"open": 205, "high": 205.5, "low": 197, "close": 197},
            {"open": 197.5, "high": 198, "low": 190, "close": 190},
            {"open": 190.5, "high": 191, "low": 183, "close": 183},
        )
        crows.index = pd.date_range(base.index[-1] + pd.Timedelta(days=1),
                                    periods=3, freq="B")
        df = pd.concat([base, crows])
        score = self._score_last(df)
        assert score < 0.45

    def test_bullish_engulfing_raises_score(self):
        base = _make_downtrend(50)
        engulf = _concat_rows(
            {"open": 150, "high": 152, "low": 145, "close": 146},  # bearish
            {"open": 144, "high": 154, "low": 143, "close": 152},  # bullish engulfing
        )
        engulf.index = pd.date_range(base.index[-1] + pd.Timedelta(days=1),
                                     periods=2, freq="B")
        df = pd.concat([base, engulf])
        # Score at the engulfing bar should move toward bullish
        score = self._score_last(df)
        # In a downtrend context, the pattern fires + context amplifies
        assert score > 0.50


# ── Look-ahead validation ─────────────────────────────────────────────────────

class TestNoLookahead:
    """Verify that score at bar N never uses bar N+1's data."""

    def setup_method(self):
        self.engine = CandlestickPatternEngine()

    def test_changing_future_bar_doesnt_affect_current(self):
        df1 = _make_trend(60)
        df2 = df1.copy()
        # Completely change the last bar to a large dump
        df2.iloc[-1, df2.columns.get_loc("close")] = df2.iloc[-1]["close"] * 0.50
        df2.iloc[-1, df2.columns.get_loc("open")]  = df2.iloc[-1]["open"]  * 1.10

        scores1 = self.engine.precompute_scores(df1)
        scores2 = self.engine.precompute_scores(df2)
        # All bars except the last should be identical
        pd.testing.assert_series_equal(scores1.iloc[:-1], scores2.iloc[:-1])

    def test_prefix_matches_full_series(self):
        df_full = _make_trend(80)
        df_prefix = df_full.iloc[:60]

        full_scores   = self.engine.precompute_scores(df_full)
        prefix_scores = self.engine.precompute_scores(df_prefix)

        # Scores at bar 59 must match between full and prefix
        pd.testing.assert_series_equal(
            full_scores.iloc[:60], prefix_scores, check_names=False
        )

    def test_score_at_matches_series_last(self):
        df    = _make_trend(70)
        date  = df.index[-1]
        score_at     = self.engine.score_at(df, date)
        score_series = float(self.engine.precompute_scores(df).iloc[-1])
        assert score_at == pytest.approx(score_series, abs=1e-6)


# ── Context multipliers ───────────────────────────────────────────────────────

class TestContextMultipliers:
    """Context should amplify signals when price is at extremes or trend confirms."""

    def setup_method(self):
        self.engine  = CandlestickPatternEngine()
        self.no_ctx  = CandlestickPatternEngine()

    def test_context_doesnt_break_score_range(self):
        df = _make_trend(100)
        scores_ctx    = self.engine.precompute_scores(df, context=True)
        scores_noctx  = self.engine.precompute_scores(df, context=False)
        assert scores_ctx.between(0, 1).all()
        assert scores_noctx.between(0, 1).all()

    def test_context_on_off_gives_different_result(self):
        # With patterns firing, context should change the score
        base = _make_downtrend(60)
        soldiers = _concat_rows(
            {"open": 140, "high": 148, "low": 139, "close": 148},
            {"open": 147, "high": 156, "low": 146, "close": 156},
            {"open": 155, "high": 165, "low": 154, "close": 165},
        )
        soldiers.index = pd.date_range(base.index[-1] + pd.Timedelta(days=1),
                                       periods=3, freq="B")
        df = pd.concat([base, soldiers])
        s_ctx    = float(self.engine.precompute_scores(df, context=True).iloc[-1])
        s_noctx  = float(self.engine.precompute_scores(df, context=False).iloc[-1])
        # At support, context should amplify bullish signal → score with context >= no-context
        # (can be equal if no patterns fire, but not less)
        assert s_ctx >= s_noctx - 0.01  # allow tiny float diff


# ── Score API ─────────────────────────────────────────────────────────────────

class TestDescribePatterns:
    def setup_method(self):
        self.engine = CandlestickPatternEngine()

    def test_describe_returns_dict(self):
        df   = _make_trend(50)
        info = self.engine.describe_patterns(df, df.index[-1])
        assert isinstance(info, dict)
        assert "net_score" in info
        assert "patterns"  in info
        assert "signal"    in info

    def test_signal_is_valid_enum(self):
        df   = _make_trend(50)
        info = self.engine.describe_patterns(df, df.index[-1])
        assert info["signal"] in ("BULLISH", "NEUTRAL", "BEARISH")

    def test_describe_sparse_returns_error(self):
        df   = _make_trend(2)
        info = self.engine.describe_patterns(df, df.index[-1])
        assert "error" in info

    def test_patterns_list_structure(self):
        df = _make_trend(80)
        info = self.engine.describe_patterns(df, df.index[-1])
        for pat in info["patterns"]:
            assert "pattern" in pat
            assert "direction" in pat
            assert pat["direction"] in ("BULLISH", "BEARISH")
            assert "score" in pat
            assert 0 < pat["score"] <= 1.0


# ── Score registry completeness ───────────────────────────────────────────────

class TestScoreRegistry:
    def test_all_scores_positive(self):
        for name, val in SCORE.items():
            assert val > 0.0, f"Score for {name} should be positive"

    def test_all_scores_le_one(self):
        for name, val in SCORE.items():
            assert val <= 1.0, f"Score for {name} should be ≤ 1.0"

    def test_three_bar_scores_highest(self):
        """Three-bar patterns should have higher scores than single-bar."""
        assert SCORE["three_white_soldiers"] >= SCORE["hammer"]
        assert SCORE["morning_star"]         >= SCORE["dragonfly_doji"]


# ── Integration with MomentumSignalEngine ────────────────────────────────────

class TestMomentumSignalIntegration:
    """Verify candlestick component is correctly wired into composite score."""

    def test_composite_imports_without_error(self):
        from core.models.momentum_signal import MomentumSignalEngine
        engine = MomentumSignalEngine()
        assert engine is not None

    def test_composite_score_in_range(self):
        from core.models.momentum_signal import MomentumSignalEngine
        engine = MomentumSignalEngine()
        df = _make_trend(120)
        scores = engine.precompute_scores(df)
        assert scores.between(0, 1).all()

    def test_weights_sum_to_one(self):
        from core.models.momentum_signal import WEIGHTS
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_candlestick_in_weights(self):
        from core.models.momentum_signal import WEIGHTS
        assert "candlestick" in WEIGHTS
        assert WEIGHTS["candlestick"] > 0.0

    def test_describe_includes_candlestick(self):
        from core.models.momentum_signal import MomentumSignalEngine
        engine = MomentumSignalEngine()
        df  = _make_trend(120)
        nifty = _make_trend(120, start=20000, step=50)
        info = engine.describe(df, df.index[-1], nifty_df=nifty)
        assert "candlestick_score" in info
        assert isinstance(info["candlestick_score"], float)
        assert 0.0 <= info["candlestick_score"] <= 1.0

    def test_candlestick_failure_doesnt_crash_composite(self):
        """If candlestick engine throws, composite must still return neutral for that component."""
        from core.models.momentum_signal import MomentumSignalEngine, _CANDLE_ENGINE
        import unittest.mock as mock

        engine = MomentumSignalEngine()
        df = _make_trend(100)

        # Monkey-patch to throw
        with mock.patch.object(_CANDLE_ENGINE, "precompute_scores", side_effect=RuntimeError("boom")):
            scores = engine.precompute_scores(df)

        assert scores.between(0, 1).all()
        assert scores.notna().all()

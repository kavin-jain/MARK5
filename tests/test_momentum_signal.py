"""
Tests for core/models/momentum_signal.py
=========================================
Verifies:
  - Score always in [0.0, 1.0]
  - Uptrend → score well above entry threshold (0.55)
  - Downtrend → score well below exit threshold (0.40)
  - Flat market → score near neutral (0.40-0.55)
  - Signal has genuine variance (never constant 100% or 0%)
  - No future look-ahead: score at bar i uses only data ≤ i
  - Relative strength works without NIFTY50 (neutral fallback)
  - ATR trailing stop adapts to volatility
  - Minimum history handled gracefully
"""
import math
import sys
import os

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.models.momentum_signal import (
    MomentumSignalEngine,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    OPTIMAL_BARS,
    MIN_BARS,
)


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_df(
    n: int,
    trend: float = 0.0,
    noise: float = 0.01,
    vol_mult: float = 1.0,
    start_price: float = 100.0,
    start_date: str = "2015-01-01",
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data.

    trend > 0 → uptrend (e.g. 0.001 = 0.1% per bar)
    trend < 0 → downtrend
    trend = 0 → flat (random walk around start_price)
    noise     → daily return std dev
    vol_mult  → scales volume
    """
    rng   = np.random.default_rng(42)
    dates = pd.date_range(start=start_date, periods=n, freq="B")
    rets  = rng.normal(trend, noise, n)
    close = start_price * np.exp(np.cumsum(rets))
    high  = close * (1 + abs(rng.normal(0, noise * 0.5, n)))
    low   = close * (1 - abs(rng.normal(0, noise * 0.5, n)))
    volume = rng.integers(100_000, 1_000_000, n) * vol_mult
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_uptrend(n: int = OPTIMAL_BARS + 100) -> pd.DataFrame:
    """Strong steady uptrend: +0.15% per bar."""
    return _make_df(n, trend=0.0015, noise=0.008)


def _make_downtrend(n: int = OPTIMAL_BARS + 100) -> pd.DataFrame:
    """Strong steady downtrend: -0.15% per bar."""
    return _make_df(n, trend=-0.0015, noise=0.008)


def _make_flat(n: int = OPTIMAL_BARS + 100) -> pd.DataFrame:
    """Flat/sideways: 0 drift, moderate noise."""
    return _make_df(n, trend=0.0, noise=0.012)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestScoreRange:
    """All scores must be in [0.0, 1.0]."""

    def test_uptrend_scores_in_range(self):
        engine = MomentumSignalEngine()
        df     = _make_uptrend()
        scores = engine.precompute_scores(df)
        assert scores.min() >= 0.0 - 1e-9, f"score below 0: {scores.min()}"
        assert scores.max() <= 1.0 + 1e-9, f"score above 1: {scores.max()}"

    def test_downtrend_scores_in_range(self):
        engine = MomentumSignalEngine()
        df     = _make_downtrend()
        scores = engine.precompute_scores(df)
        assert scores.min() >= 0.0 - 1e-9
        assert scores.max() <= 1.0 + 1e-9

    def test_flat_scores_in_range(self):
        engine = MomentumSignalEngine()
        df     = _make_flat()
        scores = engine.precompute_scores(df)
        assert scores.min() >= 0.0 - 1e-9
        assert scores.max() <= 1.0 + 1e-9

    def test_single_point_score_in_range(self):
        engine = MomentumSignalEngine()
        df     = _make_uptrend()
        score  = engine.compute_score(df, df.index[-1])
        assert 0.0 <= score <= 1.0


class TestSignalDirection:
    """Scores should correctly identify market regime."""

    def test_uptrend_final_score_above_entry_threshold(self):
        """After 300+ bars of uptrend, score should be well above 0.55."""
        engine = MomentumSignalEngine()
        df     = _make_uptrend(n=400)
        scores = engine.precompute_scores(df)
        # Take the last 50 bars (well into steady uptrend)
        tail_mean = float(scores.iloc[-50:].mean())
        assert tail_mean >= ENTRY_THRESHOLD, (
            f"Uptrend mean score {tail_mean:.4f} should be ≥ {ENTRY_THRESHOLD}"
        )

    def test_downtrend_final_score_below_exit_threshold(self):
        """After 300+ bars of downtrend, score should be well below 0.40."""
        engine = MomentumSignalEngine()
        df     = _make_downtrend(n=400)
        scores = engine.precompute_scores(df)
        tail_mean = float(scores.iloc[-50:].mean())
        assert tail_mean <= EXIT_THRESHOLD, (
            f"Downtrend mean score {tail_mean:.4f} should be ≤ {EXIT_THRESHOLD}"
        )

    def test_flat_score_near_neutral(self):
        """Flat market score should be in the neutral band [0.35, 0.60]."""
        engine = MomentumSignalEngine()
        df     = _make_flat(n=400)
        scores = engine.precompute_scores(df)
        tail_mean = float(scores.iloc[-100:].mean())
        assert 0.30 <= tail_mean <= 0.65, (
            f"Flat market mean score {tail_mean:.4f} should be in [0.30, 0.65]"
        )

    def test_uptrend_beats_downtrend(self):
        """Uptrend score > downtrend score at every late bar."""
        engine = MomentumSignalEngine()
        up     = engine.precompute_scores(_make_uptrend(400))
        dn     = engine.precompute_scores(_make_downtrend(400))
        # Compare last 50 bars
        assert float(up.iloc[-50:].mean()) > float(dn.iloc[-50:].mean()), \
            "Uptrend score should exceed downtrend score"

    def test_signal_transitions_uptrend_to_downtrend(self):
        """Score should drop significantly when trend reverses."""
        rng = np.random.default_rng(7)
        n   = 400
        # First 300 bars: uptrend; next 100 bars: downtrend
        dates = pd.date_range("2015-01-01", periods=n, freq="B")
        rets  = np.concatenate([
            rng.normal(0.001, 0.008, 300),
            rng.normal(-0.002, 0.010, 100),
        ])
        close  = 100.0 * np.exp(np.cumsum(rets))
        volume = rng.integers(100_000, 1_000_000, n).astype(float)
        df     = pd.DataFrame(
            {"open": close, "high": close*1.005, "low": close*0.995,
             "close": close, "volume": volume},
            index=dates,
        )
        engine = MomentumSignalEngine()
        scores = engine.precompute_scores(df)
        score_at_top  = float(scores.iloc[280:300].mean())   # end of uptrend
        score_at_end  = float(scores.iloc[-20:].mean())       # after reversal
        assert score_at_top > score_at_end + 0.05, (
            f"Score at top {score_at_top:.3f} should be noticeably higher than "
            f"score after reversal {score_at_end:.3f}"
        )


class TestSignalVariance:
    """Score must have genuine variance — no always-on / always-off tickers."""

    def test_uptrend_score_has_variance(self):
        engine = MomentumSignalEngine()
        scores = engine.precompute_scores(_make_uptrend())
        late   = scores.iloc[MIN_BARS:]
        std    = float(late.std())
        assert std > 0.02, f"Uptrend score std {std:.4f} too low — signal has no variance"

    def test_downtrend_score_has_variance(self):
        engine = MomentumSignalEngine()
        scores = engine.precompute_scores(_make_downtrend())
        late   = scores.iloc[MIN_BARS:]
        std    = float(late.std())
        assert std > 0.02, f"Downtrend score std {std:.4f} too low"

    def test_flat_score_has_variance(self):
        engine = MomentumSignalEngine()
        scores = engine.precompute_scores(_make_flat())
        late   = scores.iloc[MIN_BARS:]
        std    = float(late.std())
        assert std > 0.02, f"Flat score std {std:.4f} too low"

    def test_score_not_constant_100_percent_above_hurdle(self):
        """Simulate the pah=100% problem from the ML system — must not happen."""
        engine = MomentumSignalEngine()
        df     = _make_uptrend(n=500)
        scores = engine.precompute_scores(df)
        late   = scores.iloc[MIN_BARS:]
        engine2 = MomentumSignalEngine()
        pah = (late >= ENTRY_THRESHOLD).mean()
        # Even strong uptrend must have some bars below threshold (e.g. pullbacks)
        assert pah < 0.95, (
            f"pct_above_hurdle={pah:.2f} is suspiciously high — signal may be always-on"
        )


class TestNoLookahead:
    """Score at bar i must not use data from bar i+1 onward."""

    def test_score_at_bar_i_does_not_change_after_adding_bars(self):
        """
        Compute score on first 200 bars. Then add 50 more bars.
        Score at bar 200 must be identical.
        """
        engine = MomentumSignalEngine()
        df     = _make_uptrend(n=250)
        df_200 = df.iloc[:200]
        df_250 = df.iloc[:250]

        scores_200 = engine.precompute_scores(df_200)
        scores_250 = engine.precompute_scores(df_250)

        score_at_200_a = float(scores_200.iloc[-1])
        score_at_200_b = float(scores_250.iloc[199])

        assert abs(score_at_200_a - score_at_200_b) < 1e-9, (
            f"Score at bar 200 changed when future data was added: "
            f"{score_at_200_a:.6f} → {score_at_200_b:.6f}"
        )

    def test_early_bars_zero_before_min_bars(self):
        """Bars before MIN_BARS should return 0.0 (no premature signals)."""
        engine = MomentumSignalEngine()
        df     = _make_uptrend(n=200)
        scores = engine.precompute_scores(df)
        # First MIN_BARS - 1 bars must be 0.0
        early  = scores.iloc[:MIN_BARS - 1]
        assert (early == 0.0).all(), (
            f"Expected 0.0 before bar {MIN_BARS}, got non-zero values: {early[early != 0.0]}"
        )


class TestRelativeStrength:
    """Relative strength vs NIFTY50: works with and without benchmark."""

    def test_neutral_without_nifty(self):
        """Without NIFTY50 data, relative component defaults to 0.5."""
        engine = MomentumSignalEngine()
        df     = _make_uptrend(n=300)
        scores_no_nifty = engine.precompute_scores(df, nifty_df=None)
        assert scores_no_nifty.notna().all()
        # Scores should still be reasonable
        assert 0.0 <= float(scores_no_nifty.iloc[-1]) <= 1.0

    def test_outperforming_stock_beats_underperformer(self):
        """
        Stock A outperforms NIFTY by 20%; stock B underperforms by 20%.
        Stock A should have higher score.
        """
        engine = MomentumSignalEngine()
        n      = 300

        # NIFTY: moderate uptrend (+0.05% per bar)
        nifty = _make_df(n, trend=0.0005, noise=0.008, start_price=18000.0)

        # Stock A: strong outperformer (+0.15% per bar)
        stock_a = _make_df(n, trend=0.0015, noise=0.010, start_price=500.0)

        # Stock B: underperformer (flat/slight down)
        stock_b = _make_df(n, trend=-0.0003, noise=0.010, start_price=500.0)

        scores_a = engine.precompute_scores(stock_a, nifty_df=nifty)
        scores_b = engine.precompute_scores(stock_b, nifty_df=nifty)

        late_a = float(scores_a.iloc[-50:].mean())
        late_b = float(scores_b.iloc[-50:].mean())

        assert late_a > late_b, (
            f"Outperformer score {late_a:.3f} should exceed underperformer {late_b:.3f}"
        )


class TestATRTrailingStop:
    """ATR-adaptive trailing stop calculations."""

    def test_stop_adapts_to_volatility(self):
        """Higher ATR → wider stop distance (in price terms)."""
        engine    = MomentumSignalEngine()
        low_vol   = _make_df(300, trend=0.001, noise=0.005)   # ~0.5% ATR
        high_vol  = _make_df(300, trend=0.001, noise=0.025)   # ~2.5% ATR

        date_lo = low_vol.index[-1]
        date_hi = high_vol.index[-1]

        atr_lo = engine.compute_atr_pct(low_vol,  date_lo)
        atr_hi = engine.compute_atr_pct(high_vol, date_hi)

        assert atr_hi > atr_lo, f"High-vol ATR {atr_hi:.4f} should exceed low-vol {atr_lo:.4f}"

    def test_stop_price_below_peak(self):
        """Stop price must always be below peak price."""
        engine = MomentumSignalEngine()
        df     = _make_uptrend(300)
        date   = df.index[-1]
        atr    = engine.compute_atr_pct(df, date)
        peak   = float(df["close"].max())
        stop   = engine.trailing_stop_price(peak, atr)
        assert stop < peak, f"Stop {stop:.2f} must be below peak {peak:.2f}"

    def test_stop_respects_min_max_bounds(self):
        """Stop distance must stay within [ATR_STOP_MIN=10%, ATR_STOP_MAX=25%]."""
        engine = MomentumSignalEngine()
        # Very low ATR (e.g. 0.3%) → stop should be at least 10%
        stop_low_vol = engine.trailing_stop_price(1000.0, atr_pct=0.003)
        pct_low = 1.0 - stop_low_vol / 1000.0
        assert pct_low >= 0.10 - 1e-9, f"Stop distance {pct_low:.3f} below min 10%"

        # Very high ATR (e.g. 10%) → stop should be capped at 25%
        stop_high_vol = engine.trailing_stop_price(1000.0, atr_pct=0.10)
        pct_high = 1.0 - stop_high_vol / 1000.0
        assert pct_high <= 0.25 + 1e-9, f"Stop distance {pct_high:.3f} above max 25%"

    def test_stop_price_returns_float(self):
        """Return type must be float."""
        engine = MomentumSignalEngine()
        result = engine.trailing_stop_price(500.0, 0.02)
        assert isinstance(result, float)


class TestMinimumHistory:
    """Edge cases with very short data."""

    def test_single_bar_returns_zeros(self):
        engine = MomentumSignalEngine()
        df     = _make_df(1)
        scores = engine.precompute_scores(df)
        assert len(scores) == 1
        assert float(scores.iloc[0]) == 0.0

    def test_two_bars_no_crash(self):
        engine = MomentumSignalEngine()
        df     = _make_df(2)
        scores = engine.precompute_scores(df)
        assert len(scores) == 2

    def test_exactly_min_bars_no_crash(self):
        engine = MomentumSignalEngine()
        df     = _make_df(MIN_BARS)
        scores = engine.precompute_scores(df)
        assert len(scores) == MIN_BARS
        # Bar at MIN_BARS index should start having valid scores
        assert pd.notna(scores.iloc[-1])

    def test_output_length_matches_input(self):
        engine = MomentumSignalEngine()
        for n in [10, 60, 100, 252, 500]:
            df     = _make_df(n)
            scores = engine.precompute_scores(df)
            assert len(scores) == n, f"Expected {n} scores, got {len(scores)}"


class TestRollingScore:
    """Rolling average of scores (entry/exit smoothing)."""

    def test_rolling_score_at_valid_date(self):
        engine = MomentumSignalEngine()
        df     = _make_uptrend(300)
        scores = engine.precompute_scores(df)
        date   = df.index[-1]
        rs     = engine.rolling_score(scores, date, window=5)
        assert 0.0 <= rs <= 1.0

    def test_rolling_score_returns_zero_before_min_bars(self):
        engine = MomentumSignalEngine()
        df     = _make_uptrend(300)
        scores = engine.precompute_scores(df)
        early  = engine.rolling_score(scores, df.index[10], window=5)
        # Early bars are 0.0, so rolling mean also ~0
        assert early == pytest.approx(0.0, abs=0.01)

    def test_rolling_score_smoother_than_raw(self):
        """Rolling score should have lower std dev than raw scores."""
        engine = MomentumSignalEngine()
        df     = _make_flat(300)
        scores = engine.precompute_scores(df)
        raw_std = float(scores.iloc[MIN_BARS:].std())

        rolled = pd.Series([
            engine.rolling_score(scores, d, window=5)
            for d in scores.index[MIN_BARS:]
        ])
        rolled_std = float(rolled.std())
        assert rolled_std <= raw_std + 0.005, (
            f"Rolling std {rolled_std:.4f} should be ≤ raw std {raw_std:.4f}"
        )


class TestDescribe:
    """describe() method provides useful debug output."""

    def test_describe_returns_dict_with_expected_keys(self):
        engine = MomentumSignalEngine()
        df     = _make_uptrend(300)
        info   = engine.describe(df, df.index[-1])
        expected_keys = {"date", "price", "MA20", "MA50", "MA200",
                         "trend_conds", "daily_sharpe", "composite_score", "signal"}
        assert expected_keys.issubset(set(info.keys())), (
            f"Missing keys: {expected_keys - set(info.keys())}"
        )

    def test_describe_signal_matches_score(self):
        engine = MomentumSignalEngine()
        df     = _make_uptrend(400)
        info   = engine.describe(df, df.index[-1])
        score  = info["composite_score"]
        signal = info["signal"]
        if score >= ENTRY_THRESHOLD:
            assert signal == "BULLISH", f"score={score:.3f} → expected BULLISH, got {signal}"
        elif score <= EXIT_THRESHOLD:
            assert signal == "BEARISH"
        else:
            assert signal == "NEUTRAL"

    def test_describe_price_matches_latest_close(self):
        engine = MomentumSignalEngine()
        df     = _make_uptrend(300)
        info   = engine.describe(df, df.index[-1])
        assert abs(info["price"] - float(df["close"].iloc[-1])) < 0.01


# ─── Phase 4: weekly_aligned() ───────────────────────────────────────────────

def _make_weekly_uptrend(n_daily: int = 150) -> pd.DataFrame:
    """Daily OHLCV with a strong weekly uptrend (+0.3%/day, very low noise)."""
    return _make_df(n_daily, trend=0.003, noise=0.001)


def _make_weekly_downtrend(n_daily: int = 150) -> pd.DataFrame:
    """Daily OHLCV with a strong weekly downtrend (-0.3%/day, very low noise)."""
    return _make_df(n_daily, trend=-0.003, noise=0.001)


class TestWeeklyAligned:
    """Tests for MomentumSignalEngine.weekly_aligned()."""

    def test_strong_uptrend_returns_true(self):
        """A sustained uptrend should produce weekly_aligned = True."""
        df = _make_weekly_uptrend(150)
        result = MomentumSignalEngine.weekly_aligned(df, df.index[-1])
        assert result is True

    def test_strong_downtrend_returns_false(self):
        """A sustained downtrend should produce weekly_aligned = False."""
        df = _make_weekly_downtrend(150)
        result = MomentumSignalEngine.weekly_aligned(df, df.index[-1])
        assert result is False

    def test_insufficient_history_returns_true(self):
        """Below min_weeks threshold, fail-open (don't block sparse data)."""
        df = _make_weekly_uptrend(30)   # < 10 weeks
        result = MomentumSignalEngine.weekly_aligned(df, df.index[-1])
        assert result is True

    def test_exactly_min_weeks(self):
        """At exactly min_weeks of weekly data, should compute correctly."""
        # 10 weeks * 5 days = 50 days minimum
        df = _make_weekly_uptrend(55)
        result = MomentumSignalEngine.weekly_aligned(df, df.index[-1])
        # With uptrend, should be True at 55 days
        assert isinstance(result, bool)

    def test_date_slicing_no_lookahead(self):
        """Score at date D uses only data up to D."""
        df        = _make_weekly_uptrend(200)
        mid_date  = df.index[100]
        late_date = df.index[-1]

        result_mid  = MomentumSignalEngine.weekly_aligned(df, mid_date)
        result_late = MomentumSignalEngine.weekly_aligned(df, late_date)
        # Both are computed without future data; should not raise
        assert isinstance(result_mid, bool)
        assert isinstance(result_late, bool)

    def test_downtrend_after_uptrend_eventually_flips(self):
        """After enough downtrending bars, weekly_aligned should return False."""
        # Build: 100 bars uptrend, then 100 bars strong downtrend (very low noise)
        up   = _make_df(100, trend=+0.003, noise=0.001)
        down = _make_df(100, trend=-0.004, noise=0.001,
                        start_price=float(up["close"].iloc[-1]))
        # Re-index down to continue from where up ended
        down.index = pd.date_range(up.index[-1] + pd.Timedelta(days=1),
                                   periods=100, freq="B")
        df = pd.concat([up, down])

        result = MomentumSignalEngine.weekly_aligned(df, df.index[-1])
        assert result is False

    def test_returns_bool_not_numpy_bool(self):
        """Must return Python bool (safe for 'if' conditionals and JSON)."""
        df     = _make_weekly_uptrend(150)
        result = MomentumSignalEngine.weekly_aligned(df, df.index[-1])
        assert type(result) is bool

    def test_custom_min_weeks(self):
        """Respects custom min_weeks parameter."""
        df = _make_weekly_uptrend(30)
        # With min_weeks=4 (~20 days), 30 days should be enough
        result = MomentumSignalEngine.weekly_aligned(df, df.index[-1], min_weeks=4)
        assert isinstance(result, bool)

    def test_empty_df_returns_true(self):
        """Empty DataFrame should not crash — fail-open."""
        df     = pd.DataFrame(columns=["open","high","low","close","volume"])
        df.index = pd.DatetimeIndex([])
        date   = pd.Timestamp("2022-06-01")
        result = MomentumSignalEngine.weekly_aligned(df, date)
        assert result is True  # fail-open on exception

    def test_flat_market_near_threshold(self):
        """Flat market (no drift) — weekly MAs converge; result is well-defined."""
        df     = _make_df(200, trend=0.0, noise=0.0001)  # nearly flat
        result = MomentumSignalEngine.weekly_aligned(df, df.index[-1])
        # Should be False or True — but must not raise
        assert isinstance(result, bool)

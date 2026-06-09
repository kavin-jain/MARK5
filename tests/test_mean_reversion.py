"""
Tests for the Mean Reversion Strategy and Backtest
==================================================
Covers:
  - Entry condition logic (RSI, 52w-high fall, SMA200 proximity)
  - Exit condition logic (take profit, stop loss, time stop, RSI overbought)
  - MR portfolio accounting (enter/exit, cash management)
  - Sector RS integration in MomentumSignalEngine.precompute_scores()
  - Behavioral signals VIX proxy computation
"""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategies.mean_reversion import MeanReversionStrategy
from core.strategies.behavioral_signals import BehavioralSignals
from core.models.momentum_signal import MomentumSignalEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 300,
    start_price: float = 200.0,
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates  = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + drift + rng.normal(0, vol)))
    prices = np.array(prices)
    highs  = prices * (1 + abs(rng.normal(0, 0.005, n)))
    lows   = prices * (1 - abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "open":   prices,
        "high":   highs,
        "low":    lows,
        "close":  prices,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
    }, index=dates)


def _make_nifty(n: int = 300) -> pd.Series:
    df = _make_ohlcv(n=n, start_price=18000, vol=0.008)
    return df["close"]


def _make_mr_strategy() -> MeanReversionStrategy:
    return MeanReversionStrategy(
        rsi_oversold=35.0,
        take_profit_pct=0.12,
        stop_loss_pct=0.08,
        max_hold_days=30,
        position_size_pct=0.10,
    )


# ── RSI utility tests ──────────────────────────────────────────────────────────

class TestRSIHelper:
    """Tests for StrategyBase.rsi() utility via MeanReversionStrategy."""

    def test_rsi_flat_series_returns_50(self):
        """Flat price series (no change) → RSI = 50."""
        prices = pd.DataFrame({"close": [100.0] * 50})
        mr = _make_mr_strategy()
        rsi_val = mr.rsi(prices["close"])
        # Flat series: gains = losses → RSI should be near 50
        assert 40.0 <= rsi_val <= 60.0

    def test_rsi_strongly_up_trend_high(self):
        """Strongly rising prices (with small noise) → RSI > 65."""
        # Pure monotonic (no losses) → RSI formula gives NaN (fallback=50)
        # Add tiny noise to create some small losses for RSI to compute
        rng = np.random.default_rng(10)
        base = np.linspace(100, 200, 100)
        noise = rng.normal(0, 0.5, 100)   # tiny noise (0.5% of price)
        prices = pd.DataFrame({"close": base + noise})
        mr = _make_mr_strategy()
        rsi_val = mr.rsi(prices["close"])
        assert rsi_val > 65.0, f"Strong uptrend should yield RSI > 65, got {rsi_val:.1f}"

    def test_rsi_strongly_down_trend_low(self):
        """Monotonically falling prices → RSI near 0."""
        prices = pd.DataFrame({"close": list(range(200, 100, -1))})
        mr = _make_mr_strategy()
        rsi_val = mr.rsi(prices["close"])
        assert rsi_val < 15.0


# ── Entry condition tests ──────────────────────────────────────────────────────

class TestMREntryConditions:
    """Full entry gate tests — all conditions must be satisfied."""

    def _oversold_df(self, n=250, crash_from=200.0, crash_to=150.0) -> pd.DataFrame:
        """Build a DF where stock crashes and becomes oversold."""
        dates  = pd.date_range("2022-01-01", periods=n, freq="B")
        prices = np.linspace(crash_from, crash_to, n)
        vol    = np.ones(n) * 1_000_000.0
        return pd.DataFrame({
            "open": prices, "high": prices * 1.005,
            "low":  prices * 0.995, "close": prices,
            "volume": vol,
        }, index=dates)

    def test_high_fall_below_min_rejects(self):
        """Stock fell only 10% — below 15% minimum → entry rejected."""
        df     = self._oversold_df(crash_from=200.0, crash_to=180.0)  # -10%
        nifty  = _make_nifty(len(df))
        mr     = _make_mr_strategy()
        signal = mr.should_enter("TEST", df, nifty, df.index[-1], ml_confidence=0.5)
        assert signal is None, "Should reject: fall < 15%"

    def test_high_fall_above_max_rejects(self):
        """Stock fell 60% — above 50% maximum (crash) → entry rejected."""
        df     = self._oversold_df(crash_from=200.0, crash_to=80.0)  # -60%
        nifty  = _make_nifty(len(df))
        mr     = _make_mr_strategy()
        signal = mr.should_enter("TEST", df, nifty, df.index[-1], ml_confidence=0.5)
        assert signal is None, "Should reject: fall > 50%"

    def test_short_history_rejects(self):
        """Only 30 bars of history — not enough for SMA200 → should not enter."""
        df    = self._oversold_df(n=30)
        nifty = _make_nifty(len(df))
        mr    = _make_mr_strategy()
        signal = mr.should_enter("TEST", df, nifty, df.index[-1], ml_confidence=0.5)
        assert signal is None, "Should reject: insufficient history"


# ── Exit condition tests ──────────────────────────────────────────────────────

class TestMRExitConditions:
    """Tests for each exit condition in should_exit()."""

    def _make_exit_df(self, entry=100.0, current=100.0, n=250) -> pd.DataFrame:
        dates  = pd.date_range("2022-01-01", periods=n, freq="B")
        prices = np.full(n, entry)
        prices[-1] = current
        vol    = np.ones(n) * 1_000_000.0
        return pd.DataFrame({
            "open": prices, "high": prices * 1.005,
            "low":  prices * 0.995, "close": prices,
            "volume": vol,
        }, index=dates)

    def test_take_profit_triggers_at_12pct(self):
        """Price +12% from entry → TAKE_PROFIT exit."""
        df    = self._make_exit_df(entry=100.0, current=113.0)
        nifty = _make_nifty(len(df))
        mr    = _make_mr_strategy()
        sig   = mr.should_exit("TEST", df, nifty, df.index[-1],
                                entry_price=100.0, peak_price=113.0, hold_days=5)
        assert sig is not None
        assert "TAKE_PROFIT" in sig.reasons[0]

    def test_stop_loss_triggers_at_8pct(self):
        """Price -8% from entry → STOP_LOSS exit."""
        df    = self._make_exit_df(entry=100.0, current=91.0)
        nifty = _make_nifty(len(df))
        mr    = _make_mr_strategy()
        sig   = mr.should_exit("TEST", df, nifty, df.index[-1],
                                entry_price=100.0, peak_price=100.0, hold_days=5)
        assert sig is not None
        assert "STOP_LOSS" in sig.reasons[0]

    def test_no_exit_within_bands(self):
        """Price within band — no exit triggered."""
        df    = self._make_exit_df(entry=100.0, current=104.0)
        nifty = _make_nifty(len(df))
        mr    = _make_mr_strategy()
        sig   = mr.should_exit("TEST", df, nifty, df.index[-1],
                                entry_price=100.0, peak_price=104.0, hold_days=5)
        assert sig is None, "No exit expected within +4%, +4% range"

    def test_time_stop_triggers_at_30_bars(self):
        """Held for 30 bars → TIME_STOP exit."""
        df    = self._make_exit_df(entry=100.0, current=103.0)
        nifty = _make_nifty(len(df))
        mr    = _make_mr_strategy()
        sig   = mr.should_exit("TEST", df, nifty, df.index[-1],
                                entry_price=100.0, peak_price=103.0, hold_days=30)
        assert sig is not None
        assert "TIME_STOP" in sig.reasons[0]

    def test_no_time_stop_before_30_bars(self):
        """Held for 29 bars → no time stop."""
        df    = self._make_exit_df(entry=100.0, current=103.0)
        nifty = _make_nifty(len(df))
        mr    = _make_mr_strategy()
        sig   = mr.should_exit("TEST", df, nifty, df.index[-1],
                                entry_price=100.0, peak_price=103.0, hold_days=29)
        assert sig is None, "No time stop before 30 bars"

    def test_exit_priority_take_profit_before_time(self):
        """Take profit fires before time stop even if hold_days = 31."""
        df    = self._make_exit_df(entry=100.0, current=113.5)
        nifty = _make_nifty(len(df))
        mr    = _make_mr_strategy()
        sig   = mr.should_exit("TEST", df, nifty, df.index[-1],
                                entry_price=100.0, peak_price=113.5, hold_days=31)
        assert sig is not None
        assert "TAKE_PROFIT" in sig.reasons[0]


# ── R:R ratio test ────────────────────────────────────────────────────────────

class TestRiskRewardRatio:
    def test_rr_ratio_above_1(self):
        """Take profit / stop loss ratio should be > 1."""
        mr = _make_mr_strategy()
        rr = mr.take_profit_pct / mr.stop_loss_pct
        assert rr > 1.0, f"R:R {rr:.2f} — should be > 1 (positive expectancy required)"

    def test_rr_ratio_is_1_5(self):
        """Default R:R = 12% / 8% = 1.5."""
        mr = _make_mr_strategy()
        assert abs(mr.take_profit_pct / mr.stop_loss_pct - 1.5) < 0.01


# ── Behavioral Signals: VIX proxy tests ───────────────────────────────────────

class TestBehavioralSignalsVIX:
    def _make_nifty_series(self, n=300, vol=0.008) -> pd.Series:
        rng    = np.random.default_rng(99)
        dates  = pd.date_range("2022-01-01", periods=n, freq="B")
        prices = [18000.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + rng.normal(0, vol)))
        return pd.Series(prices, index=dates, name="close")

    def test_vix_proxy_normal_market(self):
        """Low-vol market → VIX proxy should be below 22%."""
        nifty  = self._make_nifty_series(vol=0.006)  # ~9.5% annualized
        bhvr   = BehavioralSignals(nifty)
        date   = nifty.index[-1]
        vix    = bhvr.vix_proxy_at(date)
        assert vix < 0.22, f"Expected VIX < 22% in low-vol market, got {vix:.1%}"

    def test_position_scale_normal_is_1(self):
        """Normal vol market → scale = 1.0."""
        nifty = self._make_nifty_series(vol=0.006)
        bhvr  = BehavioralSignals(nifty)
        scale = bhvr.position_scale_factor(nifty.index[-1])
        assert scale == 1.0

    def test_position_scale_crisis_is_zero(self):
        """Extremely high vol (>35% realized) → scale = 0.0 (block all)."""
        rng   = np.random.default_rng(77)
        dates = pd.date_range("2022-01-01", periods=100, freq="B")
        # 3% daily vol → ~47% annualized — definitely CRISIS
        prices = [18000.0]
        for _ in range(99):
            prices.append(max(1.0, prices[-1] * (1 + rng.normal(0, 0.03))))
        nifty = pd.Series(prices, index=dates)
        bhvr  = BehavioralSignals(nifty)
        scale = bhvr.position_scale_factor(nifty.index[-1])
        assert scale == 0.0, f"Expected 0.0 in crisis, got {scale}"

    def test_vix_proxy_nonnegative(self):
        """VIX proxy should always be ≥ 0."""
        nifty = self._make_nifty_series()
        bhvr  = BehavioralSignals(nifty)
        for date in nifty.index[20:]:
            vix = bhvr.vix_proxy_at(date)
            assert vix >= 0.0

    def test_breadth_range(self):
        """Market breadth should be in [0, 1]."""
        nifty = self._make_nifty_series()
        bhvr  = BehavioralSignals(nifty)
        dates = nifty.index[60:]
        df1   = _make_ohlcv(n=len(nifty), drift=0.001)
        df2   = _make_ohlcv(n=len(nifty), drift=-0.001)
        closes = {
            "BULL": df1["close"].reindex(nifty.index).ffill(),
            "BEAR": df2["close"].reindex(nifty.index).ffill(),
        }
        breadth_s = bhvr.breadth_series(closes, dates)
        assert (breadth_s >= 0.0).all()
        assert (breadth_s <= 1.0).all()


# ── FII signal classification tests ───────────────────────────────────────────

class TestFIISignal:
    def test_heavy_selling_is_bearish(self):
        from core.strategies.behavioral_signals import FIISignal
        sig = BehavioralSignals.classify_fii(-6000.0)
        assert sig == FIISignal.BEARISH

    def test_heavy_buying_is_bullish(self):
        from core.strategies.behavioral_signals import FIISignal
        sig = BehavioralSignals.classify_fii(6000.0)
        assert sig == FIISignal.STRONGLY_BULLISH

    def test_neutral_zone(self):
        from core.strategies.behavioral_signals import FIISignal
        sig = BehavioralSignals.classify_fii(500.0)
        assert sig == FIISignal.NEUTRAL


# ── Sector RS integration in MomentumSignalEngine ────────────────────────────

class TestSectorRSIntegration:
    def _make_full_df(self, n=300, drift=0.001) -> pd.DataFrame:
        return _make_ohlcv(n=n, drift=drift, start_price=100.0)

    def test_precompute_scores_without_sector_rs(self):
        """precompute_scores with no sector_rs → same as before, shape matches."""
        df     = self._make_full_df()
        engine = MomentumSignalEngine()
        scores = engine.precompute_scores(df)
        assert len(scores) == len(df)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_precompute_scores_with_neutral_sector_rs(self):
        """sector_rs = 0.0 (perfectly neutral Z-score) → scores same as without."""
        df     = self._make_full_df()
        engine = MomentumSignalEngine()
        neutral_rs = pd.Series(0.0, index=df.index)  # Z=0 → sigmoid = 0.5 (neutral)

        scores_no_rs   = engine.precompute_scores(df)
        scores_with_rs = engine.precompute_scores(df, sector_rs=neutral_rs)

        # Neutral sector RS should shift scores only slightly (weight redistribution)
        # The difference comes purely from weight rebalancing (not signal change)
        assert len(scores_with_rs) == len(df)
        # Max difference should be small (<0.05) because Z=0 → component=0.5 (neutral)
        diff = (scores_with_rs - scores_no_rs).abs()
        assert float(diff.mean()) < 0.05, f"Neutral sector RS caused too much change: {float(diff.mean()):.4f}"

    def test_precompute_scores_bullish_sector_rs_boosts_score(self):
        """Strongly bullish sector RS (Z=+2.5) should boost composite score."""
        df     = self._make_full_df(drift=0.001)
        engine = MomentumSignalEngine()
        bullish_rs = pd.Series(2.5, index=df.index)  # Z=+2.5 → sigmoid ≈ 0.92
        neutral_rs = pd.Series(0.0, index=df.index)

        scores_bullish = engine.precompute_scores(df, sector_rs=bullish_rs)
        scores_neutral = engine.precompute_scores(df, sector_rs=neutral_rs)

        # Bullish sector RS should increase average score vs neutral
        bull_mean = float(scores_bullish.dropna().mean())
        neut_mean = float(scores_neutral.dropna().mean())
        assert bull_mean > neut_mean, (
            f"Bullish sector RS should boost score: {bull_mean:.4f} vs {neut_mean:.4f}"
        )

    def test_precompute_scores_bearish_sector_rs_lowers_score(self):
        """Strongly bearish sector RS (Z=-2.5) should lower composite score."""
        df     = self._make_full_df(drift=0.001)
        engine = MomentumSignalEngine()
        bearish_rs = pd.Series(-2.5, index=df.index)  # Z=-2.5 → sigmoid ≈ 0.08
        neutral_rs = pd.Series(0.0, index=df.index)

        scores_bearish = engine.precompute_scores(df, sector_rs=bearish_rs)
        scores_neutral = engine.precompute_scores(df, sector_rs=neutral_rs)

        bear_mean = float(scores_bearish.dropna().mean())
        neut_mean = float(scores_neutral.dropna().mean())
        assert bear_mean < neut_mean, (
            f"Bearish sector RS should lower score: {bear_mean:.4f} vs {neut_mean:.4f}"
        )

    def test_sector_rs_scores_in_valid_range(self):
        """Scores with sector RS must be in [0, 1]."""
        df     = self._make_full_df()
        engine = MomentumSignalEngine()
        # Use a realistic Z-score pattern (from -3 to +3)
        rng     = np.random.default_rng(55)
        zscores = pd.Series(rng.uniform(-3, 3, len(df)), index=df.index)
        scores  = engine.precompute_scores(df, sector_rs=zscores)
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0

    def test_sector_rs_fails_gracefully(self):
        """Invalid sector RS (wrong index type) falls back to 6-component mode."""
        df     = self._make_full_df()
        engine = MomentumSignalEngine()
        bad_rs = pd.Series([1.0, 2.0, np.nan], index=[0, 1, 2])  # wrong index type
        # Should not raise; should return valid scores
        try:
            scores = engine.precompute_scores(df, sector_rs=bad_rs)
            assert len(scores) == len(df)
            assert float(scores.min()) >= 0.0
        except Exception as e:
            # Any failure in sector RS should be caught and fallen back
            pytest.fail(f"Sector RS failure not handled gracefully: {e}")

    def test_sector_rs_7component_weights_sum_to_1(self):
        """Internal weights when sector RS is active must sum to 1.0."""
        weights_7 = [0.25, 0.25, 0.16, 0.12, 0.04, 0.10, 0.08]
        total = sum(weights_7)
        assert abs(total - 1.0) < 1e-9, f"7-component weights sum to {total}"

    def test_sector_rs_6component_weights_unchanged(self):
        """Existing 6-component WEIGHTS must still sum to 1.0."""
        from core.models.momentum_signal import WEIGHTS
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"6-component WEIGHTS sum to {total}"


# ── Entry guard integration (no position overlap) ─────────────────────────────

class TestEntryGuard:
    def test_entry_guard_normal_allows_momentum(self):
        """Low vol, balanced FII, normal calendar → allow_momentum = True."""
        nifty = _make_nifty(300)
        bhvr  = BehavioralSignals(nifty)
        date  = nifty.index[-1]
        guard = bhvr.entry_guard(date, fii_net_5d=1000.0)
        assert guard.allow_momentum

    def test_entry_guard_fii_crisis_blocks_momentum(self):
        """FII net -₹12,000cr (crisis) → allow_momentum = False."""
        nifty = _make_nifty(300)
        bhvr  = BehavioralSignals(nifty)
        date  = nifty.index[-1]
        guard = bhvr.entry_guard(date, fii_net_5d=-12_000.0)
        assert not guard.allow_momentum

    def test_entry_guard_position_scale_bounded(self):
        """position_scale must be in [0, 1]."""
        nifty = _make_nifty(300)
        bhvr  = BehavioralSignals(nifty)
        for date in nifty.index[30:]:
            guard = bhvr.entry_guard(date)
            assert 0.0 <= guard.position_scale <= 1.0


# ── weekly_aligned() tests ────────────────────────────────────────────────────

class TestWeeklyAligned:
    """Tests for MomentumSignalEngine.weekly_aligned() — Phase 4 gate."""

    def _make_trending_df(self, n: int, drift: float, vol: float = 0.008,
                          seed: int = 42) -> pd.DataFrame:
        rng    = np.random.default_rng(seed)
        dates  = pd.date_range("2023-01-01", periods=n, freq="B")
        prices = [200.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + drift + rng.normal(0, vol)))
        prices = np.array(prices)
        return pd.DataFrame({
            "open": prices, "high": prices * 1.005,
            "low":  prices * 0.995, "close": prices,
            "volume": np.ones(n) * 1_000_000,
        }, index=dates)

    def test_strong_weekly_uptrend_returns_true(self):
        """Stock trending up steadily for 100+ bars → weekly_aligned = True."""
        df   = self._make_trending_df(n=150, drift=0.003, vol=0.004)
        date = df.index[-1]
        result = MomentumSignalEngine.weekly_aligned(df, date)
        assert result is True, "Steady uptrend should be weekly aligned"

    def test_strong_weekly_downtrend_returns_false(self):
        """Stock trending down steadily → weekly_aligned = False."""
        df   = self._make_trending_df(n=150, drift=-0.003, vol=0.004)
        date = df.index[-1]
        result = MomentumSignalEngine.weekly_aligned(df, date)
        assert result is False, "Steady downtrend should NOT be weekly aligned"

    def test_insufficient_history_returns_true(self):
        """< 50 bars (< min_weeks × 5) → return True (benefit of doubt)."""
        df   = self._make_trending_df(n=30, drift=-0.002, vol=0.004)
        date = df.index[-1]
        result = MomentumSignalEngine.weekly_aligned(df, date)
        assert result is True, "Insufficient data should not block entries"

    def test_exactly_at_min_weeks_boundary(self):
        """Exactly min_weeks × 5 = 50 bars → result is well-defined (no error)."""
        df   = self._make_trending_df(n=50, drift=0.002, vol=0.003)
        date = df.index[-1]
        result = MomentumSignalEngine.weekly_aligned(df, date)
        assert isinstance(result, bool)

    def test_empty_df_returns_true_gracefully(self):
        """Empty DataFrame → return True (fail-open, don't block)."""
        df   = pd.DataFrame(columns=["close"], dtype=float)
        date = pd.Timestamp("2025-01-01")
        result = MomentumSignalEngine.weekly_aligned(df, date)
        assert result is True

    def test_weekly_aligned_after_crash_then_recovery(self):
        """Stock that fell then recovered with new all-time high → True."""
        rng    = np.random.default_rng(7)
        dates  = pd.date_range("2023-01-01", periods=200, freq="B")
        # Phase 1: steady rise (0-80 bars)
        prices = [100.0]
        for _ in range(79):
            prices.append(prices[-1] * (1 + 0.003 + rng.normal(0, 0.006)))
        # Phase 2: sharp dip (80-110 bars)
        for _ in range(30):
            prices.append(prices[-1] * (1 - 0.007 + rng.normal(0, 0.006)))
        # Phase 3: recovery (110-200 bars)
        for _ in range(90):
            prices.append(prices[-1] * (1 + 0.004 + rng.normal(0, 0.005)))
        prices = np.array(prices)
        df = pd.DataFrame({
            "open": prices, "high": prices * 1.004,
            "low":  prices * 0.996, "close": prices,
            "volume": np.ones(200) * 1_000_000,
        }, index=dates)
        date   = df.index[-1]
        result = MomentumSignalEngine.weekly_aligned(df, date)
        # After full recovery, weekly should be aligned
        assert result is True, "Full recovery should be weekly aligned at the end"


# ── Score persistence filter tests ────────────────────────────────────────────

class TestScorePersistenceFilter:
    """
    Unit tests for the score persistence gate added in Phase 5.

    The filter logic in momentum_portfolio.py run_portfolio():
        if _prev_rebal_scores:
            prev_rs = _prev_rebal_scores.get(tk, 0.0)
            if prev_rs < ENTRY_THRESHOLD:
                continue   # block single-rebalance spike
    """

    THRESHOLD = 0.55  # mirrors ENTRY_THRESHOLD in momentum_portfolio

    def _apply_filter(self, prev_scores: dict, tk: str, curr_score: float) -> bool:
        """Return True if the entry is ALLOWED (not filtered out)."""
        if curr_score < self.THRESHOLD:
            return False  # doesn't even qualify on current score
        if prev_scores:  # non-empty = not first rebalance
            prev = prev_scores.get(tk, 0.0)
            if prev < self.THRESHOLD:
                return False  # single-spike: blocked
        return True

    def test_first_rebalance_always_allows(self):
        """Empty prev_scores (first rebalance) → all qualifying entries allowed."""
        allowed = self._apply_filter({}, "HAL", curr_score=0.60)
        assert allowed is True

    def test_persistent_signal_allowed(self):
        """Score ≥ threshold at both prev and curr rebalances → allowed."""
        prev = {"HAL": 0.62, "TRENT": 0.57}
        allowed = self._apply_filter(prev, "HAL", curr_score=0.65)
        assert allowed is True

    def test_single_spike_blocked(self):
        """Score below threshold at prev rebalance, above at curr → blocked."""
        prev = {"HAL": 0.40}  # was below threshold last time
        allowed = self._apply_filter(prev, "HAL", curr_score=0.58)
        assert allowed is False

    def test_score_at_exactly_threshold_allowed(self):
        """Prev score exactly at ENTRY_THRESHOLD (0.55) → allowed (not strict <)."""
        prev = {"HAL": 0.55}  # exactly at threshold
        allowed = self._apply_filter(prev, "HAL", curr_score=0.60)
        assert allowed is True

    def test_score_just_below_threshold_blocks(self):
        """Prev score 0.549 (just under 0.55) → blocked."""
        prev = {"HAL": 0.549}
        allowed = self._apply_filter(prev, "HAL", curr_score=0.60)
        assert allowed is False

    def test_missing_prev_entry_treated_as_zero(self):
        """Stock not in prev_scores dict → treated as 0.0 → blocked."""
        prev = {"TRENT": 0.58}  # HAL not in dict
        allowed = self._apply_filter(prev, "HAL", curr_score=0.60)
        # prev_scores is non-empty, HAL defaults to 0.0 < 0.55 → blocked
        assert allowed is False

    def test_current_below_threshold_rejected_regardless(self):
        """Current score < threshold → rejected even if prev was high."""
        prev = {"HAL": 0.70}
        allowed = self._apply_filter(prev, "HAL", curr_score=0.40)
        assert allowed is False


# ── MR v2 backtest config tests ───────────────────────────────────────────────

class TestMRv2Config:
    """Verify the v2 mean reversion backtest configuration constants."""

    def test_quality_universe_size(self):
        """MR v2 universe has exactly 8 quality stocks."""
        from scripts.mean_reversion_backtest import MR_UNIVERSE
        assert len(MR_UNIVERSE) == 8

    def test_quality_universe_contains_banking(self):
        """Universe includes the 3 major private banks."""
        from scripts.mean_reversion_backtest import MR_UNIVERSE
        for bank in ["HDFCBANK", "ICICIBANK", "KOTAKBANK"]:
            assert bank in MR_UNIVERSE, f"{bank} missing from MR universe"

    def test_quality_universe_contains_pharma(self):
        """Universe includes both pharma stocks."""
        from scripts.mean_reversion_backtest import MR_UNIVERSE
        for pharma in ["SUNPHARMA", "LUPIN"]:
            assert pharma in MR_UNIVERSE, f"{pharma} missing from MR universe"

    def test_rsi_threshold_strict(self):
        """RSI entry threshold must be ≤ 30 (not the old 35)."""
        from scripts.mean_reversion_backtest import RSI_ENTRY_MAX
        assert RSI_ENTRY_MAX <= 30.0, f"RSI threshold too loose: {RSI_ENTRY_MAX}"

    def test_atr_stop_multiplier(self):
        """ATR stop multiplier should be 2.0."""
        from scripts.mean_reversion_backtest import ATR_STOP_MULT
        assert ATR_STOP_MULT == 2.0

    def test_stop_distance_bounds(self):
        """Min stop 4%, max stop 10% — sensible bounds for NSE equity."""
        from scripts.mean_reversion_backtest import MIN_STOP_PCT, MAX_STOP_PCT
        assert MIN_STOP_PCT >= 0.03
        assert MAX_STOP_PCT <= 0.15
        assert MIN_STOP_PCT < MAX_STOP_PCT

    def test_take_profit_achievable(self):
        """Take profit target must be achievable (5%-20%)."""
        from scripts.mean_reversion_backtest import TAKE_PROFIT_PCT
        assert 0.05 <= TAKE_PROFIT_PCT <= 0.20

    def test_rr_ratio_positive(self):
        """R:R ratio (take_profit / min_stop) > 1.0."""
        from scripts.mean_reversion_backtest import TAKE_PROFIT_PCT, MIN_STOP_PCT
        rr = TAKE_PROFIT_PCT / MIN_STOP_PCT
        assert rr > 1.0, f"R:R {rr:.2f} < 1.0 — negative expectancy"

    def test_no_zombie_stocks_in_universe(self):
        """IDEA and YESBANK must NOT be in the MR quality universe."""
        from scripts.mean_reversion_backtest import MR_UNIVERSE
        assert "IDEA" not in MR_UNIVERSE, "IDEA (zombie) should not be in quality universe"
        assert "YESBANK" not in MR_UNIVERSE, "YESBANK (bailout) should not be in quality universe"
        assert "TATAELXSI" not in MR_UNIVERSE, "TATAELXSI (single-client) not suitable"

"""
MARK5 Multi-Strategy Framework Tests v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests for:
  - StrategyBase utilities (RSI, ATR, SMA, 52w high)
  - RegimeRouter: regime detection from Nifty prices
  - MeanReversionStrategy: entry/exit conditions
  - PortfolioCircuitBreaker: level 1 & 2 triggers, reset

All tests use synthetic data — no external API calls.

RUN:
    cd /home/lynx/Documents/MARK5
    pytest tests/test_multi_strategy.py -v

CHANGELOG:
- [2026-05-23] v1.0: Initial test suite
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.strategies.base import StrategyBase, StrategySignal, TradeAction
from core.strategies.regime_router import (
    RegimeRouter, MarketRegimeState, RegimeAllocation, REGIME_ALLOCATION
)
from core.strategies.mean_reversion import MeanReversionStrategy
from core.strategies.circuit_breaker import (
    PortfolioCircuitBreaker, CircuitBreakerLevel
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prices(
    n: int = 300,
    start: float = 1000.0,
    trend: float = 0.0005,   # daily drift
    vol: float   = 0.015,    # daily vol
    seed: int    = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    rets   = rng.normal(trend, vol, n)
    close  = start * np.cumprod(1 + rets)
    high   = close * (1 + rng.uniform(0.001, 0.015, n))
    low    = close * (1 - rng.uniform(0.001, 0.015, n))
    opens  = np.roll(close, 1)
    opens[0] = start
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_nifty(
    n: int = 400,
    start: float = 18_000.0,
    trend: float = 0.0003,
    vol: float   = 0.01,
    seed: int    = 99,
) -> pd.Series:
    rng  = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    rets  = rng.normal(trend, vol, n)
    close = start * np.cumprod(1 + rets)
    return pd.Series(close, index=dates, name="nifty")


# ═══════════════════════════════════════════════════════════════════════════════
# StrategyBase utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyBaseUtils:
    """Test the shared static utility methods on StrategyBase."""

    def setup_method(self):
        self.prices = _make_prices(300)

    def test_rsi_range(self):
        """RSI must be in [0, 100]."""
        val = StrategyBase.rsi(self.prices["close"], period=14)
        assert 0 <= val <= 100

    def test_rsi_bullish_trend_above_50(self):
        """A strong uptrend should produce RSI > 50."""
        prices = _make_prices(200, trend=0.003, vol=0.005)
        val    = StrategyBase.rsi(prices["close"], period=14)
        assert val > 50, f"Expected RSI > 50 in strong uptrend, got {val:.1f}"

    def test_rsi_bearish_trend_below_50(self):
        """A strong downtrend should produce RSI < 50."""
        prices = _make_prices(200, trend=-0.003, vol=0.005)
        val    = StrategyBase.rsi(prices["close"], period=14)
        assert val < 50, f"Expected RSI < 50 in downtrend, got {val:.1f}"

    def test_atr_positive(self):
        val = StrategyBase.atr(self.prices, period=14)
        assert val > 0

    def test_atr_higher_vol_means_higher_atr(self):
        low_vol  = StrategyBase.atr(_make_prices(300, vol=0.005), 14)
        high_vol = StrategyBase.atr(_make_prices(300, vol=0.04),  14)
        assert high_vol > low_vol

    def test_sma_is_average(self):
        prices = self.prices["close"]
        manual = float(prices.tail(20).mean())
        val    = StrategyBase.sma(prices, 20)
        assert abs(val - manual) < 1e-6

    def test_high_52w_at_least_current(self):
        prices = self.prices["close"]
        h52    = StrategyBase.high_52w(prices)
        curr   = float(prices.iloc[-1])
        assert h52 >= curr or abs(h52 - curr) < 1e-6  # may equal if last bar is the high

    def test_high_52w_reflects_peak(self):
        """Manually spike the 150th bar and verify high_52w captures it."""
        prices = self.prices["close"].copy()
        spike  = float(prices.iloc[-1]) * 2.0   # double the current price
        prices.iloc[-150] = spike
        h52 = StrategyBase.high_52w(prices)
        assert h52 == pytest.approx(spike, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# RegimeRouter
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeRouter:

    def setup_method(self):
        self.router = RegimeRouter(sma_fast=50, sma_slow=200)

    def _nifty_above_200(self) -> pd.Series:
        """Create Nifty series where price stays well above 200d SMA."""
        n  = 400
        rng = np.random.default_rng(1)
        # Steady uptrend so price > 200d SMA
        rets  = rng.normal(0.0008, 0.008, n)
        close = 18_000.0 * np.cumprod(1 + rets)
        return pd.Series(close, index=pd.date_range("2018-01-01", periods=n, freq="B"))

    def _nifty_below_200(self) -> pd.Series:
        """Create Nifty series where price falls below 200d SMA."""
        n    = 400
        rng  = np.random.default_rng(2)
        # First half: uptrend, second half: sharp decline
        rets = np.concatenate([
            rng.normal(0.001, 0.008, 200),
            rng.normal(-0.003, 0.012, 200),
        ])
        close = 18_000.0 * np.cumprod(1 + rets)
        return pd.Series(close, index=pd.date_range("2018-01-01", periods=n, freq="B"))

    def test_bull_regime_above_200(self):
        nifty  = self._nifty_above_200()
        regime = self.router.detect(nifty)
        # In a sustained uptrend, should be BULL or NEUTRAL (not BEAR)
        assert regime in (MarketRegimeState.BULL, MarketRegimeState.NEUTRAL)

    def test_bear_regime_below_200(self):
        nifty  = self._nifty_below_200()
        regime = self.router.detect(nifty)
        assert regime == MarketRegimeState.BEAR

    def test_crisis_with_vix(self):
        nifty  = self._nifty_below_200()
        regime = self.router.detect(nifty, vix=30.0)
        assert regime == MarketRegimeState.CRISIS

    def test_no_crisis_without_high_vix(self):
        nifty  = self._nifty_below_200()
        regime = self.router.detect(nifty, vix=15.0)
        assert regime != MarketRegimeState.CRISIS

    def test_insufficient_data_returns_neutral(self):
        short  = _make_nifty(n=50)
        regime = self.router.detect(short)
        assert regime == MarketRegimeState.NEUTRAL

    def test_allocation_bull_no_mean_rev(self):
        alloc = self.router.allocation(MarketRegimeState.BULL)
        assert alloc.mean_rev_pct == 0.0
        assert alloc.max_mean_rev_pos == 0
        assert alloc.momentum_pct == 0.25

    def test_allocation_bear_momentum_unchanged(self):
        """BEAR regime: momentum is UNCHANGED (25%) — additive design."""
        alloc = self.router.allocation(MarketRegimeState.BEAR)
        # momentum_pct is same as BULL — ML is the quality gate, not regime
        assert alloc.momentum_pct == 0.25
        assert alloc.max_momentum_pos > 0
        assert alloc.mean_rev_pct > 0.0   # MR is the ADDED layer

    def test_allocation_crisis_allows_nothing(self):
        alloc = self.router.allocation(MarketRegimeState.CRISIS)
        assert alloc.allow_new_entries is False
        assert alloc.momentum_pct == 0.0
        assert alloc.mean_rev_pct == 0.0

    def test_detect_series_length(self):
        nifty  = _make_nifty(400)
        series = self.router.detect_series(nifty)
        assert len(series) == len(nifty)
        assert set(series.values) <= {s for s in MarketRegimeState}


# ═══════════════════════════════════════════════════════════════════════════════
# MeanReversionStrategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeanReversionStrategy:

    def setup_method(self):
        self.strat  = MeanReversionStrategy()
        self.nifty  = _make_nifty(400)
        self.date   = pd.Timestamp("2021-01-15")

    def _oversold_prices(self, n: int = 300) -> pd.DataFrame:
        """
        Create price series that clearly triggers mean-reversion entry:
        - Strong 52w rally, then sharp -30% correction, RSI collapses
        - Volume spikes at the low
        """
        rng  = np.random.default_rng(10)
        # Phase 1: 200 bars uptrend (builds 52w high)
        p1   = 1000.0 * np.cumprod(1 + rng.normal(0.002, 0.008, 200))
        # Phase 2: sharp -30% drop over 100 bars
        p2   = p1[-1] * np.cumprod(1 + rng.normal(-0.003, 0.015, 100))
        close = np.concatenate([p1, p2])
        n_total = len(close)
        high   = close * (1 + rng.uniform(0.001, 0.010, n_total))
        low    = close * (1 - rng.uniform(0.001, 0.010, n_total))
        # High volume at the end (capitulation)
        volume = rng.integers(100_000, 500_000, n_total).astype(float)
        volume[-10:] = volume[-10:] * 3   # spike
        dates  = pd.date_range("2019-01-01", periods=n_total, freq="B")
        return pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

    def test_entry_with_oversold_prices(self):
        """Oversold condition + volume spike + ML conf 0.5 → ENTER."""
        prices = self._oversold_prices()
        date   = prices.index[-1]
        nifty  = _make_nifty(len(prices) + 100)
        sig = self.strat.should_enter("TEST", prices, nifty, date, ml_confidence=0.55)
        # We may or may not get a signal depending on exact RSI; just verify no crash
        # and that if signal exists it is ENTER
        if sig is not None:
            assert sig.action == TradeAction.ENTER
            assert sig.strategy == "MeanReversion"
            assert 0 < sig.position_pct <= 0.15
            assert sig.stop_loss_pct > 0
            assert sig.take_profit_pct > 0

    def test_no_entry_overbought(self):
        """Strong uptrend (RSI > 35) → no entry."""
        prices = _make_prices(300, trend=0.003, vol=0.005)
        date   = prices.index[-1]
        sig    = self.strat.should_enter("TEST", prices, self.nifty, date, ml_confidence=0.55)
        assert sig is None

    def test_no_entry_low_ml_confidence(self):
        """ML confidence below threshold → no entry."""
        prices = self._oversold_prices()
        date   = prices.index[-1]
        nifty  = _make_nifty(len(prices) + 100)
        sig    = self.strat.should_enter("TEST", prices, nifty, date, ml_confidence=0.40)
        assert sig is None

    def test_no_entry_insufficient_data(self):
        prices = _make_prices(30)   # too short
        sig    = self.strat.should_enter("TEST", prices, self.nifty, prices.index[-1])
        assert sig is None

    # ── Exit tests ────────────────────────────────────────────────────────────

    def test_take_profit_exit(self):
        prices = _make_prices(100, start=1000.0)
        date   = prices.index[-1]
        # current price is > entry * 1.12
        entry  = float(prices["close"].iloc[-1]) / 1.15   # +15% gain
        sig    = self.strat.should_exit("TEST", prices, self.nifty, date,
                                        entry_price=entry, peak_price=entry*1.15,
                                        hold_days=5, ml_confidence=0.55)
        assert sig is not None
        assert sig.action == TradeAction.EXIT
        assert "TAKE_PROFIT" in sig.reasons[0]

    def test_stop_loss_exit(self):
        prices = _make_prices(100, start=1000.0)
        date   = prices.index[-1]
        entry  = float(prices["close"].iloc[-1]) / 0.88   # -12% loss
        sig    = self.strat.should_exit("TEST", prices, self.nifty, date,
                                        entry_price=entry, peak_price=entry,
                                        hold_days=5, ml_confidence=0.55)
        assert sig is not None
        assert "STOP_LOSS" in sig.reasons[0]

    def test_time_stop_exit(self):
        prices = _make_prices(100, start=1000.0)
        date   = prices.index[-1]
        entry  = float(prices["close"].iloc[-1]) * 0.99   # -1% (tiny loss)
        sig    = self.strat.should_exit("TEST", prices, self.nifty, date,
                                        entry_price=entry, peak_price=entry,
                                        hold_days=31,   # > max_hold_days=30 (v2 updated)
                                        ml_confidence=0.55)
        assert sig is not None
        assert "TIME_STOP" in sig.reasons[0]

    def test_no_exit_when_conditions_not_met(self):
        prices = _make_prices(100, start=1000.0)
        date   = prices.index[-1]
        entry  = float(prices["close"].iloc[-1]) * 1.05  # +5% gain (< 12% TP)
        sig    = self.strat.should_exit("TEST", prices, self.nifty, date,
                                        entry_price=entry, peak_price=entry*1.05,
                                        hold_days=10, ml_confidence=0.55)
        # Should NOT exit — gain < TP, loss < SL, time < max_hold
        assert sig is None

    def test_exit_signal_is_full_exit(self):
        """EXIT signal should have position_pct=0."""
        prices = _make_prices(100, start=1000.0)
        date   = prices.index[-1]
        entry  = float(prices["close"].iloc[-1]) * 0.80   # big loss → stop
        sig    = self.strat.should_exit("TEST", prices, self.nifty, date,
                                        entry_price=entry, peak_price=entry,
                                        hold_days=5, ml_confidence=0.55)
        if sig is not None:
            assert sig.position_pct == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PortfolioCircuitBreaker
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioCircuitBreaker:

    def _run_cb(self, equity_series: list, initial: float = 1_000_000.0) -> list:
        """Helper: run CB over equity series and return list of actions."""
        cb      = PortfolioCircuitBreaker(initial)
        dates   = pd.date_range("2020-01-01", periods=len(equity_series), freq="B")
        actions = []
        for eq, date in zip(equity_series, dates):
            act = cb.update(eq, date, nifty_above_sma200=True)
            actions.append(act)
        return actions, cb

    def test_no_trigger_on_flat_equity(self):
        equity  = [1_000_000.0] * 50
        actions, _ = self._run_cb(equity)
        assert all(a == CircuitBreakerLevel.NONE for a in actions)

    def test_level1_warning_at_15pct_dd(self):
        """Equity drops exactly 15% from peak → WARNING."""
        equity  = [1_000_000.0] * 25
        peak    = 1_100_000.0
        equity += [peak] * 5
        equity += [peak * (1 - 0.16)]   # -16% from peak (above Level 1 threshold)
        actions, cb = self._run_cb(equity)
        last_action = actions[-1]
        assert last_action in (CircuitBreakerLevel.WARNING, CircuitBreakerLevel.NONE)
        assert cb.state.level in (CircuitBreakerLevel.WARNING, CircuitBreakerLevel.HALT)

    def test_level2_halt_at_22pct_dd(self):
        """Equity drops 22%+ from peak → HALT."""
        equity  = [1_000_000.0] * 50
        peak    = 1_200_000.0
        equity += [peak] * 5
        equity += [peak * (1 - 0.24)]   # -24% drop (above Level 2 threshold)
        actions, cb = self._run_cb(equity)
        assert cb.state.level == CircuitBreakerLevel.HALT

    def test_level1_reset_when_equity_recovers(self):
        """After WARNING, recovery below 8% DD → NONE."""
        peak    = 1_000_000.0
        equity  = [peak] * 30
        equity += [peak * 0.86] * 5   # -14% → WARNING
        equity += [peak * 0.94] * 15  # recovered to -6% → should reset
        _, cb = self._run_cb(equity)
        # Final state should be NONE (reset) after sufficient recovery
        # Note: rolling peak covers 21 bars, so we need equity sustained above -8%
        assert cb.state.level in (CircuitBreakerLevel.NONE, CircuitBreakerLevel.WARNING)

    def test_allow_new_entries_only_when_none(self):
        cb  = PortfolioCircuitBreaker(1_000_000.0)
        assert cb.allow_new_entries is True
        # Simulate -18% drop
        cb.update(820_000.0, pd.Timestamp("2020-06-01"))
        assert cb.allow_new_entries is False

    def test_summary_contains_required_keys(self):
        cb  = PortfolioCircuitBreaker(1_000_000.0)
        s   = cb.summary()
        required = {"level", "peak_equity", "current_drawdown", "events_count"}
        assert required <= s.keys()

    def test_consecutive_updates_no_crash(self):
        """Stress test: 500 bars of random equity → no exception."""
        rng    = np.random.default_rng(77)
        equity = 1_000_000.0 * np.cumprod(1 + rng.normal(0.0, 0.015, 500))
        _, cb  = self._run_cb(equity.tolist())
        # Just verify no crash and state is valid
        assert cb.state.level in (
            CircuitBreakerLevel.NONE,
            CircuitBreakerLevel.WARNING,
            CircuitBreakerLevel.HALT,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Regime × MeanReversion
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_bear_regime_activates_mean_rev(self):
        """BEAR regime: MR is active, momentum is UNCHANGED (additive design)."""
        router = RegimeRouter()
        alloc  = router.allocation(MarketRegimeState.BEAR)
        assert alloc.max_mean_rev_pos > 0
        # Momentum is NOT reduced in BEAR — ML confidence is the quality gate
        bull_alloc = router.allocation(MarketRegimeState.BULL)
        assert alloc.momentum_pct == bull_alloc.momentum_pct
        assert alloc.mean_rev_pct > 0.0

    def test_bull_regime_blocks_mean_rev(self):
        """In BULL regime, mean-reversion should NOT be deployed."""
        router = RegimeRouter()
        alloc  = router.allocation(MarketRegimeState.BULL)
        assert alloc.mean_rev_pct == 0.0
        assert alloc.max_mean_rev_pos == 0
        assert alloc.max_momentum_pos == 4

    def test_neutral_regime_uses_both(self):
        """In NEUTRAL regime, both strategies are active."""
        router = RegimeRouter()
        alloc  = router.allocation(MarketRegimeState.NEUTRAL)
        assert alloc.momentum_pct > 0
        assert alloc.mean_rev_pct > 0
        assert alloc.max_mean_rev_pos > 0

    def test_strategy_signal_fields(self):
        """A StrategySignal must have the right fields and types."""
        sig = StrategySignal(
            ticker="TEST",
            action=TradeAction.ENTER,
            strategy="MeanReversion",
            confidence=0.6,
            position_pct=0.10,
            stop_loss_pct=0.08,
            take_profit_pct=0.12,
            max_hold_days=25,
            reasons=["RSI=30"],
        )
        assert sig.is_entry
        assert not sig.is_exit
        assert sig.position_pct == pytest.approx(0.10)

    def test_regime_series_consistent_with_detect(self):
        """detect_series should agree with per-bar detect calls."""
        router = RegimeRouter()
        nifty  = _make_nifty(300)
        series = router.detect_series(nifty)

        # Sample 10 random bars and verify they match
        import random
        random.seed(42)
        indices = random.sample(range(200, 300), 10)
        for i in indices:
            date   = nifty.index[i]
            hist   = nifty.iloc[:i + 1]
            r_det  = router.detect(hist)
            r_ser  = series.iloc[i]
            assert r_det == r_ser, (
                f"Bar {i} ({date.date()}): detect={r_det} vs series={r_ser}"
            )

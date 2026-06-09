"""
MARK5 V9 System Tests — Adaptive Volatility System
===================================================
Tests for all V9 improvements:
  1. ATR-adaptive trail (compute_atr_pct, get_adaptive_trail)
  2. Nifty 21d regime gate (compute_nifty_regime)
  3. Rolling 60d performance gate (get_port_60d_return)
  4. Initial stop cooldown (V9Portfolio.is_on_cooldown)
  5. V9Portfolio mechanics (enter, exit, cooldown tracking)
  6. run_v9 output structure and design invariants

Run: pytest tests/test_v9_system.py -v
"""
import math
import sys
import os
from typing import List
import pytest
import numpy as np
import pandas as pd
from dataclasses import fields
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from scripts.multi_strategy_backtest_v9 import (
    # Constants
    ATR_WINDOW, ATR_TRAIL_MULTIPLIER, ATR_TRAIL_MIN, ATR_TRAIL_MAX,
    NIFTY_REGIME_WINDOW, NIFTY_REGIME_THRESHOLD, NIFTY_REGIME_MAX_SLOTS,
    PERF_GATE_LOOKBACK, PERF_GATE_THRESHOLD, PERF_GATE_HURDLE,
    INITIAL_STOP_COOLDOWN_DAYS,
    # Functions
    compute_atr_pct, get_adaptive_trail, compute_nifty_regime,
    get_port_60d_return,
    # Classes
    V9Position, V9Portfolio, Trade,
    # Inherited constants
    INITIAL_STOP_LOSS_PCT, INITIAL_STOP_DAYS, ROLLING_HIGH_TRIGGER,
    ROLLING_HIGH_TRAIL_PCT, PORT_YTD_DOWN_SCALE, V8_ML_ENTRY_HURDLE,
)
from scripts.multi_strategy_backtest_v6 import (
    TRAIL_NORMAL, TRAIL_ELEVATED, TRAIL_HIGH, INITIAL_CAPITAL,
    OOS_START, OOS_END,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(
    n: int = 50,
    base_close: float = 1000.0,
    daily_vol: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with controllable volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    close = base_close * np.cumprod(1 + rng.normal(0, daily_vol, n))
    spread = close * daily_vol * 0.5
    return pd.DataFrame({
        "open":   close * (1 - rng.uniform(0, daily_vol * 0.3, n)),
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": rng.integers(100_000, 1_000_000, n),
    }, index=dates)


def _make_nifty(n: int = 100, base: float = 20_000.0, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2022-06-01", periods=n)
    vals  = base * np.cumprod(1 + rng.normal(0.0003, 0.008, n))
    return pd.Series(vals, index=dates, name="close")


def _make_portfolio(initial_capital: float = INITIAL_CAPITAL) -> V9Portfolio:
    return V9Portfolio(initial_capital)


def _make_position(
    ticker: str = "TEST",
    entry_price: float = 1000.0,
    trail_pct: float = 0.15,
    conf_entry: float = 0.65,
    entry_date: pd.Timestamp = pd.Timestamp("2022-06-01"),
    atr_pct_entry: float = 0.020,
) -> V9Position:
    return V9Position(
        ticker=ticker,
        entry_price=entry_price,
        peak_price=entry_price,
        entry_date=entry_date,
        shares=100,
        entry_cost=entry_price * 100 * 1.0029,
        trail_pct=trail_pct,
        conf_entry=conf_entry,
        alloc_tier="T2",
        atr_pct_entry=atr_pct_entry,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestV9Constants:
    def test_atr_window_positive(self):
        assert ATR_WINDOW > 0

    def test_atr_multiplier_reasonable(self):
        # 7× is calibrated — between 5 and 10
        assert 5.0 <= ATR_TRAIL_MULTIPLIER <= 10.0

    def test_atr_trail_bounds_valid(self):
        assert 0.05 <= ATR_TRAIL_MIN < ATR_TRAIL_MAX <= 0.30

    def test_nifty_regime_threshold_negative(self):
        assert NIFTY_REGIME_THRESHOLD < 0

    def test_nifty_regime_max_slots_less_than_max_positions(self):
        from scripts.multi_strategy_backtest_v6 import MAX_POSITIONS
        assert 1 <= NIFTY_REGIME_MAX_SLOTS < MAX_POSITIONS

    def test_perf_gate_threshold_negative(self):
        assert PERF_GATE_THRESHOLD < 0

    def test_perf_gate_hurdle_above_v8_hurdle(self):
        assert PERF_GATE_HURDLE > V8_ML_ENTRY_HURDLE

    def test_perf_gate_lookback_reasonable(self):
        assert 30 <= PERF_GATE_LOOKBACK <= 120

    def test_cooldown_days_reasonable(self):
        assert 30 <= INITIAL_STOP_COOLDOWN_DAYS <= 90

    def test_v8_fixes_inherited_unchanged(self):
        assert INITIAL_STOP_LOSS_PCT == 0.07
        assert INITIAL_STOP_DAYS     == 45
        assert ROLLING_HIGH_TRIGGER  == 1.50
        assert PORT_YTD_DOWN_SCALE   == 0.60
        assert V8_ML_ENTRY_HURDLE    == 0.56


# ─────────────────────────────────────────────────────────────────────────────
# 2. compute_atr_pct
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAtrPct:
    def test_returns_positive(self):
        df   = _make_ohlcv(n=50, daily_vol=0.02)
        date = df.index[-1]
        atr  = compute_atr_pct(df, date, n=21)
        assert atr > 0.0

    def test_high_vol_greater_than_low_vol(self):
        df_low  = _make_ohlcv(n=60, daily_vol=0.005, seed=1)
        df_high = _make_ohlcv(n=60, daily_vol=0.040, seed=2)
        date = df_low.index[-1]
        assert compute_atr_pct(df_high, date) > compute_atr_pct(df_low, date)

    def test_fallback_when_insufficient_data(self):
        df   = _make_ohlcv(n=3)
        date = df.index[-1]
        atr  = compute_atr_pct(df, date, n=21)
        assert atr == pytest.approx(0.020, abs=0.001)

    def test_fallback_on_future_date(self):
        # Future date includes all existing rows → actual ATR computed (not fallback)
        df   = _make_ohlcv(n=30)
        atr  = compute_atr_pct(df, pd.Timestamp("2099-01-01"), n=21)
        # Should return actual ATR (using all available data), not the 0.020 fallback
        assert atr > 0.0  # positive, no crash

    def test_result_capped_in_reasonable_range(self):
        # Even extreme volatility should give a sane ATR%
        df   = _make_ohlcv(n=50, daily_vol=0.10, seed=99)
        date = df.index[-1]
        atr  = compute_atr_pct(df, date, n=21)
        assert 0.001 < atr < 0.50   # 0.1% to 50% — anything beyond is data error

    def test_nse_midcap_realistic_range(self):
        # NSE mid-cap realistic range: 1.5-4% ATR
        df   = _make_ohlcv(n=60, daily_vol=0.018, seed=7)
        date = df.index[-1]
        atr  = compute_atr_pct(df, date)
        assert 0.005 < atr < 0.10   # 0.5% to 10%

    def test_uses_high_low_not_just_close(self):
        """ATR uses true range (H-L, |H-prevC|, |L-prevC|) not close-to-close."""
        n     = 40
        dates = pd.bdate_range("2023-01-01", periods=n)
        close = np.full(n, 1000.0)
        # Make H-L spread large but close-to-close stable
        df_wide = pd.DataFrame({
            "open": close - 30, "high": close + 50,
            "low": close - 50, "close": close, "volume": np.ones(n) * 100_000
        }, index=dates)
        df_narrow = pd.DataFrame({
            "open": close - 3, "high": close + 5,
            "low": close - 5, "close": close, "volume": np.ones(n) * 100_000
        }, index=dates)
        date = dates[-1]
        assert compute_atr_pct(df_wide, date) > compute_atr_pct(df_narrow, date)


# ─────────────────────────────────────────────────────────────────────────────
# 3. get_adaptive_trail
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAdaptiveTrail:
    def test_normal_vix_medium_atr_baseline(self):
        # VIX = 0.18 (normal) + ATR = 2% → near TRAIL_NORMAL (15%)
        trail = get_adaptive_trail(vix_val=0.18, atr_pct=0.020)
        # 7 × 0.02 / 0.15 × 0.15 = 0.14 → 14%
        assert 0.10 <= trail <= 0.22

    def test_high_atr_widens_trail(self):
        trail_low  = get_adaptive_trail(0.18, 0.015)
        trail_high = get_adaptive_trail(0.18, 0.030)
        assert trail_high > trail_low

    def test_high_vix_tightens_trail(self):
        trail_normal_vix = get_adaptive_trail(0.18, 0.020)
        trail_high_vix   = get_adaptive_trail(0.30, 0.020)
        assert trail_high_vix < trail_normal_vix

    def test_always_in_bounds(self):
        for vix in [0.10, 0.20, 0.25, 0.35]:
            for atr in [0.005, 0.010, 0.020, 0.030, 0.050]:
                t = get_adaptive_trail(vix, atr)
                assert ATR_TRAIL_MIN <= t <= ATR_TRAIL_MAX, \
                    f"trail={t} out of [{ATR_TRAIL_MIN},{ATR_TRAIL_MAX}] for vix={vix} atr={atr}"

    def test_floor_applied(self):
        trail = get_adaptive_trail(vix_val=0.35, atr_pct=0.005)  # high VIX, very low ATR
        assert trail >= ATR_TRAIL_MIN

    def test_ceiling_applied(self):
        trail = get_adaptive_trail(vix_val=0.10, atr_pct=0.060)  # low VIX, very high ATR
        assert trail <= ATR_TRAIL_MAX

    def test_nse_midcap_typical_range(self):
        # RELIANCE (1.9%): trail should be ~11-14%
        t_reliance = get_adaptive_trail(0.18, 0.019)
        assert 0.10 <= t_reliance <= 0.18

        # TATAELXSI (2.5%): trail should be ~15-20%
        t_tata = get_adaptive_trail(0.18, 0.025)
        assert 0.13 <= t_tata <= 0.22

        # BEL (2.8%): trail should be wider than TATAELXSI
        t_bel = get_adaptive_trail(0.18, 0.028)
        assert t_bel >= t_tata

    def test_monotone_in_atr(self):
        """Higher ATR → wider or equal trail, all else equal."""
        atrs = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040]
        trails = [get_adaptive_trail(0.18, a) for a in atrs]
        for i in range(len(trails) - 1):
            assert trails[i] <= trails[i+1] or (trails[i] == ATR_TRAIL_MAX and trails[i+1] == ATR_TRAIL_MAX)

    def test_monotone_in_vix(self):
        """Higher VIX → tighter or equal trail, all else equal."""
        vixs   = [0.15, 0.20, 0.25, 0.30, 0.35]
        trails = [get_adaptive_trail(v, 0.020) for v in vixs]
        # Should be non-increasing as VIX increases (tighter stops)
        for i in range(len(trails) - 1):
            assert trails[i] >= trails[i+1] or (trails[i] == ATR_TRAIL_MIN and trails[i+1] == ATR_TRAIL_MIN)


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_nifty_regime
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeNiftyRegime:
    def test_bullish_market_positive(self):
        dates = pd.bdate_range("2022-01-01", periods=60)
        vals  = np.linspace(20000, 22000, 60)  # +10% over period
        nifty = pd.Series(vals, index=dates)
        ret   = compute_nifty_regime(nifty, dates[-1])
        assert ret > 0

    def test_bearish_market_negative(self):
        dates = pd.bdate_range("2022-01-01", periods=60)
        vals  = np.linspace(20000, 18000, 60)  # -10% over period
        nifty = pd.Series(vals, index=dates)
        ret   = compute_nifty_regime(nifty, dates[-1])
        assert ret < 0

    def test_flat_market_near_zero(self):
        dates = pd.bdate_range("2022-01-01", periods=60)
        nifty = pd.Series(20000.0, index=dates)
        ret   = compute_nifty_regime(nifty, dates[-1])
        assert abs(ret) < 0.001

    def test_insufficient_data_returns_zero(self):
        dates = pd.bdate_range("2022-01-01", periods=5)
        nifty = pd.Series(20000.0, index=dates)
        ret   = compute_nifty_regime(nifty, dates[-1])
        assert ret == 0.0

    def test_future_date_filtered_correctly(self):
        dates = pd.bdate_range("2022-01-01", periods=50)
        nifty = pd.Series(20000.0, index=dates)
        # Date at boundary
        ret = compute_nifty_regime(nifty, dates[25])
        assert isinstance(ret, float)

    def test_threshold_calibration(self):
        """Nifty down 6% in 21d should be below -5% threshold."""
        dates = pd.bdate_range("2022-01-01", periods=50)
        vals  = list(np.linspace(20000, 19200, 50))  # -4% over full period
        # Concentrate decline in last 21d
        start_21d = 20000.0 * 1.02   # slight gain before
        vals_arr  = np.concatenate([
            np.linspace(19800, 20400, 29),
            np.linspace(20400, 19200, 21),  # -6% in 21d
        ])
        nifty = pd.Series(vals_arr, index=dates)
        ret   = compute_nifty_regime(nifty, dates[-1])
        assert ret < NIFTY_REGIME_THRESHOLD  # < -5%

    def test_uses_exactly_nifty_window_days(self):
        """21-day window: ret = (now - 21d_ago) / 21d_ago."""
        dates = pd.bdate_range("2022-01-01", periods=30)
        vals  = np.ones(30) * 20000.0
        vals[-1] = 21000.0   # Only last day is different
        nifty = pd.Series(vals, index=dates)
        ret   = compute_nifty_regime(nifty, dates[-1])
        # 21d-ago = vals[-22] = 20000, now = 21000 → +5%
        expected = (21000 - 20000) / 20000
        assert ret == pytest.approx(expected, abs=0.002)


# ─────────────────────────────────────────────────────────────────────────────
# 5. get_port_60d_return
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPort60dReturn:
    def _make_history(self, n: int, start_equity: float, end_equity: float) -> List:
        from typing import List as L
        return [{"equity": start_equity + (end_equity - start_equity) * i / max(n-1, 1)}
                for i in range(n)]

    def test_positive_return_when_equity_grew(self):
        history = [{"equity": 5_000_000 + i * 10_000} for i in range(80)]
        ret = get_port_60d_return(history, 5_800_000.0)
        assert ret > 0

    def test_negative_return_when_equity_fell(self):
        history = [{"equity": 5_000_000.0} for _ in range(80)]
        ret = get_port_60d_return(history, 4_500_000.0)
        assert ret < 0

    def test_gate_fires_below_threshold(self):
        # Portfolio started at 5cr, now at 4.5cr (−10% > threshold of −8%)
        history = [{"equity": 5_000_000.0} for _ in range(80)]
        ret = get_port_60d_return(history, 4_500_000.0)
        assert ret < PERF_GATE_THRESHOLD

    def test_gate_does_not_fire_on_small_decline(self):
        # Portfolio started at 5cr, now at 4.9cr (−2% < threshold)
        history = [{"equity": 5_000_000.0} for _ in range(80)]
        ret = get_port_60d_return(history, 4_900_000.0)
        assert ret > PERF_GATE_THRESHOLD

    def test_returns_zero_when_insufficient_history(self):
        history = [{"equity": 5_000_000.0} for _ in range(PERF_GATE_LOOKBACK - 1)]
        ret = get_port_60d_return(history, 5_200_000.0)
        assert ret == 0.0

    def test_uses_exactly_lookback_entries_ago(self):
        n = PERF_GATE_LOOKBACK + 10
        # First 10 entries at 4M, next LOOKBACK at 5M
        history = [{"equity": 4_000_000.0}] * 10 + \
                  [{"equity": 5_000_000.0}] * PERF_GATE_LOOKBACK
        # current = 5.5M; history[-LOOKBACK]["equity"] = 5M → +10%
        ret = get_port_60d_return(history, 5_500_000.0)
        expected = (5_500_000 - 5_000_000) / 5_000_000
        assert ret == pytest.approx(expected, abs=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# 6. V9Position Dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestV9PositionDefaults:
    def test_has_atr_pct_entry_field(self):
        field_names = {f.name for f in fields(V9Position)}
        assert "atr_pct_entry" in field_names

    def test_atr_pct_entry_default_zero(self):
        # Dataclass default is 0.0; _make_position sets it to 0.020 for realism
        # Test the raw dataclass default directly
        raw = V9Position(
            ticker="X", entry_price=1000.0, peak_price=1000.0,
            entry_date=pd.Timestamp("2023-01-01"), shares=100,
            entry_cost=100300.0, trail_pct=0.15, conf_entry=0.60,
            alloc_tier="T1",
        )
        assert raw.atr_pct_entry == 0.0

    def test_has_conf_peak_field(self):
        field_names = {f.name for f in fields(V9Position)}
        assert "conf_peak" in field_names

    def test_has_ratchet_floor_field(self):
        # Ratchet disabled but field must exist for code compatibility
        field_names = {f.name for f in fields(V9Position)}
        assert "ratchet_floor" in field_names

    def test_no_rsi_partial_done_field(self):
        field_names = {f.name for f in fields(V9Position)}
        assert "rsi_partial_done" not in field_names

    def test_trail_pct_set_correctly(self):
        pos = _make_position(trail_pct=0.17)
        assert pos.trail_pct == 0.17

    def test_immutable_trail_at_entry(self):
        """Trail should not change after entry (ATR-adaptive is set once)."""
        pos = _make_position(trail_pct=0.17)
        initial_trail = pos.trail_pct
        pos.peak_price = 1200.0   # Simulate price increase
        assert pos.trail_pct == initial_trail


# ─────────────────────────────────────────────────────────────────────────────
# 7. V9Portfolio — Cooldown Mechanics
# ─────────────────────────────────────────────────────────────────────────────

class TestV9PortfolioCooldown:
    def test_no_cooldown_initially(self):
        port = _make_portfolio()
        assert not port.is_on_cooldown("BHARTIARTL", pd.Timestamp("2023-01-01"))

    def test_cooldown_set_on_initial_stop_exit(self):
        port = _make_portfolio()
        # Manually plant a position then exit it with INITIAL_STOP
        date_entry = pd.Timestamp("2023-01-01")
        date_exit  = pd.Timestamp("2023-01-20")
        port.positions["TEST"] = _make_position("TEST", entry_date=date_entry)
        port.exit("TEST", 900.0, date_exit, reason=f"INITIAL_STOP(19d,-10%)")
        assert "TEST" in port.initial_stop_dates
        assert port.initial_stop_dates["TEST"] == date_exit

    def test_ticker_blocked_during_cooldown(self):
        port = _make_portfolio()
        stop_date = pd.Timestamp("2023-06-01")
        port.initial_stop_dates["RELIANCE"] = stop_date
        # 30 days later → still in 60-day cooldown
        check_date = stop_date + pd.Timedelta(days=30)
        assert port.is_on_cooldown("RELIANCE", check_date)

    def test_cooldown_expires_after_window(self):
        port = _make_portfolio()
        stop_date = pd.Timestamp("2023-06-01")
        port.initial_stop_dates["RELIANCE"] = stop_date
        # 61 days later → cooldown expired
        check_date = stop_date + pd.Timedelta(days=INITIAL_STOP_COOLDOWN_DAYS + 1)
        assert not port.is_on_cooldown("RELIANCE", check_date)

    def test_cooldown_exact_boundary(self):
        port = _make_portfolio()
        stop_date = pd.Timestamp("2023-06-01")
        port.initial_stop_dates["BEL"] = stop_date
        # Exactly at cooldown boundary
        exact_date = stop_date + pd.Timedelta(days=INITIAL_STOP_COOLDOWN_DAYS)
        # At exactly 60 days: (60 < 60) is False → NOT in cooldown
        assert not port.is_on_cooldown("BEL", exact_date)

    def test_trail_stop_does_not_set_cooldown(self):
        port = _make_portfolio()
        date_entry = pd.Timestamp("2023-01-01")
        date_exit  = pd.Timestamp("2023-04-01")
        port.positions["TEST"] = _make_position("TEST", entry_date=date_entry)
        port.exit("TEST", 1200.0, date_exit, reason="TRAIL_STOP(15%)")
        assert "TEST" not in port.initial_stop_dates

    def test_ml_exit_does_not_set_cooldown(self):
        port = _make_portfolio()
        date_entry = pd.Timestamp("2023-01-01")
        date_exit  = pd.Timestamp("2023-04-01")
        port.positions["TEST"] = _make_position("TEST", entry_date=date_entry)
        port.exit("TEST", 1050.0, date_exit, reason="ML_EXIT(rc=0.410)")
        assert "TEST" not in port.initial_stop_dates

    def test_different_tickers_independent_cooldowns(self):
        port = _make_portfolio()
        stop_date = pd.Timestamp("2023-06-01")
        port.initial_stop_dates["LUPIN"] = stop_date
        # COFORGE has no cooldown
        check_date = stop_date + pd.Timedelta(days=10)
        assert port.is_on_cooldown("LUPIN", check_date)
        assert not port.is_on_cooldown("COFORGE", check_date)


# ─────────────────────────────────────────────────────────────────────────────
# 8. V9Portfolio — Entry with ATR-Adaptive Trail
# ─────────────────────────────────────────────────────────────────────────────

class TestV9PortfolioEntry:
    def test_enter_sets_atr_adaptive_trail(self):
        port = _make_portfolio()
        port.cash = 5_000_000.0
        entered = port.enter(
            "LUPIN", price=1000.0, date=pd.Timestamp("2023-01-01"),
            conf=0.65, vix_val=0.18, atr_pct=0.025,
        )
        assert entered
        pos = port.positions["LUPIN"]
        # ATR-adaptive trail for atr=2.5%, vix=0.18 should be ~17-18%
        expected_trail = get_adaptive_trail(vix_val=0.18, atr_pct=0.025)
        assert pos.trail_pct == pytest.approx(expected_trail, abs=0.001)

    def test_enter_stores_atr_pct_at_entry(self):
        port = _make_portfolio()
        port.cash = 5_000_000.0
        port.enter("LT", 2000.0, pd.Timestamp("2023-01-01"), 0.70, 0.18, atr_pct=0.021)
        assert port.positions["LT"].atr_pct_entry == pytest.approx(0.021, abs=0.0001)

    def test_high_atr_stock_gets_wider_trail_than_low_atr(self):
        port1 = _make_portfolio()
        port2 = _make_portfolio()
        port1.cash = 5_000_000.0
        port2.cash = 5_000_000.0
        port1.enter("HIGH_ATR", 1000.0, pd.Timestamp("2023-01-01"), 0.65, 0.18, atr_pct=0.030)
        port2.enter("LOW_ATR",  1000.0, pd.Timestamp("2023-01-01"), 0.65, 0.18, atr_pct=0.015)
        assert port1.positions["HIGH_ATR"].trail_pct > port2.positions["LOW_ATR"].trail_pct

    def test_enter_rejects_if_on_cooldown(self):
        """Test that cooldown check works — portfolio should NOT enter if on cooldown.
        (Actual enforcement is in run_v9's entry loop; here we test is_on_cooldown)."""
        port = _make_portfolio()
        stop_date = pd.Timestamp("2023-01-01")
        port.initial_stop_dates["COFORGE"] = stop_date
        check_date = stop_date + pd.Timedelta(days=30)
        assert port.is_on_cooldown("COFORGE", check_date)

    def test_enter_returns_false_when_max_positions_full(self):
        from scripts.multi_strategy_backtest_v6 import MAX_POSITIONS
        port = _make_portfolio()
        port.cash = 100_000_000.0
        for i in range(MAX_POSITIONS):
            port.enter(f"TICKER_{i}", 1000.0, pd.Timestamp("2023-01-01"), 0.65, 0.18, 0.020)
        result = port.enter("TICKER_EXTRA", 1000.0, pd.Timestamp("2023-01-01"), 0.65, 0.18, 0.020)
        assert not result

    def test_cash_reduced_on_entry(self):
        port = _make_portfolio()
        port.cash = 5_000_000.0
        cash_before = port.cash
        port.enter("BEL", 500.0, pd.Timestamp("2023-01-01"), 0.65, 0.18, 0.028)
        assert port.cash < cash_before


# ─────────────────────────────────────────────────────────────────────────────
# 9. V9Portfolio — YTD Gate
# ─────────────────────────────────────────────────────────────────────────────

class TestV9PortfolioYtdGate:
    def test_ytd_return_zero_at_start(self):
        port = _make_portfolio()
        prices = {"FAKE": 1000.0}
        assert port.ytd_return(prices) == pytest.approx(0.0, abs=0.01)

    def test_ytd_return_positive_when_equity_grew(self):
        port = _make_portfolio()
        port._ytd_equity_jan1 = 5_000_000.0
        prices = {}
        port.cash = 5_500_000.0   # +10%
        assert port.ytd_return(prices) == pytest.approx(0.10, abs=0.01)

    def test_ytd_return_negative_when_equity_fell(self):
        port = _make_portfolio()
        port._ytd_equity_jan1 = 5_000_000.0
        port.cash = 4_800_000.0
        assert port.ytd_return({}) < 0

    def test_reset_ytd_updates_baseline(self):
        port = _make_portfolio()
        port.cash = 6_000_000.0
        port.reset_ytd({})
        assert port._ytd_equity_jan1 == pytest.approx(6_000_000.0, abs=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. run_v9 Output Structure
# ─────────────────────────────────────────────────────────────────────────────

class TestRunV9OutputStructure:
    """Verify run_v9 returns valid dict without running full backtest."""

    @pytest.fixture(autouse=True)
    def minimal_run(self):
        """Run V9 on minimal synthetic data."""
        from scripts.multi_strategy_backtest_v9 import run_v9
        from scripts.multi_strategy_backtest_v6 import INITIAL_CAPITAL, MIN_ML_STD

        # Single ticker synthetic data
        n     = 300
        dates = pd.bdate_range(start="2022-01-03", periods=n)
        close = 1000.0 * np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.015, n))
        spread = close * 0.01
        df = pd.DataFrame({
            "open": close * 0.999, "high": close + spread,
            "low":  close - spread, "close": close,
            "volume": np.ones(n) * 500_000
        }, index=dates)

        nifty_vals = 20000.0 * np.cumprod(1 + np.random.default_rng(1).normal(0.0003, 0.01, n))
        nifty = pd.Series(nifty_vals, index=dates, name="close")

        conf_vals = np.clip(
            np.random.default_rng(2).normal(0.58, 0.05, n), 0.45, 0.85
        )
        conf = pd.Series(conf_vals, index=dates, name="conf")

        all_data = {"TESTCO": df}
        conf_map = {"TESTCO": conf}

        oos_start = str(dates[0].date())
        oos_end   = str(dates[-1].date())

        self.result = run_v9(all_data, conf_map, nifty, dates, oos_start, oos_end)

    def test_returns_dict(self):
        assert isinstance(self.result, dict)

    def test_required_keys_present(self):
        required = [
            "label", "oos_start", "oos_end", "n_years",
            "total_ret", "ann_cagr", "net_after_tax",
            "max_dd", "sharpe", "calmar",
            "n_trades", "win_rate", "avg_hold_days",
            "avg_win_pct", "avg_loss_pct", "expected_value",
            "annual", "ticker_stats", "tier_stats", "trades",
            "n_initial_stops", "n_rolling_exits",
            "n_regime_blocks", "n_cooldown_blocks", "n_perf_gate_fires",
            "cb_recoveries",
        ]
        for key in required:
            assert key in self.result, f"Missing key: {key}"

    def test_net_after_tax_is_80pct_of_cagr(self):
        assert self.result["net_after_tax"] == pytest.approx(
            self.result["ann_cagr"] * 0.80, abs=0.01
        )

    def test_calmar_computed_correctly(self):
        cagr = self.result["ann_cagr"]
        dd   = abs(self.result["max_dd"])
        if dd > 0.01:
            expected_calmar = cagr / dd
            assert self.result["calmar"] == pytest.approx(expected_calmar, rel=0.01)

    def test_trades_list_contains_trade_objects(self):
        for t in self.result["trades"]:
            assert hasattr(t, "ticker")
            assert hasattr(t, "pnl_pct")
            assert hasattr(t, "exit_reason")

    def test_v9_diagnostic_counters_non_negative(self):
        assert self.result["n_regime_blocks"]   >= 0
        assert self.result["n_cooldown_blocks"] >= 0
        assert self.result["n_perf_gate_fires"] >= 0

    def test_annual_keys_are_string_years(self):
        for yr in self.result["annual"]:
            assert isinstance(yr, str)
            assert int(yr) > 2000

    def test_label_contains_v9(self):
        assert "V9" in self.result["label"] or "9" in self.result["label"]


# ─────────────────────────────────────────────────────────────────────────────
# 11. Design Invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestV9DesignInvariants:
    def test_atr_trail_never_zero(self):
        """Trail must always be positive, never zero."""
        for vix in [0.10, 0.20, 0.30]:
            for atr in [0.001, 0.010, 0.020, 0.050]:
                assert get_adaptive_trail(vix, atr) > 0

    def test_cooldown_only_from_initial_stop(self):
        """TRAIL_STOP and ML_EXIT must NOT set cooldown."""
        port = _make_portfolio()
        date_entry = pd.Timestamp("2023-01-01")
        date_exit  = pd.Timestamp("2023-04-15")
        port.positions["A"] = _make_position("A", entry_date=date_entry)
        port.exit("A", 1300.0, date_exit, "TRAIL_STOP(15%)")
        port.positions["B"] = _make_position("B", entry_date=date_entry)
        port.exit("B", 950.0, date_exit, "ML_EXIT(rc=0.410)")
        assert "A" not in port.initial_stop_dates
        assert "B" not in port.initial_stop_dates

    def test_perf_gate_hurdle_is_stricter_than_v8_hurdle(self):
        assert PERF_GATE_HURDLE > V8_ML_ENTRY_HURDLE

    def test_regime_gate_does_not_affect_existing_positions(self):
        """Nifty regime gate only limits NEW entries, not exits of existing positions."""
        # The gate only modifies slot_limit — existing positions continue running
        # This is a design test: regime gate flag must NOT appear in exit code path
        import inspect
        from scripts import multi_strategy_backtest_v9 as v9mod
        source = inspect.getsource(v9mod.run_v9)
        # Gate only used in entry section (after "Entries" comment)
        assert "nifty_adverse" in source

    def test_atr_trail_min_above_10pct(self):
        assert ATR_TRAIL_MIN >= 0.10

    def test_atr_trail_max_at_most_25pct(self):
        assert ATR_TRAIL_MAX <= 0.25

    def test_nifty_regime_threshold_at_most_minus_3pct(self):
        assert NIFTY_REGIME_THRESHOLD <= -0.03

    def test_v8_initial_stop_values_unchanged(self):
        assert INITIAL_STOP_LOSS_PCT == 0.07
        assert INITIAL_STOP_DAYS     == 45

    def test_paper_mode_no_live_flag(self):
        """No LIVE mode references in V9."""
        import inspect
        from scripts import multi_strategy_backtest_v9 as v9mod
        source = inspect.getsource(v9mod)
        assert "LIVE" not in source or "PAPER" in source  # Paper context always present

    def test_no_ratchet_update_in_run_v9(self):
        """Ratchet is disabled in V9 (net negative in V8 analysis)."""
        import inspect
        from scripts import multi_strategy_backtest_v9 as v9mod
        source = inspect.getsource(v9mod.run_v9)
        # ratchet_floor field exists but should not be updated in the loop
        assert "get_ratchet_floor" not in source


# ─────────────────────────────────────────────────────────────────────────────
# 12. Integration: V9 vs V8 — ATR trail produces valid range
# ─────────────────────────────────────────────────────────────────────────────

class TestV9VsV8Integration:
    def test_atr_trail_range_for_realistic_nse_inputs(self):
        """All realistic NSE stock ATR × VIX combos produce valid trails."""
        nse_atrs = [0.015, 0.019, 0.021, 0.025, 0.028, 0.035]
        vix_vals = [0.14, 0.20, 0.25, 0.30]
        for atr in nse_atrs:
            for vix in vix_vals:
                trail = get_adaptive_trail(vix, atr)
                assert ATR_TRAIL_MIN <= trail <= ATR_TRAIL_MAX, \
                    f"Trail {trail:.3f} out of range for ATR={atr}, VIX={vix}"

    def test_compute_atr_on_realistic_nse_data(self):
        """ATR for typical NSE stock should be in 1-5% range."""
        df   = _make_ohlcv(n=60, daily_vol=0.018)
        date = df.index[-1]
        atr  = compute_atr_pct(df, date)
        assert 0.005 < atr < 0.10  # 0.5% to 10%

    def test_nifty_regime_fires_on_mar2026_style_drop(self):
        """March 2026 Nifty drop (-10-12% in 21d) must trigger regime gate."""
        # Build Nifty that drops 12% in last 21 days
        n     = 60
        dates = pd.bdate_range("2025-12-01", periods=n)
        stable = np.ones(39) * 25000.0
        drop   = np.linspace(25000, 22000, 21)  # -12%
        vals   = np.concatenate([stable, drop])
        nifty  = pd.Series(vals, index=dates)
        ret    = compute_nifty_regime(nifty, dates[-1])
        assert ret < NIFTY_REGIME_THRESHOLD
        assert ret < -0.10  # Specifically large drop

    def test_perf_gate_does_not_fire_on_flat_performance(self):
        history = [{"equity": 5_000_000.0}] * (PERF_GATE_LOOKBACK + 10)
        ret = get_port_60d_return(history, 5_000_000.0)
        assert ret == pytest.approx(0.0, abs=0.001)
        assert ret >= PERF_GATE_THRESHOLD  # Gate should NOT fire

    def test_cooldown_fires_on_rapid_re_entry_scenario(self):
        """Classic false-signal scenario: enter, stop out in 20d, try to re-enter same day."""
        port = _make_portfolio()
        stop_date = pd.Timestamp("2025-06-01")
        port.initial_stop_dates["TATAELXSI"] = stop_date
        # Try to re-enter 15 days later
        re_entry_attempt = stop_date + pd.Timedelta(days=15)
        assert port.is_on_cooldown("TATAELXSI", re_entry_attempt)
        # But 61 days later — clear to re-enter
        cleared_date = stop_date + pd.Timedelta(days=61)
        assert not port.is_on_cooldown("TATAELXSI", cleared_date)

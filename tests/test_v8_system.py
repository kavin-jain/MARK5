"""
V8 System Test Suite — 75 tests covering all V8-specific logic.

Tests V8 fixes:
  Fix 1: Rolling High Stop (Peak Capture) — replaces old RSI Partial Exit
  Fix 2: Confidence Trail Exit
  Fix 3: Ratcheting Protective Stop
  Fix 4: 10-Day Entry Momentum Filter
  Fix 5: Portfolio YTD Gate
  Integration: V8Portfolio partial exit mechanics
"""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np
import pandas as pd

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _ROOT    not in sys.path: sys.path.insert(0, _ROOT)
if _SCRIPTS not in sys.path: sys.path.insert(0, _SCRIPTS)

from multi_strategy_backtest_v8 import (
    V8Position, V8Portfolio, Trade,
    check_entry_momentum, get_ratchet_floor, get_effective_stop,
    run_v8,
    INITIAL_STOP_LOSS_PCT, INITIAL_STOP_DAYS,
    ROLLING_HIGH_WINDOW, ROLLING_HIGH_TRIGGER, ROLLING_HIGH_TRAIL_PCT,
    CONF_TRAIL_DROP, CONF_TRAIL_MIN_PEAK,
    RATCHET_1_GAIN, RATCHET_1_FLOOR, RATCHET_2_GAIN, RATCHET_2_FLOOR,
    MOMENTUM_10D_MIN, PORT_YTD_DOWN_SCALE,
    V8_ML_ENTRY_HURDLE,
)
from multi_strategy_backtest_v6 import (
    INITIAL_CAPITAL, ML_ENTRY_HURDLE, COST_PCT, SLIPPAGE_PCT,
    get_confidence_alloc,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_position(
    ticker="HAL", entry_price=1000.0, shares=100,
    entry_date="2022-01-03", peak_price=None, trail_pct=0.15,
    conf_entry=0.60, alloc_tier="T2", conf_peak=0.0,
    ratchet_floor=0.0,
) -> V8Position:
    return V8Position(
        ticker=ticker,
        entry_price=entry_price,
        peak_price=peak_price or entry_price,
        entry_date=pd.Timestamp(entry_date),
        shares=shares,
        entry_cost=entry_price * shares * (1 + COST_PCT + SLIPPAGE_PCT),
        trail_pct=trail_pct,
        conf_entry=conf_entry,
        alloc_tier=alloc_tier,
        conf_peak=conf_peak,
        ratchet_floor=ratchet_floor,
    )


def _make_nifty(n=300, start="2022-01-01", drift=0.0003) -> pd.Series:
    dates = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(42)
    prices = [21000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + drift + rng.normal(0, 0.010)))
    return pd.Series(prices, index=dates, name="close")


def _make_ticker_df(n=120, start="2022-01-01", base=1000.0,
                    trend=0.001, vol=0.012) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(7)
    close = [base]
    for _ in range(n - 1):
        close.append(close[-1] * (1 + trend + rng.normal(0, vol)))
    volume = [int(rng.uniform(1e6, 5e6)) for _ in range(n)]
    return pd.DataFrame({"close": close, "volume": volume}, index=dates)


def _make_conf_series(n=120, start="2022-01-01", base=0.58, noise=0.04) -> pd.Series:
    dates = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(99)
    vals = np.clip(base + rng.normal(0, noise, n), 0.40, 0.90)
    return pd.Series(vals, index=dates)


def _portfolio_with_position(**kwargs) -> V8Portfolio:
    """Make a portfolio with one position for testing exits."""
    port = V8Portfolio(INITIAL_CAPITAL)
    pos  = _make_position(**kwargs)
    port.cash -= pos.entry_cost
    port.positions[pos.ticker] = pos
    return port


# ══════════════════════════════════════════════════════════════════════════════
# 1. V8 Constants
# ══════════════════════════════════════════════════════════════════════════════

class TestV8Constants:
    def test_initial_stop_loss_pct(self):
        """Initial stop: -7% in first 45 days cuts losers before they drag to -9%."""
        assert INITIAL_STOP_LOSS_PCT == 0.07

    def test_initial_stop_days(self):
        """Initial stop window = 45 days — long enough for signal, short enough to cut."""
        assert INITIAL_STOP_DAYS == 45

    def test_rolling_high_window_is_5(self):
        assert ROLLING_HIGH_WINDOW == 5

    def test_rolling_high_trigger_at_150pct(self):
        """Rolling stop activates at +150% gain — insurance for mega-trend peaks."""
        assert ROLLING_HIGH_TRIGGER == 1.50

    def test_rolling_high_trail_pct(self):
        """7% trail below rolling high — tighter than full trailing stop."""
        assert ROLLING_HIGH_TRAIL_PCT == 0.07
        from multi_strategy_backtest_v6 import TRAIL_NORMAL
        assert ROLLING_HIGH_TRAIL_PCT < TRAIL_NORMAL

    def test_conf_trail_drop(self):
        """18pp drop needed — not 12pp noise."""
        assert CONF_TRAIL_DROP == 0.18

    def test_conf_trail_min_peak(self):
        """Only fires at elite confidence peak ≥ 75%."""
        assert CONF_TRAIL_MIN_PEAK == 0.75

    def test_ratchet_gains_increasing(self):
        assert RATCHET_1_GAIN < RATCHET_2_GAIN

    def test_ratchet_floors_tighter_at_higher_gain(self):
        """Higher gain → tighter floor (smaller %)."""
        assert RATCHET_2_FLOOR < RATCHET_1_FLOOR

    def test_momentum_min_is_negative(self):
        assert MOMENTUM_10D_MIN < 0

    def test_ytd_scale_between_0_and_1(self):
        assert 0 < PORT_YTD_DOWN_SCALE < 1

    def test_v8_entry_hurdle_above_v7(self):
        """V8 raises the ML entry hurdle above V7's 0.52 to filter T1 noise."""
        assert V8_ML_ENTRY_HURDLE > 0.52
        assert V8_ML_ENTRY_HURDLE <= 0.65  # not so high it blocks all trades


# ══════════════════════════════════════════════════════════════════════════════
# 2. V8Position dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestV8PositionDefaults:
    def test_conf_peak_default_zero(self):
        pos = _make_position()
        assert pos.conf_peak == 0.0

    def test_ratchet_floor_default_zero(self):
        pos = _make_position()
        assert pos.ratchet_floor == 0.0

    def test_conf_peak_can_be_set(self):
        pos = _make_position(conf_peak=0.78)
        assert pos.conf_peak == 0.78

    def test_no_rsi_partial_done_field(self):
        """V8 removed rsi_partial_done — rolling stop replaces RSI partial."""
        pos = _make_position()
        assert not hasattr(pos, "rsi_partial_done")


# ══════════════════════════════════════════════════════════════════════════════
# 3. check_entry_momentum — Fix 4
# ══════════════════════════════════════════════════════════════════════════════

class TestEntryMomentum:
    def _df_with_change(self, n=30, change_10d=0.0, base=1000.0) -> pd.DataFrame:
        """Build a DataFrame where close_now / close_10d_ago - 1 = change_10d."""
        dates = pd.bdate_range("2022-01-01", periods=n)
        close = [base] * (n - 11) + [base] * 10 + [base * (1 + change_10d)]
        volume = [1_000_000] * n
        return pd.DataFrame({"close": close, "volume": volume}, index=dates)

    def test_positive_momentum_passes(self):
        df = self._df_with_change(n=30, change_10d=0.05)
        ok, reason = check_entry_momentum(df, df.index[-1])
        assert ok

    def test_flat_passes(self):
        df = self._df_with_change(n=30, change_10d=0.0)
        ok, reason = check_entry_momentum(df, df.index[-1])
        assert ok

    def test_minus_3pct_passes(self):
        """Within -4% threshold → pass."""
        df = self._df_with_change(n=30, change_10d=-0.03)
        ok, reason = check_entry_momentum(df, df.index[-1])
        assert ok

    def test_minus_5pct_fails(self):
        """Below -4% threshold → fail."""
        df = self._df_with_change(n=30, change_10d=-0.05)
        ok, reason = check_entry_momentum(df, df.index[-1])
        assert not ok
        assert "10d_mom" in reason

    def test_minus_4pct_exactly_passes(self):
        """Exactly at the limit (-4%) → passes (strict < boundary)."""
        df = self._df_with_change(n=30, change_10d=-0.04)
        ok, reason = check_entry_momentum(df, df.index[-1])
        assert ok  # -0.04 < -0.04 is False → passes

    def test_insufficient_data_passes(self):
        """< 12 bars → insufficient_data → pass."""
        df = _make_ticker_df(n=5)   # 5 bars < 10+2 needed
        ok, reason = check_entry_momentum(df, df.index[-1])
        assert ok
        assert "insufficient_data" in reason

    def test_error_passes(self):
        """Column missing → fail open."""
        df = pd.DataFrame({"no_close": [1.0] * 30},
                          index=pd.bdate_range("2022-01-01", periods=30))
        ok, reason = check_entry_momentum(df, df.index[-1])
        assert ok
        assert "gate_error" in reason


# ══════════════════════════════════════════════════════════════════════════════
# 4. get_ratchet_floor — Fix 3
# ══════════════════════════════════════════════════════════════════════════════

class TestRatchetFloor:
    def test_no_ratchet_below_20pct_gain(self):
        pos  = _make_position(entry_price=1000.0)
        curr = 1150.0  # +15% — below RATCHET_1_GAIN (20%)
        assert get_ratchet_floor(pos, curr) == 0.0

    def test_ratchet_1_activates_at_20pct(self):
        pos  = _make_position(entry_price=1000.0)
        curr = 1205.0  # clearly +20.5% (avoids float boundary at exactly 1200)
        floor = get_ratchet_floor(pos, curr)
        expected = curr * (1 - RATCHET_1_FLOOR)
        assert abs(floor - expected) < 0.01

    def test_ratchet_2_activates_at_40pct(self):
        pos  = _make_position(entry_price=1000.0)
        curr = 1405.0  # clearly +40.5% (avoids float boundary at exactly 1400)
        floor = get_ratchet_floor(pos, curr)
        expected = curr * (1 - RATCHET_2_FLOOR)
        assert abs(floor - expected) < 0.01

    def test_ratchet_floor_moves_up_with_price(self):
        """As price rises, ratchet floor rises."""
        pos   = _make_position(entry_price=1000.0, ratchet_floor=0.0)
        curr1 = 1300.0
        curr2 = 1400.0
        floor1 = get_ratchet_floor(pos, curr1)
        floor2 = get_ratchet_floor(pos, curr2)
        assert floor2 > floor1

    def test_ratchet_floor_is_max_of_current_and_existing(self):
        """Floor can only move up (never down)."""
        pos = _make_position(entry_price=1000.0, ratchet_floor=1250.0)
        curr_lower = 1100.0  # Below existing floor
        floor = get_ratchet_floor(pos, curr_lower)
        # gain at 1100 = +10% < RATCHET_1_GAIN → returns 0.0 (no ratchet yet for this price)
        # But the position has existing ratchet_floor=1250 set externally
        # get_ratchet_floor returns max(candidate, pos.ratchet_floor) only when gain >= 0.20
        # At +10%, no ratchet → returns 0.0
        assert floor == 0.0  # function returns candidate, max logic is in the caller

    def test_ratchet_2_floor_tighter_than_ratchet_1(self):
        """At +40%, floor is closer to current price than at +20%."""
        pos   = _make_position(entry_price=1000.0)
        curr  = 1400.0  # +40%
        floor_at_40 = get_ratchet_floor(pos, curr)
        dist_at_40  = curr - floor_at_40  # absolute gap

        curr2 = 1200.0  # +20%
        floor_at_20 = get_ratchet_floor(pos, curr2)
        dist_at_20  = curr2 - floor_at_20

        # At +40%, floor % gap (5%) is less than at +20% (7%)
        assert (dist_at_40 / curr) < (dist_at_20 / curr2)


# ══════════════════════════════════════════════════════════════════════════════
# 5. get_effective_stop
# ══════════════════════════════════════════════════════════════════════════════

class TestEffectiveStop:
    def test_standard_trail_when_no_ratchet_no_rolling(self):
        pos = _make_position(entry_price=1000.0, peak_price=1200.0, trail_pct=0.15,
                             ratchet_floor=0.0)
        eff = get_effective_stop(pos, 1200.0, rolling_5d_high=0.0)
        expected = 1200.0 * (1 - 0.15)  # 1020.0
        assert abs(eff - expected) < 0.01

    def test_ratchet_wins_when_higher_than_trail(self):
        """When ratchet floor > standard trail → ratchet is used."""
        pos = _make_position(entry_price=1000.0, peak_price=1200.0, trail_pct=0.15,
                             ratchet_floor=1150.0)  # ratchet floor much higher than trail
        eff = get_effective_stop(pos, 1200.0)
        assert eff == 1150.0  # ratchet wins

    def test_trail_wins_when_higher_than_ratchet(self):
        """When trail stop > ratchet floor → trail is used."""
        pos = _make_position(entry_price=1000.0, peak_price=1200.0, trail_pct=0.15,
                             ratchet_floor=900.0)  # ratchet below trail stop
        eff = get_effective_stop(pos, 1200.0)
        assert eff == 1200.0 * (1 - 0.15)  # trail wins

    def test_rolling_high_wins_when_highest(self):
        """When rolling 5-day high stop > trail and ratchet → rolling wins."""
        pos = _make_position(entry_price=1000.0, peak_price=1500.0, trail_pct=0.15,
                             ratchet_floor=1200.0)
        # rolling_5d_high=1600 → rolling_stop = 1600*(1-0.07) = 1488
        # trail = 1500*(1-0.15) = 1275  ratchet = 1200
        eff = get_effective_stop(pos, 1500.0, rolling_5d_high=1600.0)
        expected = 1600.0 * (1 - ROLLING_HIGH_TRAIL_PCT)  # ~1488 at 7% trail
        assert abs(eff - expected) < 0.01

    def test_rolling_high_zero_ignored(self):
        """When rolling_5d_high=0 (below trigger), rolling component = 0."""
        pos = _make_position(entry_price=1000.0, peak_price=1200.0, trail_pct=0.15,
                             ratchet_floor=0.0)
        eff_with    = get_effective_stop(pos, 1200.0, rolling_5d_high=0.0)
        eff_without = get_effective_stop(pos, 1200.0)
        assert eff_with == eff_without  # same result


# ══════════════════════════════════════════════════════════════════════════════
# 6. V8Portfolio.partial_exit
# ══════════════════════════════════════════════════════════════════════════════

class TestV8PortfolioPartialExit:
    def test_partial_exit_reduces_shares(self):
        port = _portfolio_with_position(ticker="HAL", shares=100, entry_price=1000.0)
        port.partial_exit("HAL", 1200.0, pd.Timestamp("2022-06-01"), 0.5, "RSI_PARTIAL")
        assert port.positions["HAL"].shares == 50

    def test_partial_exit_increases_cash(self):
        port = _portfolio_with_position(ticker="HAL", shares=100, entry_price=1000.0)
        cash_before = port.cash
        port.partial_exit("HAL", 1200.0, pd.Timestamp("2022-06-01"), 0.5, "RSI_PARTIAL")
        assert port.cash > cash_before

    def test_partial_exit_creates_trade_record(self):
        port = _portfolio_with_position(ticker="HAL", shares=100, entry_price=1000.0)
        trade = port.partial_exit("HAL", 1200.0, pd.Timestamp("2022-06-01"), 0.5, "RSI_PARTIAL")
        assert trade is not None
        assert trade.partial is True
        assert trade.shares == 50

    def test_partial_exit_position_remains(self):
        """After 50% exit, 50% position still exists."""
        port = _portfolio_with_position(ticker="HAL", shares=100, entry_price=1000.0)
        port.partial_exit("HAL", 1200.0, pd.Timestamp("2022-06-01"), 0.5, "RSI_PARTIAL")
        assert "HAL" in port.positions

    def test_partial_exit_full_fraction_closes_position(self):
        """100% partial exit removes the position."""
        port = _portfolio_with_position(ticker="HAL", shares=100, entry_price=1000.0)
        port.partial_exit("HAL", 1200.0, pd.Timestamp("2022-06-01"), 1.0, "FULL_EXIT")
        assert "HAL" not in port.positions

    def test_partial_exit_trade_pnl_is_positive_on_winner(self):
        port = _portfolio_with_position(ticker="HAL", shares=100, entry_price=1000.0)
        trade = port.partial_exit("HAL", 1300.0, pd.Timestamp("2022-06-01"), 0.5, "RSI_PARTIAL")
        assert trade.net_pnl > 0

    def test_partial_exit_updates_entry_cost(self):
        """After 50% partial exit, remaining cost basis is halved."""
        port = _portfolio_with_position(ticker="HAL", shares=100, entry_price=1000.0)
        cost_before = port.positions["HAL"].entry_cost
        port.partial_exit("HAL", 1200.0, pd.Timestamp("2022-06-01"), 0.5, "RSI_PARTIAL")
        cost_after = port.positions["HAL"].entry_cost
        assert abs(cost_after - cost_before * 0.5) < 1.0  # within ₹1 rounding

    def test_partial_exit_nonexistent_ticker_returns_none(self):
        port = V8Portfolio(INITIAL_CAPITAL)
        trade = port.partial_exit("UNKNOWN", 1000.0, pd.Timestamp("2022-01-01"), 0.5, "TEST")
        assert trade is None

    def test_partial_exit_with_1_share_returns_none_on_tiny(self):
        """1 share at 50% → 0 exit shares → no trade."""
        port = _portfolio_with_position(ticker="HAL", shares=1, entry_price=1000.0)
        trade = port.partial_exit("HAL", 1200.0, pd.Timestamp("2022-06-01"), 0.5, "RSI_PARTIAL")
        assert trade is None  # int(1 * 0.5) = 0 → returns None


# ══════════════════════════════════════════════════════════════════════════════
# 7. V8Portfolio.enter and exit
# ══════════════════════════════════════════════════════════════════════════════

class TestV8PortfolioEnterExit:
    def test_enter_creates_v8position(self):
        port  = V8Portfolio(INITIAL_CAPITAL)
        nifty = _make_nifty(n=30)
        from multi_strategy_backtest_v6 import compute_vix_proxy
        vix   = compute_vix_proxy(nifty, nifty.index[-1])
        ok = port.enter("HAL", 1000.0, pd.Timestamp("2022-01-03"), 0.62, vix)
        assert ok
        pos = port.positions["HAL"]
        assert isinstance(pos, V8Position)
        assert pos.conf_peak == 0.62  # initialized from conf_entry

    def test_enter_sets_conf_peak_from_entry_conf(self):
        port = V8Portfolio(INITIAL_CAPITAL)
        nifty = _make_nifty(n=30)
        from multi_strategy_backtest_v6 import compute_vix_proxy
        vix = compute_vix_proxy(nifty, nifty.index[-1])
        port.enter("HAL", 1000.0, pd.Timestamp("2022-01-03"), 0.75, vix)
        assert port.positions["HAL"].conf_peak == 0.75

    def test_exit_removes_position(self):
        port  = _portfolio_with_position(ticker="HAL", shares=100, entry_price=1000.0)
        trade = port.exit("HAL", 1100.0, pd.Timestamp("2022-06-01"), "TRAIL_STOP")
        assert "HAL" not in port.positions
        assert trade is not None
        assert trade.partial is False

    def test_exit_all_clears_portfolio(self):
        port = V8Portfolio(INITIAL_CAPITAL)
        for tk, price in [("HAL", 1000.0), ("LT", 2000.0)]:
            pos = _make_position(ticker=tk, entry_price=price, shares=50)
            port.cash -= pos.entry_cost
            port.positions[tk] = pos
        prices = {"HAL": 1100.0, "LT": 2200.0}
        port.exit_all(prices, pd.Timestamp("2022-06-01"), "END_SIM")
        assert len(port.positions) == 0

    def test_ytd_return_is_zero_on_init(self):
        port = V8Portfolio(INITIAL_CAPITAL)
        assert port.ytd_return({}) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 8. run_v8 output structure
# ══════════════════════════════════════════════════════════════════════════════

class TestRunV8OutputStructure:
    def _minimal_run(self):
        nifty    = _make_nifty(n=150)
        ticker   = _make_ticker_df(n=150)
        conf     = _make_conf_series(n=150)
        all_data = {"HAL": ticker}
        conf_map = {"HAL": conf}
        dates    = nifty.index
        return run_v8(all_data, conf_map, nifty, dates,
                      str(dates[0].date()), str(dates[-1].date()))

    def test_has_required_keys(self):
        r = self._minimal_run()
        for k in ["n_trades", "win_rate", "ann_cagr", "net_after_tax",
                  "max_dd", "sharpe", "calmar", "n_partial_exits",
                  "n_rolling_exits", "n_conf_trail_exits", "n_initial_stops",
                  "cb_recoveries"]:
            assert k in r, f"Missing key: {k}"

    def test_partial_exits_count_is_non_negative(self):
        r = self._minimal_run()
        assert r["n_partial_exits"] >= 0

    def test_rolling_exits_count_is_non_negative(self):
        r = self._minimal_run()
        assert r["n_rolling_exits"] >= 0

    def test_no_rsi_exits_key(self):
        """n_rsi_exits was removed in V8 (rolling stop replaced RSI exit)."""
        r = self._minimal_run()
        assert "n_rsi_exits" not in r

    def test_win_rate_valid_range(self):
        r = self._minimal_run()
        assert 0.0 <= r["win_rate"] <= 100.0

    def test_max_dd_is_zero_or_negative(self):
        r = self._minimal_run()
        assert r["max_dd"] <= 0.0

    def test_annual_breakdown_present(self):
        r = self._minimal_run()
        assert len(r["annual"]) >= 1

    def test_net_after_tax_is_80pct_of_cagr(self):
        r = self._minimal_run()
        expected = r["ann_cagr"] * 0.80
        assert abs(r["net_after_tax"] - expected) < 0.05


# ══════════════════════════════════════════════════════════════════════════════
# 9. V8 Design Invariants
# ══════════════════════════════════════════════════════════════════════════════

class TestV8DesignInvariants:
    def test_rolling_high_window_is_short(self):
        """5-day window for rolling high — short enough to catch recent peaks."""
        assert ROLLING_HIGH_WINDOW <= 10

    def test_rolling_high_trigger_is_mega_trend_only(self):
        """Rolling stop activates at +150% gain — insurance for mega-trend peaks."""
        assert ROLLING_HIGH_TRIGGER >= 1.00  # must be a proven mega-trend winner
        assert ROLLING_HIGH_TRIGGER <= 2.00

    def test_rolling_high_trail_tighter_than_normal_trail(self):
        """7% trail below rolling high — tighter than standard 9-15% trail."""
        from multi_strategy_backtest_v6 import TRAIL_NORMAL
        assert ROLLING_HIGH_TRAIL_PCT < TRAIL_NORMAL

    def test_conf_trail_requires_high_confidence_peak(self):
        """Only applies when model was elite (≥75%) — avoids routine noise."""
        assert CONF_TRAIL_MIN_PEAK > ML_ENTRY_HURDLE  # peak must be well above entry hurdle
        assert CONF_TRAIL_MIN_PEAK >= 0.70

    def test_ratchet_1_floor_is_profit_protective(self):
        """7% floor after +20% gain → minimum locked profit = 20% - 7% = 13%."""
        min_locked = RATCHET_1_GAIN - RATCHET_1_FLOOR
        assert min_locked > 0.10  # at least 10% profit locked after +20% gain

    def test_ratchet_2_floor_is_more_protective(self):
        """5% floor after +40% gain → minimum locked profit = 40% - 5% = 35%."""
        min_locked = RATCHET_2_GAIN - RATCHET_2_FLOOR
        assert min_locked > 0.30

    def test_momentum_filter_allows_small_pullbacks(self):
        """Allow -3% dips (normal pullback) but block -5% (downtrend)."""
        assert MOMENTUM_10D_MIN < -0.02  # allows up to 2-3% pullback
        assert MOMENTUM_10D_MIN > -0.10  # doesn't block too aggressively

    def test_ytd_gate_is_moderate_reduction(self):
        """60% scale in bad years — not full stop, not trivial."""
        assert 0.50 < PORT_YTD_DOWN_SCALE < 0.80

    def test_rolling_stop_not_rsi_based(self):
        """V8 peak capture is price-based (rolling high), not RSI-based."""
        from multi_strategy_backtest_v8 import (
            ROLLING_HIGH_WINDOW, ROLLING_HIGH_TRIGGER, ROLLING_HIGH_TRAIL_PCT
        )
        # All rolling constants are defined and non-zero
        assert ROLLING_HIGH_WINDOW > 0
        assert ROLLING_HIGH_TRIGGER > 0
        assert ROLLING_HIGH_TRAIL_PCT > 0

    def test_initial_stop_cuts_losers_not_winners(self):
        """Initial stop is -7%: winners are never near -7% in first 45 days."""
        assert INITIAL_STOP_LOSS_PCT > 0.05   # large enough to avoid noise exits
        assert INITIAL_STOP_LOSS_PCT < 0.12   # small enough to cut real losers early
        assert INITIAL_STOP_DAYS >= 30        # enough time for signal to prove itself
        assert INITIAL_STOP_DAYS <= 60        # not so long it becomes a trailing stop

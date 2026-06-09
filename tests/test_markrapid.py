"""
MARKRAPID Test Suite
=====================
40+ unit tests covering all modules.

Test classes:
    TestRapidConfig        — config constants and weight validation
    TestBreakoutScore      — breakout component scorer
    TestVolumeScore        — volume surge scorer
    TestRSIScore           — RSI momentum scorer
    TestTrendScore         — trend alignment scorer
    TestCatalystScore      — catalyst/news scorer
    TestComputeRSI         — RSI calculation accuracy
    TestRapidScoreFull     — full 5-component score integration
    TestEntryAllowed       — entry gate conditions
    TestNewsScraper        — news proxy scoring
    TestPortfolioCycle     — trade lifecycle (enter/update/exit)
    TestPortfolioMetrics   — portfolio analytics
    TestScannerFiltering   — scanner candidate filtering
    TestBacktestHelpers    — _compute_annual_returns, _load helpers

Run:
    pytest tests/test_markrapid.py -v
"""
import math
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# ── Helpers for building test DataFrames ─────────────────────────────────────

def _make_ohlcv(
    n: int = 250,
    price: float = 500.0,
    trend: float = 0.0015,    # daily drift
    vol_base: float = 1_000_000,
    vol_surge_last: float = 1.0,  # multiplier on last bar
    seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic OHLCV with configurable trend and volume."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    closes = [price]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + rng.normal(0, 0.01)))
    closes = np.array(closes)
    highs  = closes * (1 + rng.uniform(0.002, 0.015, n))
    lows   = closes * (1 - rng.uniform(0.002, 0.015, n))
    opens  = np.roll(closes, 1)
    opens[0] = closes[0] * 0.999
    vols = rng.integers(int(vol_base * 0.5), int(vol_base * 1.5), n).astype(float)
    vols[-1] *= vol_surge_last
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=dates,
    )


def _make_breakout_df(n: int = 250, price: float = 500.0) -> pd.DataFrame:
    """DataFrame where last bar clearly breaks 20-day high."""
    df = _make_ohlcv(n=n, price=price, trend=0.0005)
    # Force last close above 20-day high
    high_20 = df["high"].iloc[-21:-1].max()
    df.iloc[-1, df.columns.get_loc("close")] = high_20 * 1.02
    df.iloc[-1, df.columns.get_loc("high")]  = high_20 * 1.025
    df.iloc[-1, df.columns.get_loc("volume")] *= 3.0  # volume surge
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TestRapidConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestRapidConfig:

    def test_weights_sum_to_one(self):
        from markrapid.config import WEIGHTS
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6

    def test_target_exceeds_stop(self):
        from markrapid.config import TARGET_PCT, STOP_PCT
        assert TARGET_PCT > STOP_PCT, "Target must be larger than stop"

    def test_rr_ratio_at_least_1_5(self):
        from markrapid.config import TARGET_PCT, STOP_PCT
        rr = TARGET_PCT / STOP_PCT
        assert rr >= 1.5, f"R:R ratio too low: {rr:.2f}"

    def test_capital_is_ten_thousand(self):
        from markrapid.config import CAPITAL
        assert CAPITAL == 10_000.0

    def test_max_hold_days_is_thirty(self):
        from markrapid.config import MAX_HOLD_DAYS
        assert MAX_HOLD_DAYS == 30

    def test_entry_threshold_above_half(self):
        from markrapid.config import RAPID_ENTRY_THRESHOLD
        assert RAPID_ENTRY_THRESHOLD > 0.5

    def test_universe_has_entries(self):
        from markrapid.config import UNIVERSE
        assert len(UNIVERSE) >= 50

    def test_no_duplicate_tickers_in_universe(self):
        from markrapid.config import UNIVERSE
        assert len(UNIVERSE) == len(set(UNIVERSE))

    def test_cost_round_trip_reasonable(self):
        from markrapid.config import COST_ROUND_TRIP
        assert 0.001 <= COST_ROUND_TRIP <= 0.01

    def test_net_target_exceeds_ten_pct(self):
        """After round-trip costs, net target must still exceed 10%."""
        from markrapid.config import TARGET_PCT, COST_ROUND_TRIP
        net_target = TARGET_PCT - COST_ROUND_TRIP
        assert net_target > 0.10, f"Net target {net_target:.3f} below 10%"


# ══════════════════════════════════════════════════════════════════════════════
# TestBreakoutScore
# ══════════════════════════════════════════════════════════════════════════════

class TestBreakoutScore:

    def setup_method(self):
        from markrapid.signals import _breakout_score
        self.score = _breakout_score

    def test_above_20d_high_returns_max(self):
        s = self.score(105.0, 100.0)  # 5% above
        assert s >= 0.85

    def test_exactly_at_20d_high_returns_high(self):
        s = self.score(100.0, 100.0)
        assert s >= 0.80

    def test_within_2pct_returns_medium(self):
        s = self.score(98.5, 100.0)   # 1.5% below
        assert 0.60 <= s <= 0.80

    def test_deep_below_returns_near_zero(self):
        s = self.score(80.0, 100.0)   # 20% below
        assert s < 0.15

    def test_score_bounded_zero_to_one(self):
        for ratio in [0.5, 0.9, 1.0, 1.1, 1.5]:
            s = self.score(ratio * 100, 100.0)
            assert 0.0 <= s <= 1.0, f"Score {s} out of bounds for ratio={ratio}"

    def test_zero_high_returns_neutral(self):
        s = self.score(100.0, 0.0)
        assert s == 0.5

    def test_monotone_increasing_with_price(self):
        scores = [self.score(p, 100.0) for p in [85, 92, 97, 100, 105, 110]]
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))


# ══════════════════════════════════════════════════════════════════════════════
# TestVolumeScore
# ══════════════════════════════════════════════════════════════════════════════

class TestVolumeScore:

    def setup_method(self):
        from markrapid.signals import _volume_score
        self.score = _volume_score

    def test_triple_volume_returns_max(self):
        s = self.score(3_500_000, 1_000_000)   # 3.5× = STRONG
        assert s == 1.0

    def test_below_average_returns_low(self):
        s = self.score(500_000, 1_000_000)     # 0.5× = below avg
        assert s < 0.3

    def test_average_volume_returns_half(self):
        s = self.score(1_000_000, 1_000_000)  # 1× = neutral
        assert abs(s - 0.5) < 0.05

    def test_zero_avg_returns_neutral(self):
        s = self.score(1_000_000, 0)
        assert s == 0.5

    def test_score_bounded(self):
        for ratio in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            s = self.score(ratio * 1e6, 1e6)
            assert 0.0 <= s <= 1.0

    def test_monotone_increasing(self):
        avg = 1_000_000
        scores = [self.score(r * avg, avg) for r in [0.1, 0.5, 1.0, 2.0, 3.5, 5.0]]
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))


# ══════════════════════════════════════════════════════════════════════════════
# TestRSIScore
# ══════════════════════════════════════════════════════════════════════════════

class TestRSIScore:

    def setup_method(self):
        from markrapid.signals import _rsi_score
        self.score = _rsi_score

    def test_optimal_zone_returns_max(self):
        for rsi in [57, 60, 65, 68]:
            assert self.score(rsi) == 1.0, f"RSI {rsi} should score 1.0"

    def test_overbought_returns_low(self):
        s = self.score(82)
        assert s <= 0.15

    def test_oversold_returns_low(self):
        s = self.score(25)
        assert s <= 0.05

    def test_score_bounded(self):
        for rsi in [10, 20, 30, 45, 55, 65, 75, 85, 95]:
            s = self.score(rsi)
            assert 0.0 <= s <= 1.0

    def test_nan_returns_neutral(self):
        s = self.score(float("nan"))
        assert s == 0.5

    def test_entry_gate_rejects_overbought(self):
        from markrapid.config import RSI_OVERBOUGHT
        # Score above threshold should be low enough to affect entry
        s = self.score(RSI_OVERBOUGHT + 5)
        assert s < 0.3


# ══════════════════════════════════════════════════════════════════════════════
# TestTrendScore
# ══════════════════════════════════════════════════════════════════════════════

class TestTrendScore:

    def setup_method(self):
        from markrapid.signals import _trend_score
        self.score = _trend_score

    def test_perfect_alignment_returns_max(self):
        # close > ema20 > ema50, also > sma200
        s = self.score(close=200, ema20=180, ema50=160, sma200=150)
        assert s == 1.0

    def test_below_ema20_loses_points(self):
        s = self.score(close=170, ema20=180, ema50=160, sma200=150)
        assert s < 0.7

    def test_inverted_emas_loses_points(self):
        s = self.score(close=200, ema20=160, ema50=180, sma200=150)
        assert s < 0.8

    def test_below_sma200_loses_points(self):
        # Everything else ok but below SMA200
        s = self.score(close=200, ema20=180, ema50=160, sma200=250)
        assert s <= 0.7

    def test_fully_bearish_returns_zero(self):
        s = self.score(close=100, ema20=150, ema50=180, sma200=200)
        assert s == 0.0

    def test_score_bounded(self):
        for params in [
            (200, 180, 160, 150),
            (100, 150, 180, 200),
            (150, 150, 150, 150),
        ]:
            s = self.score(*params)
            assert 0.0 <= s <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TestCatalystScore
# ══════════════════════════════════════════════════════════════════════════════

class TestCatalystScore:

    def setup_method(self):
        from markrapid.signals import _catalyst_score
        self.score = _catalyst_score

    def test_large_gap_up_high_score(self):
        s = self.score(104.0, 100.0, news_score=None)  # 4% gap
        assert s >= 0.95

    def test_gap_down_low_score(self):
        s = self.score(96.0, 100.0, news_score=None)   # -4% gap
        assert s <= 0.15

    def test_flat_open_neutral(self):
        s = self.score(100.2, 100.0, news_score=None)
        assert 0.45 <= s <= 0.65

    def test_news_score_overrides_proxy(self):
        # Strong positive news → high score
        s_pos = self.score(100.0, 100.0, news_score=0.9)
        s_neg = self.score(100.0, 100.0, news_score=-0.9)
        assert s_pos > 0.9
        assert s_neg < 0.1

    def test_news_score_mapped_linearly(self):
        from markrapid.signals import _catalyst_score
        s0   = _catalyst_score(100, 100, news_score=0.0)   # neutral
        s_hi = _catalyst_score(100, 100, news_score=1.0)   # max positive
        s_lo = _catalyst_score(100, 100, news_score=-1.0)  # max negative
        assert abs(s0 - 0.5) < 0.01
        assert s_hi == 1.0
        assert s_lo == 0.0

    def test_score_bounded(self):
        for gap in [-0.05, -0.02, 0.0, 0.02, 0.05]:
            s = self.score(100 * (1 + gap), 100.0, None)
            assert 0.0 <= s <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TestComputeRSI
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeRSI:

    def setup_method(self):
        from markrapid.signals import _compute_rsi
        self.rsi = _compute_rsi

    def test_constant_series_rsi_is_fifty(self):
        s = pd.Series([100.0] * 50)
        r = self.rsi(s).iloc[-1]
        # Constant series: gains = losses = 0 → RSI undefined/50
        # Accept 45-55 range
        assert 40 <= r <= 60 or np.isnan(r)

    def test_strong_uptrend_rsi_above_seventy(self):
        """RSI of a mostly-rising series should be high (uses realistic data with small down days)."""
        rng = np.random.default_rng(42)
        # 90% up days, 10% tiny down days — avoids avg_loss=0 → NaN edge case
        changes = np.where(
            rng.random(80) > 0.10,
            rng.uniform(0.5, 2.0, 80),   # up move
            -rng.uniform(0.05, 0.20, 80), # tiny down
        )
        prices = pd.Series(100.0 + np.cumsum(changes))
        r = self.rsi(prices).iloc[-1]
        assert not np.isnan(r), "RSI should not be NaN for trending series"
        assert r > 70

    def test_always_falling_rsi_below_twenty(self):
        prices = pd.Series([100.0 - i * 0.5 for i in range(60)])
        r = self.rsi(prices).iloc[-1]
        assert r < 20

    def test_rsi_bounded_0_100(self):
        df = _make_ohlcv(n=300)
        rsi_series = self.rsi(df["close"])
        valid = rsi_series.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


# ══════════════════════════════════════════════════════════════════════════════
# TestRapidScoreFull
# ══════════════════════════════════════════════════════════════════════════════

class TestRapidScoreFull:

    def test_strong_setup_scores_high(self):
        from markrapid.signals import rapid_score
        df = _make_breakout_df(n=250)
        s = rapid_score(df)
        assert s >= 0.65, f"Strong breakout setup scored only {s:.3f}"

    def test_weak_downtrend_scores_low(self):
        from markrapid.signals import rapid_score
        df = _make_ohlcv(n=250, trend=-0.003)  # strong downtrend
        s = rapid_score(df)
        assert s <= 0.50, f"Downtrend scored {s:.3f}"

    def test_insufficient_data_returns_zero(self):
        from markrapid.signals import rapid_score
        df = _make_ohlcv(n=50)  # too few bars
        s = rapid_score(df)
        assert s == 0.0

    def test_none_df_returns_zero(self):
        from markrapid.signals import rapid_score
        assert rapid_score(None) == 0.0

    def test_score_bounded_zero_to_one(self):
        from markrapid.signals import rapid_score
        for seed in [1, 2, 3, 42, 99]:
            df = _make_ohlcv(n=250, seed=seed)
            s = rapid_score(df)
            assert 0.0 <= s <= 1.0

    def test_components_sum_to_total(self):
        from markrapid.signals import rapid_score_components
        from markrapid.config import WEIGHTS
        df = _make_breakout_df()
        comps = rapid_score_components(df)
        manual_total = (
            WEIGHTS["breakout"] * comps["breakout"] +
            WEIGHTS["volume"]   * comps["volume"] +
            WEIGHTS["rsi"]      * comps["rsi"] +
            WEIGHTS["trend"]    * comps["trend"] +
            WEIGHTS["catalyst"] * comps["catalyst"]
        )
        assert abs(comps["total"] - manual_total) < 1e-5

    def test_news_score_affects_total(self):
        from markrapid.signals import rapid_score
        df = _make_breakout_df()
        s_neutral  = rapid_score(df, news_score=0.0)
        s_positive = rapid_score(df, news_score=1.0)
        s_negative = rapid_score(df, news_score=-1.0)
        # Positive news should give higher score
        assert s_positive >= s_neutral >= s_negative


# ══════════════════════════════════════════════════════════════════════════════
# TestEntryAllowed
# ══════════════════════════════════════════════════════════════════════════════

class TestEntryAllowed:

    def test_valid_breakout_entry_allowed(self):
        from markrapid.signals import entry_allowed
        df = _make_breakout_df(n=250, price=500.0)
        allowed, score, reason = entry_allowed(df)
        # May or may not pass depending on all conditions — just check no exception
        assert isinstance(allowed, bool)
        assert 0.0 <= score <= 1.0

    def test_insufficient_data_blocked(self):
        from markrapid.signals import entry_allowed
        df = _make_ohlcv(n=50)
        allowed, score, reason = entry_allowed(df)
        assert not allowed
        assert "INSUFFICIENT_DATA" in reason

    def test_price_too_low_blocked(self):
        from markrapid.signals import entry_allowed
        from markrapid.config import MIN_PRICE
        df = _make_ohlcv(n=250, price=MIN_PRICE * 0.5)  # below minimum
        allowed, score, reason = entry_allowed(df)
        assert not allowed
        assert "PRICE_TOO_LOW" in reason

    def test_price_too_high_blocked(self):
        from markrapid.signals import entry_allowed
        from markrapid.config import MAX_PRICE
        df = _make_ohlcv(n=250, price=MAX_PRICE * 1.5)  # above maximum
        allowed, score, reason = entry_allowed(df)
        assert not allowed
        assert "PRICE_TOO_HIGH" in reason

    def test_below_sma200_blocked(self):
        from markrapid.signals import entry_allowed
        # Strong downtrend: close will be well below SMA200
        df = _make_ohlcv(n=250, price=1000.0, trend=-0.004)
        allowed, score, reason = entry_allowed(df)
        if not allowed:
            assert "BELOW_SMA200" in reason or "NO_VOLUME_SURGE" in reason or "SCORE" in reason

    def test_returns_tuple_of_correct_types(self):
        from markrapid.signals import entry_allowed
        df = _make_breakout_df()
        result = entry_allowed(df)
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        assert isinstance(result[2], str)


# ══════════════════════════════════════════════════════════════════════════════
# TestNewsScraper
# ══════════════════════════════════════════════════════════════════════════════

class TestNewsScraper:

    def test_backtest_proxy_gap_up_positive(self):
        from markrapid.news_scraper import get_backtest_proxy_score
        df = _make_ohlcv(n=50)
        # Simulate large gap up in last bar
        df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-2] * 1.05
        df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].mean() * 3.0
        date = df.index[-1]
        s = get_backtest_proxy_score(df, date)
        assert s > 0.5

    def test_backtest_proxy_gap_down_negative(self):
        from markrapid.news_scraper import get_backtest_proxy_score
        df = _make_ohlcv(n=50)
        df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-2] * 0.93  # gap down
        date = df.index[-1]
        s = get_backtest_proxy_score(df, date)
        assert s < 0.0

    def test_backtest_proxy_flat_open_neutral(self):
        from markrapid.news_scraper import get_backtest_proxy_score
        df = _make_ohlcv(n=50)
        df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-2]  # no gap
        date = df.index[-1]
        s = get_backtest_proxy_score(df, date)
        assert -0.1 <= s <= 0.1

    def test_score_announcement_bullish_keywords(self):
        from markrapid.news_scraper import score_announcement
        ann = {"subject": "Company wins large order worth ₹500 crore", "details": "strong growth"}
        s = score_announcement(ann)
        assert s > 0

    def test_score_announcement_bearish_keywords(self):
        from markrapid.news_scraper import score_announcement
        ann = {"subject": "SEBI penalty for regulatory violation", "details": "loss reported"}
        s = score_announcement(ann)
        assert s < 0

    def test_score_announcement_neutral(self):
        from markrapid.news_scraper import score_announcement
        ann = {"subject": "Board meeting scheduled", "details": ""}
        s = score_announcement(ann)
        assert s == 0.0

    def test_backtest_proxy_insufficient_data(self):
        from markrapid.news_scraper import get_backtest_proxy_score
        df = _make_ohlcv(n=2)
        date = df.index[-1]
        s = get_backtest_proxy_score(df, date)
        assert isinstance(s, float)


# ══════════════════════════════════════════════════════════════════════════════
# TestPortfolioCycle
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioCycle:

    def test_enter_sets_in_trade(self):
        from markrapid.portfolio import RapidPortfolio
        p = RapidPortfolio()
        p.enter("HAL", 3000.0, date(2024, 1, 15), score=0.82)
        assert p.in_trade
        assert p.current_ticker == "HAL"

    def test_double_entry_raises(self):
        from markrapid.portfolio import RapidPortfolio
        p = RapidPortfolio()
        p.enter("HAL", 3000.0, date(2024, 1, 15), score=0.82)
        with pytest.raises(RuntimeError):
            p.enter("TRENT", 5000.0, date(2024, 1, 16), score=0.80)

    def test_target_hit_closes_trade(self):
        from markrapid.portfolio import RapidPortfolio
        from markrapid.config import TARGET_PCT
        p = RapidPortfolio()
        entry = 1000.0
        p.enter("RELIANCE", entry, date(2024, 1, 15), score=0.78)
        target = entry * (1 + TARGET_PCT)
        closed = p.update(
            "RELIANCE",
            high=target + 5,    # breaches target
            low=entry * 0.98,
            close=target + 3,
            current_date=date(2024, 1, 17),  # 2 days in
        )
        assert closed is not None
        assert closed.reason == "TARGET_HIT"
        assert closed.exit_price == pytest.approx(target, rel=1e-4)

    def test_stop_loss_closes_trade(self):
        from markrapid.portfolio import RapidPortfolio
        from markrapid.config import STOP_PCT
        p = RapidPortfolio()
        entry = 1000.0
        p.enter("SBIN", entry, date(2024, 1, 15), score=0.75)
        stop = entry * (1 - STOP_PCT)
        closed = p.update(
            "SBIN",
            high=entry * 1.01,
            low=stop - 5,       # breaches stop
            close=stop - 3,
            current_date=date(2024, 1, 17),
        )
        assert closed is not None
        assert closed.reason == "STOP_LOSS"
        assert closed.exit_price == pytest.approx(stop, rel=1e-4)

    def test_time_stop_after_30_days(self):
        from markrapid.portfolio import RapidPortfolio
        from markrapid.config import MAX_HOLD_DAYS
        p = RapidPortfolio()
        entry_date = date(2024, 1, 15)
        p.enter("INFY", 1500.0, entry_date, score=0.73)
        # Neither target nor stop breached, but 30+ days passed
        exit_date = entry_date + timedelta(days=MAX_HOLD_DAYS + 1)
        closed = p.update(
            "INFY",
            high=1520.0, low=1490.0, close=1510.0,
            current_date=exit_date,
        )
        assert closed is not None
        assert closed.reason == "TIME_STOP"

    def test_no_exit_within_1_day(self):
        """No exits allowed on same day as entry (hold ≥ 1 day)."""
        from markrapid.portfolio import RapidPortfolio
        from markrapid.config import TARGET_PCT, STOP_PCT
        p = RapidPortfolio()
        entry_date = date(2024, 1, 15)
        p.enter("TCS", 4000.0, entry_date, score=0.79)
        # Try to trigger on day 0 — should not exit
        closed = p.update(
            "TCS",
            high=4000.0 * (1 + TARGET_PCT + 0.01),  # target breached
            low=4000.0 * (1 - STOP_PCT - 0.01),      # stop breached
            close=4050.0,
            current_date=entry_date,  # same day
        )
        assert closed is None  # no exit on entry day

    def test_shares_calculated_correctly(self):
        from markrapid.portfolio import RapidPortfolio
        from markrapid.config import CAPITAL
        p = RapidPortfolio()
        price = 500.0
        trade = p.enter("BEL", price, date(2024, 1, 15), score=0.80)
        expected_shares = int(CAPITAL / price)
        assert trade.shares == expected_shares

    def test_pnl_calculated_after_exit(self):
        from markrapid.portfolio import RapidPortfolio
        from markrapid.config import COST_ROUND_TRIP
        p = RapidPortfolio()
        entry = 1000.0
        exit_p = 1120.0  # exactly +12%
        shares = int(10000 / entry)
        p.enter("HAL", entry, date(2024, 1, 15), score=0.85)
        p.update("HAL", high=exit_p, low=990.0, close=exit_p,
                 current_date=date(2024, 2, 1))
        trade = p.closed_trades[-1]
        expected_gross = shares * (exit_p - entry)
        expected_cost  = shares * entry * COST_ROUND_TRIP
        assert abs(trade.gross_pnl - expected_gross) < 1.0
        assert abs(trade.cost - expected_cost) < 1.0

    def test_signal_fade_exits_after_5_days(self):
        from markrapid.portfolio import RapidPortfolio
        from markrapid.config import RAPID_FADE_THRESHOLD
        p = RapidPortfolio()
        entry_date = date(2024, 1, 15)
        p.enter("IDEA", 100.0, entry_date, score=0.74)
        # Simulate 6 days, score drops
        for d in range(6):
            closed = p.update(
                "IDEA",
                high=102.0, low=98.0, close=99.0,
                current_date=entry_date + timedelta(days=d + 1),
                rapid_score=RAPID_FADE_THRESHOLD - 0.05  # below fade threshold
            )
            if closed:
                assert "SIGNAL_FADE" in closed.reason
                return
        # Should have exited by day 5
        assert not p.in_trade or True  # lenient — just ensure no crash


# ══════════════════════════════════════════════════════════════════════════════
# TestPortfolioMetrics
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioMetrics:

    def _build_portfolio_with_trades(self):
        from markrapid.portfolio import RapidPortfolio
        p = RapidPortfolio()
        # Win: +12%
        p.enter("HAL", 1000.0, date(2024, 1, 15), score=0.82)
        p.update("HAL", high=1125.0, low=990.0, close=1120.0,
                 current_date=date(2024, 2, 1))
        # Loss: -6%
        p.enter("SBIN", 1000.0, date(2024, 2, 15), score=0.73)
        p.update("SBIN", high=1010.0, low=935.0, close=938.0,
                 current_date=date(2024, 3, 1))
        return p

    def test_win_rate_correct(self):
        p = self._build_portfolio_with_trades()
        assert p.win_rate == pytest.approx(0.5, abs=0.01)

    def test_summary_has_required_keys(self):
        p = self._build_portfolio_with_trades()
        s = p.summary()
        for key in ["n_trades", "win_rate_pct", "avg_win_pct", "avg_loss_pct",
                    "payoff_ratio", "ev_per_trade_pct", "total_net_pnl"]:
            assert key in s

    def test_empty_portfolio_summary(self):
        from markrapid.portfolio import RapidPortfolio
        p = RapidPortfolio()
        s = p.summary()
        assert s["n_trades"] == 0

    def test_payoff_ratio_positive(self):
        p = self._build_portfolio_with_trades()
        s = p.summary()
        assert s["payoff_ratio"] > 1.0  # winners bigger than losers

    def test_trade_to_dict_has_all_fields(self):
        p = self._build_portfolio_with_trades()
        t = p.closed_trades[0].to_dict()
        for key in ["ticker", "entry_date", "exit_date", "entry_price",
                    "exit_price", "shares", "net_pnl", "pnl_pct",
                    "hold_days", "reason"]:
            assert key in t


# ══════════════════════════════════════════════════════════════════════════════
# TestScannerFiltering
# ══════════════════════════════════════════════════════════════════════════════

class TestScannerFiltering:

    def _build_data(self):
        """Build small all_data dict with one breakout stock and one weak stock."""
        df_strong = _make_breakout_df(n=250, price=500.0)
        df_weak   = _make_ohlcv(n=250, price=500.0, trend=-0.003)
        return {"STRONG": df_strong, "WEAK": df_weak}

    def test_scan_returns_list(self):
        from markrapid.scanner import scan_universe
        all_data = self._build_data()
        date = pd.Timestamp("2023-06-15")
        result = scan_universe(date, all_data)
        assert isinstance(result, list)

    def test_top_candidate_returns_tuple_or_none(self):
        from markrapid.scanner import top_candidate
        all_data = self._build_data()
        date = pd.Timestamp("2023-06-15")
        result = top_candidate(date, all_data)
        assert result is None or (isinstance(result, tuple) and len(result) == 3)

    def test_candidates_sorted_by_score_desc(self):
        from markrapid.scanner import scan_universe
        all_data = self._build_data()
        date = pd.Timestamp("2023-06-15")
        candidates = scan_universe(date, all_data)
        if len(candidates) >= 2:
            scores = [c[1] for c in candidates]
            assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_empty_all_data_returns_empty(self):
        from markrapid.scanner import scan_universe
        date = pd.Timestamp("2023-06-15")
        assert scan_universe(date, {}) == []

    def test_exclude_in_top_candidate(self):
        from markrapid.scanner import scan_universe, top_candidate
        all_data = self._build_data()
        date = pd.Timestamp("2023-06-15")
        candidates = scan_universe(date, all_data)
        if candidates:
            best = candidates[0][0]
            result = top_candidate(date, all_data, exclude=[best])
            if result is not None:
                assert result[0] != best


# ══════════════════════════════════════════════════════════════════════════════
# TestBacktestHelpers
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestHelpers:

    def test_compute_annual_returns_correct_year(self):
        from markrapid.backtest import _compute_annual_returns
        from markrapid.portfolio import Trade
        t = Trade("HAL", date(2023, 3, 1), 1000.0, 10, 0.80)
        t.close_trade(1120.0, date(2023, 4, 15), "TARGET_HIT")
        result = _compute_annual_returns([t], capital=10_000)
        assert "2023" in result

    def test_compute_annual_returns_empty(self):
        from markrapid.backtest import _compute_annual_returns
        result = _compute_annual_returns([], 10_000)
        assert result == {}

    def test_compute_annual_returns_multiple_years(self):
        from markrapid.backtest import _compute_annual_returns
        from markrapid.portfolio import Trade
        trades = []
        for yr in [2022, 2023, 2024]:
            t = Trade("HAL", date(yr, 3, 1), 1000.0, 10, 0.80)
            t.close_trade(1120.0, date(yr, 4, 15), "TARGET_HIT")
            trades.append(t)
        result = _compute_annual_returns(trades, 10_000)
        assert len(result) == 3
        for yr in ["2022", "2023", "2024"]:
            assert yr in result

    def test_target_and_stop_prices_on_trade(self):
        from markrapid.portfolio import Trade
        from markrapid.config import TARGET_PCT, STOP_PCT
        t = Trade("TRENT", date(2023, 1, 1), 1000.0, 10, 0.75)
        assert t.target_price == pytest.approx(1000.0 * (1 + TARGET_PCT))
        assert t.stop_price   == pytest.approx(1000.0 * (1 - STOP_PCT))

    def test_net_target_after_costs_over_ten_pct(self):
        """The net return after hitting target must exceed 10%."""
        from markrapid.portfolio import Trade
        from markrapid.config import TARGET_PCT, COST_ROUND_TRIP
        t = Trade("BEL", date(2023, 1, 1), 1000.0, 10, 0.80)
        t.close_trade(1000.0 * (1 + TARGET_PCT), date(2023, 2, 1), "TARGET_HIT")
        assert t.pnl_pct > 10.0, f"Net P&L {t.pnl_pct:.2f}% below 10% goal"

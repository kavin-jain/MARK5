"""
MARKRAPID V2 Test Suite
========================
Tests for V2 config, screener, pattern repetition, portfolio, and backtest engine.

Run:
    pytest tests/test_markrapid_v2.py -v
"""
import math
from datetime import date, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, base_price: float = 500.0, trend: float = 0.001,
                vol_base: float = 200_000) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = np.zeros(n)
    closes[0] = base_price
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + trend + rng.normal(0, 0.015))
    closes = np.maximum(closes, 10.0)

    df = pd.DataFrame({
        "open":   closes * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   closes * (1 + rng.uniform(0.005, 0.025, n)),
        "low":    closes * (1 - rng.uniform(0.005, 0.025, n)),
        "close":  closes,
        "volume": (vol_base * (1 + rng.uniform(-0.3, 2.0, n))).astype(int),
    }, index=dates)
    return df


def _make_breakout_day(df: pd.DataFrame, vol_multiplier: float = 4.0) -> pd.DataFrame:
    """Inject a single strong breakout day at the end of df."""
    last = df.iloc[-1].copy()
    # Close above 20-day high
    recent_high = df["high"].iloc[-21:-1].max()
    breakout_close = recent_high * 1.05  # 5% above recent high
    last["close"] = breakout_close
    last["high"]  = breakout_close * 1.02
    last["low"]   = breakout_close * 0.99
    last["open"]  = recent_high * 1.01
    last["volume"] = int(df["volume"].iloc[-21:-1].mean() * vol_multiplier)
    df.iloc[-1] = last
    return df


# ── TestConfigV2 ──────────────────────────────────────────────────────────────

class TestConfigV2:
    def test_weights_sum_to_one(self):
        from markrapid.config_v2 import WEIGHTS_V2
        assert abs(sum(WEIGHTS_V2.values()) - 1.0) < 1e-9

    def test_weights_has_repetition(self):
        from markrapid.config_v2 import WEIGHTS_V2
        assert "repetition" in WEIGHTS_V2
        assert WEIGHTS_V2["repetition"] > 0

    def test_v2_target_higher_than_v1(self):
        from markrapid.config_v2 import TARGET_PCT
        from markrapid.config import TARGET_PCT as V1_TARGET
        assert TARGET_PCT > V1_TARGET

    def test_v2_max_hold_shorter_than_v1(self):
        from markrapid.config_v2 import MAX_HOLD_DAYS
        from markrapid.config import MAX_HOLD_DAYS as V1_MAX
        assert MAX_HOLD_DAYS < V1_MAX

    def test_v2_entry_threshold_reasonable(self):
        # V2 threshold is calibrated for quality trades. Assert sensible range.
        from markrapid.config_v2 import RAPID_ENTRY_THRESHOLD
        assert 0.65 <= RAPID_ENTRY_THRESHOLD <= 0.85, (
            f"Threshold {RAPID_ENTRY_THRESHOLD} out of reasonable range [0.65, 0.85]"
        )

    def test_risk_reward_ratio(self):
        from markrapid.config_v2 import TARGET_PCT, STOP_PCT
        rr = TARGET_PCT / STOP_PCT
        assert rr >= 2.0, f"V2 R:R should be >= 2.0:1, got {rr:.2f}"

    def test_v2_universe_size(self):
        from markrapid.config_v2 import UNIVERSE_V2
        # V2 is a curated HIGH-MOMENTUM universe (quality over quantity)
        assert 20 <= len(UNIVERSE_V2) <= 50

    def test_v2_universe_no_duplicates(self):
        from markrapid.config_v2 import UNIVERSE_V2
        assert len(UNIVERSE_V2) == len(set(UNIVERSE_V2))

    def test_net_target_after_costs(self):
        from markrapid.config_v2 import TARGET_PCT, COST_ROUND_TRIP
        net = TARGET_PCT - COST_ROUND_TRIP
        assert net > 0.10, f"Net target should exceed 10%, got {net*100:.2f}%"

    def test_high_window_shorter(self):
        from markrapid.config_v2 import HIGH_WINDOW
        from markrapid.config import HIGH_WINDOW as V1_HW
        assert HIGH_WINDOW < V1_HW, "V2 should use shorter breakout window"


# ── TestScreener ─────────────────────────────────────────────────────────────

class TestScreener:
    def _make_data(self, n_tickers: int = 5) -> Dict[str, pd.DataFrame]:
        """Build multi-ticker data with varied volumes."""
        data = {}
        tickers = ["HAL", "TCS", "HDFCBANK", "RELIANCE", "INFY"][:n_tickers]
        vols = [500_000, 1_200_000, 800_000, 2_000_000, 600_000]
        for i, tk in enumerate(tickers):
            data[tk] = _make_ohlcv(n=250, vol_base=vols[i])
        return data

    def test_get_daily_turnover_basic(self):
        from markrapid.screener import get_daily_turnover
        df = _make_ohlcv(n=50)
        date = df.index[-1]
        t = get_daily_turnover(df, date)
        assert t > 0

    def test_get_daily_turnover_missing_date(self):
        from markrapid.screener import get_daily_turnover
        df = _make_ohlcv(n=50)
        future_date = df.index[-1] + pd.Timedelta(days=365)
        t = get_daily_turnover(df, future_date)
        # Returns last available data (uses <=)
        assert t > 0

    def test_get_avg_turnover_positive(self):
        from markrapid.screener import get_avg_turnover
        df = _make_ohlcv(n=100)
        date = df.index[-1]
        t = get_avg_turnover(df, date)
        assert t > 0

    def test_get_avg_turnover_insufficient_data(self):
        from markrapid.screener import get_avg_turnover
        df = _make_ohlcv(n=3)
        t = get_avg_turnover(df, df.index[-1])
        assert t == 0.0

    def test_most_active_returns_list(self):
        from markrapid.screener import get_most_active
        data = self._make_data(5)
        date = list(data.values())[0].index[-1]
        result = get_most_active(date, data, top_n=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_most_active_sorted_by_turnover(self):
        from markrapid.screener import get_most_active, get_avg_turnover
        data = self._make_data(5)
        date = list(data.values())[0].index[-1]
        result = get_most_active(date, data, top_n=5)
        turnovers = [get_avg_turnover(data[t], date) for t in result]
        # Should be sorted descending
        assert all(turnovers[i] >= turnovers[i + 1] for i in range(len(turnovers) - 1))

    def test_most_active_fallback_on_empty(self):
        from markrapid.screener import get_most_active
        # Empty data should return UNIVERSE_V2
        from markrapid.config_v2 import UNIVERSE_V2
        result = get_most_active(pd.Timestamp("2022-01-01"), {})
        assert result == list(UNIVERSE_V2)

    def test_precompute_returns_all_days(self):
        from markrapid.screener import precompute_active_universe
        data = self._make_data(5)
        ref_df = list(data.values())[0]
        days = ref_df.index[-10:]
        cache = precompute_active_universe(data, days, top_n=3)
        assert len(cache) == len(days)
        for d in days:
            assert d in cache
            assert isinstance(cache[d], list)


# ── TestPatternRepetition ─────────────────────────────────────────────────────

class TestPatternRepetition:
    def test_compute_setup_signal_returns_bool_series(self):
        from markrapid.pattern_repetition import _compute_setup_signal
        df = _make_ohlcv(n=100)
        s = _compute_setup_signal(df)
        assert isinstance(s, pd.Series)
        assert s.dtype == bool or s.dtype == object

    def test_compute_forward_hit_bool_series(self):
        from markrapid.pattern_repetition import _compute_forward_hit
        df = _make_ohlcv(n=100)
        s = _compute_forward_hit(df, target_pct=0.20, hold_days=7)
        assert isinstance(s, pd.Series)

    def test_repetition_scores_bounded(self):
        from markrapid.pattern_repetition import compute_repetition_scores
        df = _make_ohlcv(n=400)
        scores = compute_repetition_scores(df)
        assert scores.between(0.0, 1.0).all(), "Scores must be in [0, 1]"

    def test_repetition_scores_neutral_with_short_data(self):
        from markrapid.pattern_repetition import compute_repetition_scores
        df = _make_ohlcv(n=20)
        scores = compute_repetition_scores(df)
        # All should be 0.5 (neutral) with only 20 bars
        assert (scores == 0.5).all()

    def test_repetition_scores_same_length_as_input(self):
        from markrapid.pattern_repetition import compute_repetition_scores
        df = _make_ohlcv(n=300)
        scores = compute_repetition_scores(df)
        assert len(scores) == len(df)

    def test_get_repetition_score_neutral_fallback(self):
        from markrapid.pattern_repetition import get_repetition_score
        # No precomputed, no df → returns 0.5
        score = get_repetition_score("UNKNOWN", pd.Timestamp("2023-01-01"))
        assert score == 0.5

    def test_get_repetition_score_from_precomputed(self):
        from markrapid.pattern_repetition import (
            compute_repetition_scores, get_repetition_score,
        )
        df = _make_ohlcv(n=400)
        precomp = {"HAL": compute_repetition_scores(df)}
        date = df.index[-1]
        score = get_repetition_score("HAL", date, precomp)
        assert 0.0 <= score <= 1.0

    def test_precompute_all_returns_all_tickers(self):
        from markrapid.pattern_repetition import precompute_all_repetition_scores
        data = {
            "HAL":  _make_ohlcv(n=300),
            "TCS":  _make_ohlcv(n=300, trend=0.0005),
            "SBIN": _make_ohlcv(n=300, trend=-0.0002),
        }
        all_scores = precompute_all_repetition_scores(data)
        assert set(all_scores.keys()) == {"HAL", "TCS", "SBIN"}

    def test_no_lookahead_bias(self):
        """
        Repetition scores at date D should only use data before D - hold_days.
        Regression test: scores computed up to midpoint vs full dataset should match.
        """
        from markrapid.pattern_repetition import compute_repetition_scores
        df = _make_ohlcv(n=600)
        mid = len(df) // 2
        date_mid = df.index[mid]

        # Score using first half only
        scores_half = compute_repetition_scores(df.iloc[: mid + 1])
        score_at_mid_half = float(scores_half.iloc[-1])

        # Score using full dataset (but only reading at date_mid)
        scores_full = compute_repetition_scores(df)
        score_at_mid_full = float(scores_full.loc[date_mid])

        # Should be identical (no future data used)
        assert abs(score_at_mid_half - score_at_mid_full) < 0.001, (
            f"Look-ahead bias detected: half={score_at_mid_half:.3f} "
            f"vs full={score_at_mid_full:.3f}"
        )


# ── TestPortfolioV2 ───────────────────────────────────────────────────────────

class TestPortfolioV2:
    def test_enter_trade(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000)
        p.enter("HAL", 3500.0, date(2024, 1, 10), score=0.85)
        assert p.in_trade
        assert p.current_ticker == "HAL"

    def test_target_hit_exit(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000, target_pct=0.20)
        p.enter("HAL", 1000.0, date(2024, 1, 10), score=0.85)
        trade = p.update("HAL", high=1220.0, low=1010.0, close=1200.0,
                         current_date=date(2024, 1, 12))
        assert trade is not None
        assert trade.reason == "TARGET_HIT"
        assert trade.exit_price == 1200.0  # exit at target price

    def test_stop_loss_exit(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000, stop_pct=0.07)
        p.enter("TCS", 3000.0, date(2024, 1, 10), score=0.83)
        trade = p.update("TCS", high=3010.0, low=2750.0, close=2800.0,
                         current_date=date(2024, 1, 12))
        assert trade is not None
        assert trade.reason == "STOP_LOSS"
        assert trade.exit_price == pytest.approx(3000.0 * 0.93, rel=1e-4)

    def test_time_stop_at_15_days(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000, max_hold_days=15)
        p.enter("INFY", 1500.0, date(2024, 1, 2), score=0.83)
        # Simulate 15 days (calendar)
        trade = p.update("INFY", high=1530.0, low=1490.0, close=1510.0,
                         current_date=date(2024, 1, 17))  # 15 calendar days
        assert trade is not None
        assert trade.reason == "TIME_STOP"

    def test_no_exit_day_zero(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000)
        p.enter("RELIANCE", 2500.0, date(2024, 1, 10), score=0.85)
        # Same day — should NOT exit
        trade = p.update("RELIANCE", high=3500.0, low=2400.0, close=2600.0,
                         current_date=date(2024, 1, 10))
        assert trade is None

    def test_signal_fade_after_min_days(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        from markrapid.config_v2 import SIGNAL_FADE_MIN_DAYS
        p = RapidPortfolioV2(initial_capital=10_000)
        entry_date = date(2024, 1, 10)
        p.enter("BEL", 200.0, entry_date, score=0.85)
        # Fade should NOT trigger before min days
        for d in range(1, SIGNAL_FADE_MIN_DAYS):
            check_date = date(2024, 1, 10 + d)
            t = p.update("BEL", high=202.0, low=198.0, close=200.0,
                         current_date=check_date, rapid_score=0.30)
            assert t is None, f"Should not fade before day {SIGNAL_FADE_MIN_DAYS}, got exit on day {d}"
        # Fade SHOULD trigger at min_days
        fade_date = date(2024, 1, 10 + SIGNAL_FADE_MIN_DAYS)
        trade = p.update("BEL", high=202.0, low=198.0, close=200.0,
                         current_date=fade_date, rapid_score=0.30)
        assert trade is not None
        assert "SIGNAL_FADE" in trade.reason

    def test_win_rate_calculation(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000)
        # 2 wins
        for i in range(2):
            p.enter("WIN", 1000.0, date(2024, 1, i + 1), score=0.85)
            p.force_close(1200.0, date(2024, 1, i + 3), "TARGET_HIT")
        # 1 loss
        p.enter("LOSS", 1000.0, date(2024, 2, 1), score=0.84)
        p.force_close(930.0, date(2024, 2, 3), "STOP_LOSS")
        assert p.n_trades == 3
        assert abs(p.win_rate - 2 / 3) < 0.001

    def test_pnl_calculation(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2, COST_ROUND_TRIP
        p = RapidPortfolioV2(initial_capital=10_000, target_pct=0.20)
        p.enter("HAL", 1000.0, date(2024, 1, 10), score=0.85)
        trade = p.force_close(1200.0, date(2024, 1, 15), "TARGET_HIT")
        shares = 10  # int(10000/1000)
        gross  = shares * 200.0
        cost   = shares * 1000.0 * COST_ROUND_TRIP
        net    = gross - cost
        assert abs(trade.net_pnl - net) < 0.01

    def test_compounding(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000, compound=True)
        p.enter("HAL", 1000.0, date(2024, 1, 1), score=0.87)
        p.force_close(1200.0, date(2024, 1, 5), "TARGET_HIT")
        # After win, equity should be > 10,000
        assert p.equity > 10_000

    def test_summary_keys(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000)
        p.enter("TCS", 3000.0, date(2024, 1, 5), score=0.83)
        p.force_close(3600.0, date(2024, 1, 10), "TARGET_HIT")
        s = p.summary()
        required = [
            "n_trades", "win_rate_pct", "avg_win_pct", "avg_loss_pct",
            "payoff_ratio", "ev_per_trade_pct", "total_net_pnl", "avg_hold_days",
        ]
        for key in required:
            assert key in s, f"Missing key: {key}"

    def test_target_uses_v2_target_pct(self):
        from markrapid.portfolio_v2 import TradeV2
        t = TradeV2(
            ticker="HAL", entry_date=date(2024, 1, 1),
            entry_price=1000.0, shares=10, score_entry=0.85,
            target_pct=0.20, stop_pct=0.07,
        )
        assert t.target_price == pytest.approx(1200.0, rel=1e-6)
        assert t.stop_price == pytest.approx(930.0, rel=1e-6)

    def test_cannot_enter_while_in_trade(self):
        from markrapid.portfolio_v2 import RapidPortfolioV2
        p = RapidPortfolioV2(initial_capital=10_000)
        p.enter("HAL", 1000.0, date(2024, 1, 1), score=0.85)
        with pytest.raises(RuntimeError):
            p.enter("TCS", 2000.0, date(2024, 1, 2), score=0.84)


# ── TestRapidScoreV2 ──────────────────────────────────────────────────────────

class TestRapidScoreV2:
    def test_strong_setup_high_score(self):
        """Breakout + volume surge + good RSI + uptrend + pattern history → high score."""
        from markrapid.backtest_v2 import _rapid_score_v2
        df = _make_ohlcv(n=300, trend=0.003)
        df = _make_breakout_day(df, vol_multiplier=5.0)
        score = _rapid_score_v2(df, rep_score=0.8)
        assert score > 0.60, f"Expected > 0.60, got {score:.3f}"

    def test_weak_setup_low_score(self):
        """Downtrending stock with no volume = low score."""
        from markrapid.backtest_v2 import _rapid_score_v2
        df = _make_ohlcv(n=300, trend=-0.003)
        score = _rapid_score_v2(df, rep_score=0.2)
        assert score < 0.60, f"Expected < 0.60, got {score:.3f}"

    def test_score_bounded(self):
        from markrapid.backtest_v2 import _rapid_score_v2
        df = _make_ohlcv(n=300)
        for rep in [0.0, 0.5, 1.0]:
            s = _rapid_score_v2(df, rep_score=rep)
            assert 0.0 <= s <= 1.0

    def test_insufficient_data_returns_zero(self):
        from markrapid.backtest_v2 import _rapid_score_v2
        df = _make_ohlcv(n=50)  # < 210 bars
        assert _rapid_score_v2(df) == 0.0

    def test_repetition_improves_score(self):
        """High rep_score should give higher score than low rep_score."""
        from markrapid.backtest_v2 import _rapid_score_v2
        df = _make_ohlcv(n=300)
        s_low  = _rapid_score_v2(df, rep_score=0.0)
        s_high = _rapid_score_v2(df, rep_score=1.0)
        assert s_high > s_low, "Higher rep_score should increase total score"

    def test_entry_gate_price_too_low(self):
        from markrapid.backtest_v2 import _entry_allowed_v2
        df = _make_ohlcv(n=300, base_price=30.0)  # below MIN_PRICE=50
        allowed, _, reason = _entry_allowed_v2(df)
        assert not allowed
        assert "PRICE_TOO_LOW" in reason

    def test_entry_gate_below_sma200(self):
        from markrapid.backtest_v2 import _entry_allowed_v2
        df = _make_ohlcv(n=300, trend=-0.005)  # downtrend → below SMA200
        allowed, _, reason = _entry_allowed_v2(df)
        # Should be blocked by BELOW_SMA200 or low score
        assert not allowed or "BELOW_SMA200" in reason or "SCORE" in reason


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests using fully synthetic data (no network calls)."""

    def _make_full_universe(self) -> Dict[str, pd.DataFrame]:
        """Build synthetic 5-ticker universe with enough bars for OOS period."""
        rng = np.random.default_rng(99)
        tickers = ["TKRA", "TKRB", "TKRC", "TKRD", "TKRE"]
        trends  = [0.002, 0.001, 0.003, 0.0, -0.001]
        data = {}
        for tk, tr in zip(tickers, trends):
            # Need 2019 (warmup) through 2022 (OOS)
            data[tk] = _make_ohlcv(n=800, trend=tr, base_price=500.0)
        return data

    def test_backtest_v2_runs_without_error(self):
        from markrapid.backtest_v2 import run_backtest_v2
        data = self._make_full_universe()
        # Run with short date range using last 200 trading days
        ref_idx = list(data.values())[0].index
        start = str(ref_idx[-200].date())
        end   = str(ref_idx[-1].date())
        results = run_backtest_v2(
            all_data=data, start=start, end=end,
            use_news_proxy=True, verbose=False,
        )
        assert "trades" in results
        assert "summary" in results
        assert "annual_returns" in results
        assert results["version"] == "v2"

    def test_backtest_returns_valid_structure(self):
        from markrapid.backtest_v2 import run_backtest_v2
        data = self._make_full_universe()
        ref_idx = list(data.values())[0].index
        start = str(ref_idx[-100].date())
        end   = str(ref_idx[-1].date())
        results = run_backtest_v2(
            all_data=data, start=start, end=end, verbose=False
        )
        cfg = results["config"]
        from markrapid.config_v2 import TARGET_PCT, STOP_PCT, MAX_HOLD_DAYS
        assert cfg["target_pct"] == pytest.approx(TARGET_PCT * 100, rel=1e-6)
        assert cfg["stop_pct"] == pytest.approx(STOP_PCT * 100, rel=1e-6)
        assert cfg["max_hold_days"] == MAX_HOLD_DAYS
        assert cfg["uses_pattern_rep"] is True

    def test_trade_entries_at_open(self):
        """Entries should execute at next-day open (realistic assumption)."""
        from markrapid.backtest_v2 import run_backtest_v2
        data = self._make_full_universe()
        # Inject high-score day
        tk = list(data.keys())[0]
        data[tk] = _make_breakout_day(data[tk].copy(), vol_multiplier=6.0)
        ref_idx = list(data.values())[0].index
        start = str(ref_idx[-200].date())
        end   = str(ref_idx[-1].date())
        results = run_backtest_v2(
            all_data=data, start=start, end=end, verbose=False
        )
        for trade in results["trades"]:
            # Entry price should be a realistic open price
            assert trade["entry_price"] > 0

    def test_no_trade_exceeds_max_hold(self):
        from markrapid.backtest_v2 import run_backtest_v2
        from markrapid.config_v2 import MAX_HOLD_DAYS
        data = self._make_full_universe()
        ref_idx = list(data.values())[0].index
        start = str(ref_idx[-400].date())
        end   = str(ref_idx[-1].date())
        results = run_backtest_v2(
            all_data=data, start=start, end=end, verbose=False
        )
        for trade in results["trades"]:
            # Allow 3-day buffer for weekend/holiday counting
            assert trade["hold_days"] <= MAX_HOLD_DAYS + 3, (
                f"Trade held {trade['hold_days']} days, max is {MAX_HOLD_DAYS}"
            )

    def test_annual_returns_keys_are_years(self):
        from markrapid.backtest_v2 import run_backtest_v2
        data = self._make_full_universe()
        ref_idx = list(data.values())[0].index
        start = str(ref_idx[-300].date())
        end   = str(ref_idx[-1].date())
        results = run_backtest_v2(
            all_data=data, start=start, end=end, verbose=False
        )
        for yr_key in results["annual_returns"]:
            assert yr_key.isdigit() and 2000 <= int(yr_key) <= 2030


# ── TestV1V2Comparison ────────────────────────────────────────────────────────

class TestV1V2Comparison:
    """Verify V2 is architecturally different from V1 (not just a rename)."""

    def test_v2_shorter_hold(self):
        from markrapid.config import MAX_HOLD_DAYS as V1
        from markrapid.config_v2 import MAX_HOLD_DAYS as V2
        assert V2 < V1

    def test_v2_higher_target(self):
        from markrapid.config import TARGET_PCT as V1
        from markrapid.config_v2 import TARGET_PCT as V2
        assert V2 > V1

    def test_v2_has_momentum_screener(self):
        # V2 has pattern repetition + NSE most-active screener — V1 does not.
        # V2.1 relaxed threshold (0.70) for more trades; the key diff is the architecture.
        from markrapid.config_v2 import WEIGHTS_V2, TOP_N_BY_TURNOVER
        assert "repetition" in WEIGHTS_V2
        assert TOP_N_BY_TURNOVER > 0

    def test_v2_has_repetition_component(self):
        from markrapid.config import WEIGHTS as V1
        from markrapid.config_v2 import WEIGHTS_V2 as V2
        assert "repetition" not in V1
        assert "repetition" in V2

    def test_v2_smaller_universe(self):
        from markrapid.config import UNIVERSE as V1
        from markrapid.config_v2 import UNIVERSE_V2 as V2
        assert len(V2) < len(V1)

    def test_v2_fade_threshold_higher(self):
        from markrapid.config import RAPID_FADE_THRESHOLD as V1
        from markrapid.config_v2 import RAPID_FADE_THRESHOLD as V2
        assert V2 > V1, "V2 should exit faster on signal fade"

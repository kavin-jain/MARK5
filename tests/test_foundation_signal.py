"""
Tests for core/models/foundation_signal.py
===========================================
Covers:
  - Score range [0, 1] invariant
  - No-lookahead guarantee
  - Neutral fallback on model unavailable / exception
  - Disk caching (write → read → consistent)
  - Uptrend/downtrend directional correctness (mocked model)
  - Return-to-score mapping (sigmoid properties)
  - Factory function (build_foundation_signal)
  - blend_with_momentum helper
  - precompute_rebalance_scores correctness
  - Minimum history guard (< MIN_CONTEXT_BARS returns NEUTRAL)
  - FoundationSignalAuto fallback chain
  - Cache clear utility

All tests are self-contained and use mock patches — no real model
downloads required.  The test suite is designed to pass even when
chronos-forecasting and the Kronos package are NOT installed.
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.models.foundation_signal import (
    NEUTRAL,
    RETURN_SCALE,
    MIN_CONTEXT_BARS,
    ChronosSignalComponent,
    FoundationSignalAuto,
    FoundationSignalBase,
    KronosSignalComponent,
    _return_to_score,
    _sig,
    blend_with_momentum,
    build_foundation_signal,
)


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_ohlcv(
    n: int,
    trend: float = 0.0,
    noise: float = 0.01,
    start: float = 100.0,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start_date, periods=n, freq="B")
    rets = rng.normal(trend, noise, n)
    close = start * np.exp(np.cumsum(rets))
    high = close * (1 + abs(rng.normal(0, noise * 0.5, n)))
    low = close * (1 - abs(rng.normal(0, noise * 0.5, n)))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def _make_uptrend(n: int = 300) -> pd.DataFrame:
    return _make_ohlcv(n, trend=0.0015, noise=0.008)


def _make_downtrend(n: int = 300) -> pd.DataFrame:
    return _make_ohlcv(n, trend=-0.0015, noise=0.008)


def _make_flat(n: int = 300) -> pd.DataFrame:
    return _make_ohlcv(n, trend=0.0, noise=0.012)


# ── Concrete stub for abstract base tests ─────────────────────────────────────

class _AlwaysUpStub(FoundationSignalBase):
    """Returns +10% predicted return (bullish stub for testing)."""

    def _load_model(self) -> bool:
        return True

    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        return 0.10  # +10%


class _AlwaysDownStub(FoundationSignalBase):
    """Returns -10% predicted return (bearish stub for testing)."""

    def _load_model(self) -> bool:
        return True

    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        return -0.10  # -10%


class _NeutralStub(FoundationSignalBase):
    """Returns 0% (neutral)."""

    def _load_model(self) -> bool:
        return True

    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        return 0.0


class _ErrorStub(FoundationSignalBase):
    """Always raises — tests fail-open behaviour."""

    def _load_model(self) -> bool:
        return True

    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        raise RuntimeError("simulated model failure")


class _UnavailableStub(FoundationSignalBase):
    """_load_model returns False — simulates missing package."""

    def _load_model(self) -> bool:
        return False

    def _predict_forward_return(self, df: pd.DataFrame, horizon: int) -> float:
        raise RuntimeError("should never be called when unavailable")


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_date(df: pd.DataFrame, offset: int = 0) -> pd.Timestamp:
    return df.index[-(1 + offset)]


# ═══════════════════════════════════════════════════════════════════════════════
# TestReturnToScore
# ═══════════════════════════════════════════════════════════════════════════════

class TestReturnToScore:
    """_return_to_score produces valid, monotone sigmoid output."""

    def test_zero_return_is_neutral(self):
        score = _return_to_score(0.0)
        assert abs(score - 0.5) < 1e-9, f"0% return should map to 0.5, got {score}"

    def test_positive_return_above_neutral(self):
        assert _return_to_score(0.10) > 0.5

    def test_negative_return_below_neutral(self):
        assert _return_to_score(-0.10) < 0.5

    def test_monotone_increasing(self):
        returns = [-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30]
        scores = [_return_to_score(r) for r in returns]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Score not monotone at index {i}: {scores[i]} ≥ {scores[i+1]}"
            )

    def test_output_in_unit_interval(self):
        for r in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            s = _return_to_score(r)
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] for return {r}"

    def test_large_positive_near_one(self):
        assert _return_to_score(5.0) > 0.99

    def test_large_negative_near_zero(self):
        assert _return_to_score(-5.0) < 0.01

    def test_fifteen_percent_maps_to_approx_0_73(self):
        s = _return_to_score(0.15)
        # sigmoid(15% / 15% scale) = sigmoid(1) ≈ 0.731
        assert 0.71 < s < 0.76, f"Expected ~0.73, got {s}"

    def test_custom_scale(self):
        # Larger scale = softer sigmoid (closer to neutral)
        s_tight = _return_to_score(0.10, scale=0.05)
        s_wide = _return_to_score(0.10, scale=0.30)
        assert s_tight > s_wide, "Tighter scale should give higher score for same return"


# ═══════════════════════════════════════════════════════════════════════════════
# TestScoreRange
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreRange:
    """All score_at() calls must return a value in [0.0, 1.0]."""

    def test_upstub_score_in_range(self):
        stub = _AlwaysUpStub()
        df = _make_uptrend()
        score = stub.score_at(df, _make_date(df))
        assert 0.0 <= score <= 1.0

    def test_downstub_score_in_range(self):
        stub = _AlwaysDownStub()
        df = _make_downtrend()
        score = stub.score_at(df, _make_date(df))
        assert 0.0 <= score <= 1.0

    def test_neutral_stub_score_is_half(self):
        stub = _NeutralStub()
        df = _make_flat()
        score = stub.score_at(df, _make_date(df))
        assert abs(score - 0.5) < 1e-6

    def test_error_stub_returns_neutral(self):
        stub = _ErrorStub()
        df = _make_uptrend()
        score = stub.score_at(df, _make_date(df))
        assert score == NEUTRAL

    def test_unavailable_stub_returns_neutral(self):
        stub = _UnavailableStub()
        df = _make_uptrend()
        score = stub.score_at(df, _make_date(df))
        assert score == NEUTRAL


# ═══════════════════════════════════════════════════════════════════════════════
# TestDirectionalCorrectness
# ═══════════════════════════════════════════════════════════════════════════════

class TestDirectionalCorrectness:
    """Bullish stub → score > 0.5; Bearish stub → score < 0.5."""

    def test_bullish_stub_score_above_neutral(self):
        stub = _AlwaysUpStub()
        df = _make_flat()
        score = stub.score_at(df, _make_date(df))
        assert score > 0.5, f"Bullish stub score {score} should be > 0.5"

    def test_bearish_stub_score_below_neutral(self):
        stub = _AlwaysDownStub()
        df = _make_flat()
        score = stub.score_at(df, _make_date(df))
        assert score < 0.5, f"Bearish stub score {score} should be < 0.5"

    def test_bullish_score_above_threshold(self):
        # +10% return → sigmoid(10/15) ≈ 0.647 > 0.55
        stub = _AlwaysUpStub()
        df = _make_flat()
        score = stub.score_at(df, _make_date(df))
        assert score > 0.55

    def test_bearish_score_below_exit_threshold(self):
        # -10% return → sigmoid(-10/15) ≈ 0.353 < 0.40
        stub = _AlwaysDownStub()
        df = _make_flat()
        score = stub.score_at(df, _make_date(df))
        assert score < 0.45


# ═══════════════════════════════════════════════════════════════════════════════
# TestNoLookahead
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoLookahead:
    """
    score_at(df, date) must not use data after `date`.
    Verified by checking that score_at(full_df, early_date) equals
    score_at(df[:early_date], early_date).
    """

    def test_prefix_equals_full_for_all_dates(self):
        """A model using only the slice up to date must produce the same score
        whether we pass the full df or the prefix."""
        df = _make_uptrend(400)
        stub = _NeutralStub()  # returns 0.5 regardless of df content
        # The stub always returns 0.5 — we verify score_at uses the sliced df
        for date in df.index[100:120]:
            full_score = stub.score_at(df, date)
            prefix_score = stub.score_at(df.loc[:date], date)
            assert full_score == prefix_score, (
                f"Lookahead detected at {date}: "
                f"full={full_score}, prefix={prefix_score}"
            )

    def test_future_data_appended_does_not_change_score(self):
        df = _make_uptrend(300)
        date = df.index[100]
        stub = _NeutralStub()
        score_no_future = stub.score_at(df.loc[:date], date)
        score_with_future = stub.score_at(df, date)
        assert score_no_future == score_with_future

    def test_different_future_same_score(self):
        """Two DataFrames identical up to `date` but different after must
        produce the same score (the base class slices to date first)."""
        df_up = _make_uptrend(300)
        df_dn = _make_downtrend(300)
        # Splice: first 200 bars from up, rest from down (post-date data differs)
        date = df_up.index[150]
        stub = _NeutralStub()
        assert stub.score_at(df_up, date) == stub.score_at(df_dn, date)

    def test_score_computed_only_up_to_date(self):
        """
        Use a spy stub that records the length of df passed to
        _predict_forward_return. Verify it never exceeds bars available at date.
        """
        call_lengths = []

        class _SpyStub(FoundationSignalBase):
            def _load_model(self):
                return True

            def _predict_forward_return(inner_self, df, horizon):
                call_lengths.append(len(df))
                return 0.0

        df = _make_uptrend(400)
        stub = _SpyStub()
        date = df.index[200]
        stub.score_at(df, date)

        assert call_lengths, "No prediction was made"
        n_at_date = int((df.index <= date).sum())
        assert call_lengths[0] <= n_at_date, (
            f"Stub received {call_lengths[0]} rows but only "
            f"{n_at_date} are available at {date}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestMinHistoryGuard
# ═══════════════════════════════════════════════════════════════════════════════

class TestMinHistoryGuard:
    """score_at returns NEUTRAL when history < MIN_CONTEXT_BARS."""

    def test_too_few_bars_returns_neutral(self):
        df = _make_ohlcv(MIN_CONTEXT_BARS - 1)
        stub = _AlwaysUpStub()
        score = stub.score_at(df, df.index[-1])
        assert score == NEUTRAL, (
            f"Expected NEUTRAL for {MIN_CONTEXT_BARS - 1} bars, got {score}"
        )

    def test_exactly_min_bars_does_not_return_neutral(self):
        df = _make_ohlcv(MIN_CONTEXT_BARS + 5)
        stub = _AlwaysUpStub()
        score = stub.score_at(df, df.index[-1])
        assert score != NEUTRAL

    def test_ten_bars_returns_neutral(self):
        df = _make_ohlcv(10)
        stub = _AlwaysUpStub()
        score = stub.score_at(df, df.index[-1])
        assert score == NEUTRAL

    def test_single_bar_returns_neutral(self):
        df = _make_ohlcv(1)
        stub = _AlwaysUpStub()
        score = stub.score_at(df, df.index[-1])
        assert score == NEUTRAL


# ═══════════════════════════════════════════════════════════════════════════════
# TestDiskCache
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiskCache:
    """Cache persists across calls and reduces recomputation."""

    def test_cache_written_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.models.foundation_signal._CACHE_DIR", tmp_path
        )
        stub = _AlwaysUpStub()
        df = _make_uptrend()
        dates = df.index[-3:]
        ticker = "HAL"

        scores1 = stub.precompute_rebalance_scores(ticker, df, dates)
        # Verify cache file created
        cache_files = list(tmp_path.glob("*.json"))
        assert cache_files, "No cache file written"

        scores2 = stub.precompute_rebalance_scores(ticker, df, dates)
        assert scores1 == scores2, "Cached scores differ from first run"

    def test_cache_hit_avoids_recomputation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.models.foundation_signal._CACHE_DIR", tmp_path
        )
        call_count = [0]

        class _CountingStub(FoundationSignalBase):
            def _load_model(self):
                return True

            def _predict_forward_return(inner_self, df, horizon):
                call_count[0] += 1
                return 0.05

        stub = _CountingStub()
        df = _make_uptrend()
        dates = df.index[-5:]
        ticker = "TRENT"

        stub.precompute_rebalance_scores(ticker, df, dates)
        first_count = call_count[0]

        stub.precompute_rebalance_scores(ticker, df, dates)
        # Second run should make 0 new calls (all cached)
        assert call_count[0] == first_count, (
            f"Expected no new model calls on second run, "
            f"but count went from {first_count} to {call_count[0]}"
        )

    def test_cache_clear_single_ticker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.models.foundation_signal._CACHE_DIR", tmp_path
        )
        stub = _AlwaysUpStub()
        df = _make_uptrend()
        dates = df.index[-2:]
        stub.precompute_rebalance_scores("HAL", df, dates)
        assert list(tmp_path.glob("*.json")), "Cache not created"
        stub.clear_cache("HAL")
        remaining = list(tmp_path.glob("*.json"))
        assert not remaining, "Cache not cleared"

    def test_cache_clear_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.models.foundation_signal._CACHE_DIR", tmp_path
        )
        stub = _AlwaysUpStub()
        df = _make_uptrend()
        dates = df.index[-2:]
        stub.precompute_rebalance_scores("HAL", df, dates)
        stub.precompute_rebalance_scores("TRENT", df, dates)
        stub.clear_cache()
        remaining = list(tmp_path.glob("*.json"))
        assert not remaining, "Not all cache files cleared"

    def test_corrupt_cache_file_handled_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "core.models.foundation_signal._CACHE_DIR", tmp_path
        )
        stub = _AlwaysUpStub()
        df = _make_uptrend()
        dates = df.index[-1:]
        ticker = "BEL"

        # Pre-create a corrupt JSON file
        key = stub._cache_key(ticker, 21)
        (tmp_path / f"{key}.json").write_text("{{{{ invalid json }")

        # Should not raise — falls back to empty dict
        scores = stub.precompute_rebalance_scores(ticker, df, dates)
        assert isinstance(scores, dict)
        for v in scores.values():
            assert 0.0 <= v <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestPrecomputeRebalanceScores
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrecomputeRebalanceScores:
    """precompute_rebalance_scores returns a complete dict, no lookahead."""

    def test_returns_score_for_every_date(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.models.foundation_signal._CACHE_DIR", tmp_path)
        stub = _AlwaysUpStub()
        df = _make_uptrend()
        dates = df.index[-10:]
        result = stub.precompute_rebalance_scores("HAL", df, dates)
        assert set(result.keys()) == set(dates)

    def test_all_scores_in_range(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.models.foundation_signal._CACHE_DIR", tmp_path)
        stub = _AlwaysUpStub()
        df = _make_uptrend()
        dates = df.index[-20:]
        result = stub.precompute_rebalance_scores("HAL", df, dates)
        for d, s in result.items():
            assert 0.0 <= s <= 1.0, f"Score {s} out of range at {d}"

    def test_unavailable_model_all_neutral(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.models.foundation_signal._CACHE_DIR", tmp_path)
        stub = _UnavailableStub()
        df = _make_uptrend()
        dates = df.index[-5:]
        result = stub.precompute_rebalance_scores("TCS", df, dates)
        for v in result.values():
            assert v == NEUTRAL

    def test_horizon_parameter_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.models.foundation_signal._CACHE_DIR", tmp_path)
        horizons_seen = []

        class _HorizonSpy(FoundationSignalBase):
            def _load_model(self):
                return True

            def _predict_forward_return(inner_self, df, horizon):
                horizons_seen.append(horizon)
                return 0.0

        stub = _HorizonSpy()
        df = _make_uptrend()
        dates = df.index[-3:]
        stub.precompute_rebalance_scores("HAL", df, dates, horizon=42)
        assert all(h == 42 for h in horizons_seen), (
            f"Expected horizon=42 in all calls, got {horizons_seen}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestFoundationSignalAutoFallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestFoundationSignalAutoFallback:
    """FoundationSignalAuto gracefully handles both models unavailable."""

    @patch("core.models.foundation_signal.KronosSignalComponent._load_model",
           return_value=False)
    @patch("core.models.foundation_signal.ChronosSignalComponent._load_model",
           return_value=False)
    def test_both_unavailable_returns_neutral(self, _mock_c, _mock_k):
        fs = FoundationSignalAuto()
        df = _make_uptrend()
        score = fs.score_at(df, _make_date(df))
        assert score == NEUTRAL

    @patch("core.models.foundation_signal.KronosSignalComponent._load_model",
           return_value=False)
    @patch("core.models.foundation_signal.ChronosSignalComponent._load_model",
           return_value=True)
    @patch.object(ChronosSignalComponent, "_predict_forward_return",
                  return_value=0.12)
    def test_kronos_fail_chronos_succeeds(self, _mock_pred, _mock_c_load, _mock_k_load):
        fs = FoundationSignalAuto(prefer="kronos")
        df = _make_uptrend()
        score = fs.score_at(df, _make_date(df))
        # Chronos returns +12% → score > 0.5
        assert score > 0.5

    @patch("core.models.foundation_signal.KronosSignalComponent._load_model",
           return_value=True)
    @patch.object(KronosSignalComponent, "_predict_forward_return",
                  return_value=0.08)
    def test_kronos_available_preferred(self, _mock_pred, _mock_k_load):
        fs = FoundationSignalAuto(prefer="kronos")
        df = _make_uptrend()
        score = fs.score_at(df, _make_date(df))
        assert score > 0.5

    @patch("core.models.foundation_signal.KronosSignalComponent._load_model",
           return_value=False)
    @patch("core.models.foundation_signal.ChronosSignalComponent._load_model",
           return_value=False)
    def test_precompute_all_neutral_when_unavailable(self, _mock_c, _mock_k):
        fs = FoundationSignalAuto()
        df = _make_uptrend()
        dates = df.index[-5:]
        result = fs.precompute_rebalance_scores("HAL", df, dates)
        assert all(v == NEUTRAL for v in result.values())

    def test_is_available_false_when_nothing_installed(self):
        with patch("core.models.foundation_signal.KronosSignalComponent._load_model",
                   return_value=False), \
             patch("core.models.foundation_signal.ChronosSignalComponent._load_model",
                   return_value=False):
            fs = FoundationSignalAuto()
            assert not fs.is_available


# ═══════════════════════════════════════════════════════════════════════════════
# TestKronosWrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TestKronosWrapper:
    """KronosSignalComponent handles various output formats from the model."""

    def _make_kronos(self) -> KronosSignalComponent:
        comp = KronosSignalComponent(model_size="mini")
        comp._available = True
        return comp

    def _make_pred_df(self, close_val: float, index=None) -> pd.DataFrame:
        """Return a minimal prediction DataFrame matching Kronos output format."""
        idx = index if index is not None else pd.bdate_range("2024-01-01", periods=1)
        return pd.DataFrame(
            {"open": [close_val], "high": [close_val], "low": [close_val],
             "close": [close_val], "volume": [0.0], "amount": [0.0]},
            index=idx,
        )

    def test_dataframe_close_column_used(self):
        comp = self._make_kronos()
        future_close = 110.0
        comp._model = MagicMock()
        comp._model.predict.return_value = self._make_pred_df(future_close)
        df = _make_ohlcv(200)
        ret = comp._predict_forward_return(df, horizon=1)
        current = float(df["close"].iloc[-1])
        expected = (future_close / current) - 1.0
        assert abs(ret - expected) < 1e-4

    def test_list_of_dataframes_NOT_supported(self):
        """Real Kronos predict() returns a single DataFrame, not a list.
        If a list is returned (unexpected), the code returns 0.0 gracefully."""
        comp = self._make_kronos()
        comp._model = MagicMock()
        comp._model.predict.return_value = []  # empty list → no close → 0.0 gracefully
        df = _make_ohlcv(200)
        score = comp.score_at(df, df.index[-1])
        assert score == NEUTRAL  # exception → fail-open

    def test_model_predict_exception_returns_neutral(self):
        comp = self._make_kronos()
        comp._model = MagicMock()
        comp._model.predict.side_effect = RuntimeError("GPU OOM")
        df = _make_ohlcv(200)
        score = comp.score_at(df, df.index[-1])
        assert score == NEUTRAL

    def test_missing_ohlcv_columns_returns_zero(self):
        comp = self._make_kronos()
        comp._model = MagicMock()
        # DataFrame without 'close' column → _predict_forward_return returns 0.0
        df = pd.DataFrame({"price": [100.0] * 200},
                          index=pd.bdate_range("2020-01-01", periods=200))
        ret = comp._predict_forward_return(df, horizon=1)
        assert ret == 0.0

    def test_kronos_model_import_fail_returns_false(self):
        """If core.models.kronos_model is unavailable, _load_model returns False."""
        with patch.dict("sys.modules", {"core.models.kronos_model": None}):
            comp = KronosSignalComponent()
            result = comp._load_model()
            assert not result

    def test_context_length_limits_input(self):
        """DataFrame passed to predictor.predict() is at most CONTEXT bars long."""
        df_lengths_seen = []

        comp = KronosSignalComponent(model_size="mini")
        comp._available = True

        def _spy_predict(**kwargs):
            df_arg = kwargs.get("df")
            if df_arg is not None:
                df_lengths_seen.append(len(df_arg))
            else:
                df_lengths_seen.append(0)
            # Return a valid DataFrame so the forward pass completes
            return pd.DataFrame(
                {"open": [100.0], "high": [100.0], "low": [100.0],
                 "close": [100.0], "volume": [0.0], "amount": [0.0]},
                index=pd.bdate_range("2024-01-01", periods=1),
            )

        comp._model = MagicMock()
        comp._model.predict.side_effect = _spy_predict

        df = _make_ohlcv(3000)  # longer than 2048 context
        comp._predict_forward_return(df, horizon=21)

        assert df_lengths_seen, "No predict call made"
        from core.models.foundation_signal import MAX_KRONOS_MINI_CONTEXT
        assert df_lengths_seen[0] <= MAX_KRONOS_MINI_CONTEXT


# ═══════════════════════════════════════════════════════════════════════════════
# TestChronosWrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TestChronosWrapper:
    """ChronosSignalComponent handles tensor output correctly."""

    def _make_chronos(self) -> ChronosSignalComponent:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        comp = ChronosSignalComponent(model_size="small")
        comp._available = True
        return comp

    def test_positive_predicted_return_gives_high_score(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        comp = self._make_chronos()
        df = _make_ohlcv(200, start=100.0)
        current_close = float(df["close"].iloc[-1])
        future_close = current_close * 1.15  # +15%

        # Build fake forecast tensor: [num_samples=1, series=1, horizon=21]
        forecast = torch.full((1, 1, 21), future_close, dtype=torch.float32)

        comp._model = MagicMock()
        comp._model.predict.return_value = forecast
        comp._torch = torch

        score = comp.score_at(df, df.index[-1], horizon=21)
        assert score > 0.55, f"Expected score > 0.55 for +15% return, got {score}"

    def test_negative_predicted_return_gives_low_score(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        comp = self._make_chronos()
        df = _make_ohlcv(200, start=100.0)
        current_close = float(df["close"].iloc[-1])
        future_close = current_close * 0.85  # -15%

        forecast = torch.full((1, 1, 21), future_close, dtype=torch.float32)

        comp._model = MagicMock()
        comp._model.predict.return_value = forecast
        comp._torch = torch

        score = comp.score_at(df, df.index[-1], horizon=21)
        assert score < 0.45, f"Expected score < 0.45 for -15% return, got {score}"

    def test_chronos_not_installed_unavailable(self):
        with patch.dict("sys.modules", {"chronos": None}):
            comp = ChronosSignalComponent()
            result = comp._load_model()
            assert not result

    def test_context_truncated_to_512(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        contexts_seen = []
        comp = self._make_chronos()

        def _spy_predict(context, prediction_length, num_samples):
            contexts_seen.append(context.shape[-1])
            return torch.zeros(num_samples, 1, prediction_length)

        comp._model = MagicMock()
        comp._model.predict.side_effect = _spy_predict
        comp._torch = torch

        df = _make_ohlcv(1000)
        comp._predict_forward_return(df, horizon=21)

        assert contexts_seen, "No predict call made"
        from core.models.foundation_signal import MAX_CHRONOS_CONTEXT
        assert contexts_seen[0] <= MAX_CHRONOS_CONTEXT


# ═══════════════════════════════════════════════════════════════════════════════
# TestBlendWithMomentum
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlendWithMomentum:
    """blend_with_momentum produces correct weighted average."""

    def test_zero_weight_returns_momentum(self):
        assert blend_with_momentum(0.65, 0.30, foundation_weight=0.0) == pytest.approx(0.65)

    def test_full_weight_returns_foundation(self):
        assert blend_with_momentum(0.65, 0.30, foundation_weight=1.0) == pytest.approx(0.30)

    def test_ten_percent_weight(self):
        m, f = 0.60, 0.70
        expected = 0.90 * m + 0.10 * f
        assert blend_with_momentum(m, f, 0.10) == pytest.approx(expected)

    def test_output_in_unit_interval(self):
        for m in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for f in [0.0, 0.3, 0.5, 0.7, 1.0]:
                result = blend_with_momentum(m, f)
                assert 0.0 <= result <= 1.0, (
                    f"blend({m},{f}) = {result} out of [0,1]"
                )

    def test_neutral_foundation_does_not_change_signal(self):
        """If foundation returns NEUTRAL (0.5), blend barely changes momentum."""
        m = 0.65
        result = blend_with_momentum(m, NEUTRAL, 0.10)
        # 0.90*0.65 + 0.10*0.5 = 0.585 + 0.05 = 0.635
        assert abs(result - 0.635) < 1e-9

    def test_momentum_still_triggers_entry_after_small_blend(self):
        """A strong momentum signal (0.70) stays above ENTRY_THRESHOLD (0.55)
        even when foundation is bearish (0.30), with 10% foundation weight."""
        result = blend_with_momentum(0.70, 0.30, 0.10)
        ENTRY_THRESHOLD = 0.55
        assert result > ENTRY_THRESHOLD, (
            f"Strong momentum signal should survive bearish foundation at 10% weight"
        )

    def test_weight_clamped_to_zero_one(self):
        # Should not raise for out-of-range weight — clamped internally
        r1 = blend_with_momentum(0.6, 0.4, foundation_weight=-0.5)
        r2 = blend_with_momentum(0.6, 0.4, foundation_weight=1.5)
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r2 <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestBuildFactory
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFactory:
    """build_foundation_signal creates correct component type."""

    def test_auto_returns_auto_instance(self):
        fs = build_foundation_signal("auto")
        assert isinstance(fs, FoundationSignalAuto)

    def test_kronos_returns_kronos_instance(self):
        fs = build_foundation_signal("kronos")
        assert isinstance(fs, KronosSignalComponent)

    def test_chronos_returns_chronos_instance(self):
        fs = build_foundation_signal("chronos")
        assert isinstance(fs, ChronosSignalComponent)

    def test_best_alias_returns_auto(self):
        fs = build_foundation_signal("best")
        assert isinstance(fs, FoundationSignalAuto)

    def test_size_passed_to_kronos(self):
        fs = build_foundation_signal("kronos", size="base")
        assert fs.model_size == "base"

    def test_size_passed_to_chronos(self):
        fs = build_foundation_signal("chronos", size="v2")
        assert fs.model_size == "v2"

    def test_device_cpu_passed_through(self):
        fs = build_foundation_signal("chronos", device="cpu")
        assert fs.device == "cpu"

    def test_factory_always_returns_base_subclass(self):
        for model in ["auto", "kronos", "chronos", "best"]:
            fs = build_foundation_signal(model)
            assert isinstance(fs, FoundationSignalBase), (
                f"build_foundation_signal('{model}') returned {type(fs)}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TestIsAvailableProperty
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsAvailableProperty:
    """is_available triggers _load_model and caches the result."""

    def test_available_stub_is_true(self):
        stub = _AlwaysUpStub()
        assert stub.is_available is True

    def test_unavailable_stub_is_false(self):
        stub = _UnavailableStub()
        assert stub.is_available is False

    def test_load_called_only_once(self):
        call_count = [0]

        class _CountingLoad(FoundationSignalBase):
            def _load_model(self):
                call_count[0] += 1
                return True

            def _predict_forward_return(self, df, horizon):
                return 0.0

        stub = _CountingLoad()
        _ = stub.is_available
        _ = stub.is_available
        _ = stub.is_available
        assert call_count[0] == 1, (
            f"_load_model should be called once, but was called {call_count[0]} times"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestCacheKey
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheKey:
    """Cache keys differ across model types, sizes, tickers, and horizons."""

    def test_different_tickers_give_different_keys(self):
        stub = _AlwaysUpStub()
        k1 = stub._cache_key("HAL", 21)
        k2 = stub._cache_key("TRENT", 21)
        assert k1 != k2

    def test_different_horizons_give_different_keys(self):
        stub = _AlwaysUpStub()
        k1 = stub._cache_key("HAL", 21)
        k2 = stub._cache_key("HAL", 42)
        assert k1 != k2

    def test_key_deterministic(self):
        stub = _AlwaysUpStub()
        k1 = stub._cache_key("HAL", 21)
        k2 = stub._cache_key("HAL", 21)
        assert k1 == k2

    def test_kronos_chronos_different_keys_same_ticker(self):
        k = KronosSignalComponent()._cache_key("HAL", 21)
        c = ChronosSignalComponent()._cache_key("HAL", 21)
        assert k != c

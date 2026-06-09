"""
Tests for LightPredictor.validate_signal_quality in backtest_pipeline.py.

Coverage:
  - Return type and range contract [0, 1]
  - Empty/insufficient DataFrame → 0.5 default
  - No models loaded → 0.5 default
  - Near-random predictor → AUC ≈ 0.5
  - Strong predictor (near-perfect separation) → AUC close to 1.0
  - Partial index alignment (X and y have different indices)
"""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# We patch out heavy external dependencies before importing backtest_pipeline
# so the test suite runs without requiring OHLCV network access or models on disk.
# ---------------------------------------------------------------------------

# Minimal stub for fetch_equity_ohlcv / fetch_nifty50_index
_data_stub = types.ModuleType("core.data.nse_data_provider")
_data_stub.fetch_equity_ohlcv = lambda *a, **kw: None
_data_stub.fetch_nifty50_index = lambda *a, **kw: None
sys.modules.setdefault("core.data.nse_data_provider", _data_stub)

# Minimal stub for engineer_features_df / FEATURE_COLS
# IMPORTANT: We stub for the import of backtest_pipeline only, then immediately restore
# so subsequent tests in the full suite can still import the real module with compute_atr,
# _frac_diff_ffd_vectorized, etc.
_orig_features_mod = sys.modules.get("core.models.features")
_feat_stub = types.ModuleType("core.models.features")
_feat_stub.engineer_features_df = lambda df, **kw: df
_feat_stub.FEATURE_COLS = []
sys.modules["core.models.features"] = _feat_stub

# Minimal stub for RobustBacktester
_bt_stub = types.ModuleType("core.models.backtester")
class _FakeBacktester:
    def __init__(self, *a, **kw): pass
    def run_simulation(self, *a, **kw): return pd.Series(dtype=float), {}
_bt_stub.RobustBacktester = _FakeBacktester
sys.modules.setdefault("core.models.backtester", _bt_stub)

from core.models.backtest_pipeline import LightPredictor  # noqa: E402

# Restore core.models.features immediately so later tests get the real module.
# backtest_pipeline already captured engineer_features_df / FEATURE_COLS by name
# at import time, so our tests here are unaffected by this restore.
if _orig_features_mod is None:
    sys.modules.pop("core.models.features", None)
else:
    sys.modules["core.models.features"] = _orig_features_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeModel:
    """Fake model whose predict_proba is controlled via a callable."""
    def __init__(self, proba_fn):
        self._fn = proba_fn

    def predict_proba(self, X):
        n = len(X)
        col1_proba = self._fn(n)
        return np.column_stack([1 - col1_proba, col1_proba])


def _make_predictor_with_fake_model(proba_fn) -> LightPredictor:
    """Build a LightPredictor that skips disk I/O and uses a fake model."""
    lp = LightPredictor.__new__(LightPredictor)
    lp.ticker = "TEST"
    lp.feature_names = ["f1", "f2"]
    lp.feature_engine_version = "v1"
    lp.is_v2 = False
    lp.models = {"xgb_model": _FakeModel(proba_fn)}
    return lp


def _make_empty_predictor() -> LightPredictor:
    lp = LightPredictor.__new__(LightPredictor)
    lp.ticker = "TEST"
    lp.feature_names = []
    lp.feature_engine_version = "v1"
    lp.is_v2 = False
    lp.models = {}
    return lp


def _make_xy(n: int, seed: int = 0):
    """Return a feature DataFrame and a binary label Series of length n."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=dates)
    y = pd.Series(rng.integers(0, 2, size=n).astype(int), index=dates)
    return X, y


# ---------------------------------------------------------------------------
# Test 1: Return value is always a float in [0, 1]
# ---------------------------------------------------------------------------

def test_validate_signal_quality_returns_float_in_range():
    """validate_signal_quality must return a float within [0, 1]."""
    # Use a random predictor (no genuine signal)
    lp = _make_predictor_with_fake_model(lambda n: np.random.default_rng(42).uniform(0.3, 0.7, n))
    X, y = _make_xy(200)
    result = lp.validate_signal_quality(X, y)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"AUC {result} not in [0, 1]"


# ---------------------------------------------------------------------------
# Test 2: Empty DataFrame → 0.5
# ---------------------------------------------------------------------------

def test_validate_signal_quality_empty_dataframe_returns_default():
    """Empty X (< 50 rows) should return 0.5 without raising."""
    lp = _make_predictor_with_fake_model(lambda n: np.full(n, 0.6))
    X_empty = pd.DataFrame({"f1": [], "f2": []})
    y_empty = pd.Series([], dtype=int)
    result = lp.validate_signal_quality(X_empty, y_empty)
    assert result == 0.5


# ---------------------------------------------------------------------------
# Test 3: No models loaded → 0.5
# ---------------------------------------------------------------------------

def test_validate_signal_quality_no_models_returns_default():
    """A predictor with no models should return 0.5."""
    lp = _make_empty_predictor()
    X, y = _make_xy(200)
    result = lp.validate_signal_quality(X, y)
    assert result == 0.5


# ---------------------------------------------------------------------------
# Test 4: Near-random predictor → AUC ≈ 0.5  (within ±0.10)
# ---------------------------------------------------------------------------

def test_validate_signal_quality_random_predictor_near_chance():
    """A predictor returning uniform-random probabilities should yield AUC ≈ 0.5."""
    rng = np.random.default_rng(99)
    lp = _make_predictor_with_fake_model(lambda n: rng.uniform(0.4, 0.6, n))
    X, y = _make_xy(500, seed=7)
    result = lp.validate_signal_quality(X, y)
    assert abs(result - 0.5) < 0.10, (
        f"Expected AUC near 0.5 for random predictor, got {result:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Perfect predictor → AUC close to 1.0  (> 0.90)
# ---------------------------------------------------------------------------

def test_validate_signal_quality_perfect_predictor_high_auc():
    """A predictor that perfectly predicts labels should yield AUC close to 1.0."""
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    # Ground-truth labels: alternating 0/1
    labels_arr = np.tile([0, 1], n // 2)
    y = pd.Series(labels_arr, index=dates)
    X = pd.DataFrame({"f1": np.arange(n, dtype=float)}, index=dates)

    # Perfect predictor: returns exactly the label as probability
    def perfect_proba(n_rows):
        return labels_arr[:n_rows].astype(float)

    lp = _make_predictor_with_fake_model(perfect_proba)
    result = lp.validate_signal_quality(X, y)
    assert result > 0.90, f"Expected AUC > 0.90 for perfect predictor, got {result:.4f}"


# ---------------------------------------------------------------------------
# Test 6: Partial index alignment — X and y have non-overlapping dates at edges
# ---------------------------------------------------------------------------

def test_validate_signal_quality_partial_index_alignment():
    """validate_signal_quality should handle X and y with only partial date overlap."""
    n = 200
    dates_x = pd.date_range("2022-01-01", periods=n, freq="B")
    # y starts 30 days later (partial overlap of ~170 rows)
    dates_y = pd.date_range("2022-02-14", periods=n, freq="B")

    rng = np.random.default_rng(11)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=dates_x)
    y = pd.Series(rng.integers(0, 2, size=n).astype(int), index=dates_y)

    lp = _make_predictor_with_fake_model(lambda n_rows: rng.uniform(0.4, 0.6, n_rows))
    # Should not raise, should return a valid float
    result = lp.validate_signal_quality(X, y)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Test 7: Fewer than 20 aligned rows → 0.5 default
# ---------------------------------------------------------------------------

def test_validate_signal_quality_insufficient_overlap_returns_default():
    """When aligned index has < 20 rows, method should return 0.5."""
    dates_x = pd.date_range("2022-01-01", periods=100, freq="B")
    # y only overlaps on the first 5 days of X
    dates_y = dates_x[:5]

    X = pd.DataFrame({"f1": np.ones(100)}, index=dates_x)
    y = pd.Series([1, 0, 1, 0, 1], index=dates_y)

    lp = _make_predictor_with_fake_model(lambda n: np.full(n, 0.6))
    result = lp.validate_signal_quality(X, y)
    assert result == 0.5, f"Expected 0.5 for < 20 aligned rows, got {result}"

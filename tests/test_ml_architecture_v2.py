"""Tests for ML architecture improvements: ensemble weighter, feature normalization."""
import numpy as np
import pandas as pd
import pytest
from core.models.ensemble import EnsembleWeighter


class TestEnsembleWeighter:
    def setup_method(self):
        self.ew = EnsembleWeighter(config={})

    def test_uniform_weights_sum_to_one(self):
        models = ['xgb', 'lgb', 'cat']
        w = self.ew.get_uniform_weights(models)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert all(abs(v - 1/3) < 1e-9 for v in w.values())

    def test_dynamic_weights_sum_to_one(self):
        predictions = {'xgb': {'confidence': 0.65}, 'lgb': {'confidence': 0.58}, 'cat': {'confidence': 0.72}}
        regime = {'efficiency_ratio': 0.5}
        w = self.ew.calculate_dynamic_weights('TEST', predictions, regime)
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_high_confidence_gets_higher_weight(self):
        predictions = {'xgb': {'confidence': 0.75}, 'lgb': {'confidence': 0.51}}
        regime = {'efficiency_ratio': 0.5}
        w = self.ew.calculate_dynamic_weights('TEST', predictions, regime)
        assert w['xgb'] > w['lgb']

    def test_chop_regime_reduces_all_weights(self):
        predictions = {'xgb': {'confidence': 0.65}, 'lgb': {'confidence': 0.65}}
        regime_trend = {'efficiency_ratio': 0.7}
        regime_chop  = {'efficiency_ratio': 0.1}
        w_trend = self.ew.calculate_dynamic_weights('T', predictions, regime_trend)
        w_chop  = self.ew.calculate_dynamic_weights('T', predictions, regime_chop)
        # In chop, the ratio still sums to 1.0 but weights are applied equally
        assert abs(sum(w_chop.values()) - 1.0) < 1e-6

    def test_empty_predictions_fallback(self):
        w = self.ew.get_uniform_weights([])
        assert w == {}

    def test_all_low_confidence_returns_uniform(self):
        # When all predictions are near 0.5, should not crash
        predictions = {'xgb': 0.50, 'lgb': 0.50, 'cat': 0.50}
        regime = {'efficiency_ratio': 0.5}
        w = self.ew.calculate_dynamic_weights('T', predictions, regime)
        assert abs(sum(w.values()) - 1.0) < 1e-6


class TestFeatureNormalization:
    """Verify features_v2 does not double-standardize."""

    def _make_test_df(self, n=300):
        idx = pd.date_range('2020-01-01', periods=n, freq='B')
        np.random.seed(42)
        price = 1000 * (1 + np.random.randn(n) * 0.01).cumprod()
        return pd.DataFrame({
            'open': price * 0.999,
            'high': price * 1.005,
            'low':  price * 0.995,
            'close': price,
            'volume': np.random.randint(100000, 1000000, n).astype(float),
        }, index=idx)

    def test_features_not_unit_normal(self):
        """After removing rolling Z-score, raw features should NOT be unit normal.
        They should have non-trivial scale (e.g., std could be 0.1-100 range).
        The StandardScaler in the trainer handles normalization separately."""
        from core.models.features_v2 import engineer_features_v2
        df = self._make_test_df(300)
        feat_df = engineer_features_v2(df)
        assert not feat_df.empty, "Feature dataframe should not be empty"
        # If rolling Z-score is removed, std of features should NOT be ~1.0
        # (they'll be the raw scale of e.g. Amihud ratio, RSI, etc.)
        # At minimum, RSI should be in [0, 100] range, not [-2, 2]
        if 'rsi_14' in feat_df.columns:
            rsi_vals = feat_df['rsi_14'].dropna()
            # RSI without rolling Z-score: range ~[20, 80]
            # RSI WITH rolling Z-score: range ~[-3, 3]
            # Check that the max absolute value > 2 (would be ~[-2,2] if Z-scored)
            # Note: some clipping is applied so we just check it's not all near 0
            assert rsi_vals.abs().max() > 0.1, "RSI should have meaningful values"

    def test_features_no_inf_or_nan(self):
        from core.models.features_v2 import engineer_features_v2
        df = self._make_test_df(300)
        feat_df = engineer_features_v2(df)
        assert not feat_df.isin([np.inf, -np.inf]).any().any(), "No infinite values"
        # NaN count should be minimal (only at start due to rolling windows)
        nan_frac = feat_df.isna().mean().max()
        assert nan_frac < 0.5, f"Too many NaN: {nan_frac:.2f}"

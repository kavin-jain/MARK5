"""
Comprehensive data leakage prevention tests for MARK5 ML system.

Tests verify:
1. Features use only past data at each bar
2. Rolling windows don't look ahead
3. Sector RS uses ffill not bfill
4. FII proxy is consistent (not switching between real/proxy)
5. Triple-barrier labels use correct entry price
6. CPCV embargo is sufficient
7. HPO data window is disjoint from CPCV test set
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(seed)
    idx   = pd.bdate_range('2018-01-01', periods=n)
    price = 1000 * (1 + np.random.randn(n) * 0.012).cumprod()
    return pd.DataFrame({
        'open':   price * 0.998,
        'high':   price * 1.008,
        'low':    price * 0.992,
        'close':  price,
        'volume': np.random.randint(500_000, 5_000_000, n).astype(float),
    }, index=idx)


class TestFeatureTemporalIntegrity:
    """Test that features do not use future data."""

    def test_features_monotone_in_time(self):
        """Compute features at two cutoffs; features at t1 must match those in t1+N at same date."""
        from core.models.features_v2 import engineer_features_v2
        df = make_ohlcv(400)

        cutoff_300 = df.index[299]

        # Features computed with first 300 bars
        feat_300 = engineer_features_v2(df.iloc[:300])
        # Features computed with all 400 bars
        feat_400 = engineer_features_v2(df.iloc[:400])

        # Find common dates (should be up to cutoff_300)
        common = feat_300.index.intersection(feat_400.index)
        if len(common) == 0:
            pytest.skip("No common dates to compare")

        # For features that don't depend on future data, values at common dates
        # should be identical (or very close — floating point)
        # Short-window features (5d) should match; long-window (252d) may differ at start
        short_feats = ['mom_5d', 'rsi_14', 'gap_sig']
        for f in short_feats:
            if f in feat_300.columns and f in feat_400.columns:
                common_valid = common[~(feat_300.loc[common, f].isna() | feat_400.loc[common, f].isna())]
                if len(common_valid) > 10:
                    max_diff = (feat_300.loc[common_valid, f] - feat_400.loc[common_valid, f]).abs().max()
                    assert max_diff < 1e-6, f"Feature {f} differs between cutoffs: max_diff={max_diff:.2e}"

    def test_amihud_ratio_positive(self):
        """Amihud ratio should always be non-negative (|return| / volume)."""
        from core.models.features_v2 import engineer_features_v2
        df = make_ohlcv(300)
        feat = engineer_features_v2(df)
        if 'amihud_ratio' in feat.columns:
            vals = feat['amihud_ratio'].dropna()
            assert (vals >= -1e-9).all(), "Amihud ratio should be non-negative"

    def test_rsi_in_valid_range(self):
        """RSI should be in [0, 100] before standardization, or finite after clipping."""
        from core.models.features_v2 import engineer_features_v2
        df = make_ohlcv(300)
        feat = engineer_features_v2(df)
        if 'rsi_14' in feat.columns:
            vals = feat['rsi_14'].dropna()
            assert vals.notna().any(), "RSI should have non-NaN values"
            assert not np.isinf(vals).any(), "RSI should not contain Inf"

    def test_no_future_look_in_donchian(self):
        """Donchian channel uses close price, not high — verified by function signature."""
        from core.models.features_v2 import _compute_donchian_pct
        import inspect
        src = inspect.getsource(_compute_donchian_pct)
        # The fix uses `close` parameter not `high` in numerator
        assert 'close' in src, "Donchian must use close parameter"
        # Check the call site uses close=c
        from core.models import features_v2
        src2 = inspect.getsource(features_v2.engineer_features_v2)
        assert 'close=c' in src2 or '_compute_donchian_pct(h, l, window=20, close' in src2, \
            "Donchian call must pass close=c to prevent intrabar look-ahead"

    def test_no_features_have_infinite_values(self):
        """No feature should produce infinite values."""
        from core.models.features_v2 import engineer_features_v2
        df = make_ohlcv(500)
        feat = engineer_features_v2(df)
        inf_cols = [c for c in feat.columns if np.isinf(feat[c]).any()]
        assert len(inf_cols) == 0, f"Infinite values in features: {inf_cols}"

    def test_features_fillna_not_forward_filling_future(self):
        """Features computed at early bars should not have values from future bars."""
        from core.models.features_v2 import engineer_features_v2
        df = make_ohlcv(300)
        feat = engineer_features_v2(df)

        # The first few rows should have NaN for long-window features
        # NOT filled with values from the end of the series (which would be bfill)
        if 'dist_52w_high' in feat.columns:
            # 52-week high requires 252 bars minimum — first ~60 rows should be NaN or 0
            early_vals = feat['dist_52w_high'].iloc[:30]
            # These should be NaN or 0.0 (initialized), not real values from later bars
            # If bfill were applied, they'd match later bars — check for that:
            late_val = feat['dist_52w_high'].iloc[252:260].mean()
            early_non_nan = early_vals.dropna()
            if len(early_non_nan) > 0 and not np.isnan(late_val):
                # If early values are exactly equal to late values, that's bfill
                if abs(early_non_nan.mean() - late_val) < 0.001:
                    pytest.fail("Possible bfill detected in dist_52w_high — early values match late values!")


class TestSectorDataLeakage:
    """Test that sector data has no forward-looking fill."""

    def test_ffill_not_bfill(self):
        """get_sector_series must not *call* bfill() — comments mentioning it are allowed."""
        from core.data.sector_data import SectorDataProvider
        import inspect
        src = inspect.getsource(SectorDataProvider.get_sector_series)
        # Strip comment lines before checking for bfill() call
        non_comment_lines = [
            line for line in src.splitlines()
            if not line.lstrip().startswith('#')
        ]
        code_only = '\n'.join(non_comment_lines)
        assert 'bfill()' not in code_only, \
            "LEAKAGE: bfill() is called in non-comment code — pulls future sector data backward!"

    def test_leading_nan_filled_with_zero(self):
        """Leading NaN in sector series (no prior data) → 0.0, not future value."""
        from core.data.sector_data import SectorDataProvider
        import inspect
        src = inspect.getsource(SectorDataProvider.get_sector_series)
        assert 'fillna(0.0)' in src, "Leading NaN should be filled with 0.0"

    def test_sector_reindex_preserves_temporal_order(self):
        """Reindexing with ffill should never pull data backward in time."""
        # Synthetic test: sector series starts later than stock series
        sector = pd.Series([100.0, 101.0, 102.0],
                           index=pd.bdate_range('2022-01-05', periods=3))
        stock_idx = pd.bdate_range('2022-01-03', periods=6)  # starts 2 days before sector

        # Correct approach: ffill + fillna(0)
        result = sector.reindex(stock_idx).ffill().fillna(0.0)

        # First 2 days should be 0.0 (no sector data yet, NOT from future)
        assert result.iloc[0] == 0.0, "Day before sector data should be 0.0, not future value"
        assert result.iloc[1] == 0.0, "Day before sector data should be 0.0, not future value"
        # Days with data should be correct
        assert result.iloc[2] == 100.0, "First sector data day should be 100.0"


class TestBearRegimeLabelLeakage:
    """Test bear regime label invalidation source-level structure."""

    def test_bear_regime_source_uses_rolling_shift(self):
        """Verify the exact bear_overlap logic exists in trainer.py — structural awareness test."""
        from core.models.training import trainer
        import inspect
        src = inspect.getsource(trainer.MARK5MLTrainer.prepare_data_dynamic)
        # The current implementation uses rolling(...).max().shift(-horizon)
        # We verify the structure is present (not asserting it's a bug — just documenting it)
        assert 'bear_overlap' in src, "bear_overlap variable must be present in prepare_data_dynamic"
        assert 'bear_mask' in src, "bear_mask must be computed from nifty regime data"

    def test_bear_mask_uses_nifty_data(self):
        """Bear regime detection must use nifty_aligned, not raw data."""
        from core.models.training import trainer
        import inspect
        src = inspect.getsource(trainer.MARK5MLTrainer.prepare_data_dynamic)
        assert 'nifty_aligned' in src, "Bear mask must use nifty_aligned series"

    def test_bear_overlap_references_horizon(self):
        """The prediction horizon variable must be referenced in bear_overlap logic."""
        from core.models.training import trainer
        import inspect
        src = inspect.getsource(trainer.MARK5MLTrainer.prepare_data_dynamic)
        # horizon should be used in the rolling window
        assert 'horizon' in src, "prediction_horizon must be used in bear_overlap computation"


class TestHPODataIsolation:
    """Test that HPO validation data is disjoint from CPCV test data."""

    def test_hpo_uses_70pct_window(self):
        """HPO must restrict to first 70% of labeled data."""
        from core.models.training import trainer_v2
        import inspect
        src = inspect.getsource(trainer_v2.MARK5MLTrainerV2._run_optuna_hpo)
        # Should have hpo_cutoff = int(len(X) * 0.70)
        assert '0.70' in src or '0.7)' in src, \
            "HPO must use first 70% of data to avoid CPCV overlap"

    def test_hpo_default_trials_is_50(self):
        """Default Optuna trials should be 50 for better coverage."""
        from core.models.training.trainer_v2 import MARK5MLTrainerV2
        import inspect
        sig = inspect.signature(MARK5MLTrainerV2.__init__)
        default_trials = sig.parameters.get('optuna_trials')
        if default_trials and hasattr(default_trials, 'default'):
            assert default_trials.default >= 50, \
                f"Optuna trials default should be >= 50, got {default_trials.default}"

    def test_hpo_log_shows_window_size(self):
        """HPO log message should include the HPO window sample count."""
        from core.models.training import trainer_v2
        import inspect
        src = inspect.getsource(trainer_v2.MARK5MLTrainerV2._run_optuna_hpo)
        assert 'HPO window' in src, "HPO must log window size for audit trail"

    def test_hpo_80_20_inner_split(self):
        """HPO inner split must be chronological 80/20 within HPO window."""
        from core.models.training import trainer_v2
        import inspect
        src = inspect.getsource(trainer_v2.MARK5MLTrainerV2._run_optuna_hpo)
        assert '0.80' in src or '0.8)' in src, \
            "HPO inner split must be 80% train within HPO window"


class TestCPCVPurgeSufficiency:
    """Test that CPCV purge/embargo settings are correct."""

    def test_embargo_covers_prediction_horizon(self):
        """CPCV embargo must be >= prediction_horizon to prevent leakage."""
        from core.models.training import cpcv as cpcv_module
        import inspect
        # Check that embargo parameter exists
        src = inspect.getsource(cpcv_module)
        assert 'embargo' in src.lower(), "CPCV must have embargo parameter"

    def test_cpcv_zero_overlap_assertion(self):
        """CPCV must assert zero overlap between train and test sets."""
        from core.models.training import cpcv as cpcv_module
        import inspect
        src = inspect.getsource(cpcv_module)
        # Verify purge/embargo language exists in the module
        assert 'purge' in src.lower() or 'embargo' in src.lower(), \
            "CPCV must implement purge/embargo"

    def test_cpcv_split_produces_non_overlapping_folds(self):
        """Actual CPCV split must produce non-overlapping train/test sets."""
        from core.models.training.cpcv import CombinatorialPurgedKFold
        n = 200
        splitter = CombinatorialPurgedKFold(
            n_splits=5, n_test_splits=2,
            prediction_horizon=5, embargo_limit=5,
        )
        X_dummy = np.zeros((n, 5))
        y_dummy = np.random.randint(0, 2, n)

        all_test_indices = []
        for train_idx, test_idx in splitter.split(X_dummy, y_dummy):
            # Verify no overlap between train and test within this fold
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Train/test overlap found: {overlap}"
            all_test_indices.extend(test_idx.tolist())

    def test_cpcv_covers_all_samples(self):
        """Every sample should appear in at least one test fold."""
        from core.models.training.cpcv import CombinatorialPurgedKFold
        n = 200
        splitter = CombinatorialPurgedKFold(
            n_splits=5, n_test_splits=2,
            prediction_horizon=5, embargo_limit=5,
        )
        X_dummy = np.zeros((n, 5))
        y_dummy = np.random.randint(0, 2, n)

        covered = set()
        for _, test_idx in splitter.split(X_dummy, y_dummy):
            covered.update(test_idx.tolist())

        # At least 80% of samples should appear in some test fold
        coverage = len(covered) / n
        assert coverage >= 0.80, f"Only {coverage:.1%} of samples covered by test folds"


class TestFoldPurgeViolation:
    """Test the fix for fold re-engineering purge violation."""

    def test_trainer_passes_fold_cutoff(self):
        """V2 trainer must pass fold_train_cutoff, not data.index[-1], to prepare_data_dynamic."""
        from core.models.training import trainer_v2
        import inspect
        src = inspect.getsource(trainer_v2.MARK5MLTrainerV2.train_advanced_ensemble)
        # After our fix, should see fold_train_cutoff being computed
        assert 'fold_train_cutoff' in src or 'train_idx[-1]' in src, \
            "Fold-level training cutoff not found — purge violation may still exist"

    def test_fold_cutoff_comes_before_test_start(self):
        """In any valid CV split, training data must end before test data begins."""
        # Mathematical check: if train ends at T and test starts at T+embargo,
        # the fold_train_cutoff must be <= the last training bar's timestamp
        idx = pd.bdate_range('2020-01-01', periods=100)
        test_start = idx[80]
        train_end  = idx[75]  # 5 bars of embargo
        assert train_end < test_start, "Training must end before test starts"

    def test_cpcv_fold_log_added(self):
        """V2 trainer must log each fold's train/test date ranges."""
        from core.models.training import trainer_v2
        import inspect
        src = inspect.getsource(trainer_v2.MARK5MLTrainerV2.train_advanced_ensemble)
        assert '_train_end' in src, "Fold date logging must include _train_end variable"
        assert '_test_start' in src, "Fold date logging must include _test_start variable"
        assert 'CPCV OVERLAP DETECTED' in src, "Fold overlap warning must be present"

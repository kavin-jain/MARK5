"""
MARK5 Feature Leakage Tests
━━━━━━━━━━━━━━━━━━━━━━━━━━━

This test suite proves data leakage caused by feature engineering rolling windows
overlapping with test splits in the CPCV framework.

LEAKAGE DEFINITION:
In time-series cross-validation, training features at time T must not depend on 
any data from the test set. If a feature uses a rolling window of size W, 
and a test set exists at [T_start, T_end], then any training sample at T > T_end
must have its window [T-W, T] entirely outside [T_start, T_end], OR the test
data must be masked/purged.
"""

import pytest
import numpy as np
import pandas as pd
from core.models.features import engineer_features_df, FEATURE_COLS

def test_feature_engineering_isolation_across_splits():
    """
    ASSERT: Training features do not change when test set data is modified.
    
    If features use rolling windows (like MA200), then training samples 
    following a test block will 'see' the test data in their rolling windows.
    """
    # 1. Create a synthetic dataset
    n_samples = 2000
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="h")
    data = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # 2. Generate features on the original data
    test_indices = list(range(400, 500))
    X_orig = engineer_features_df(data, test_indices=test_indices)
    
    # 3. Identify a potential "test block" in the middle
    test_start = 400
    test_end = 500
    
    # 4. Create a "corrupted" dataset where the test block is different
    data_corrupted = data.copy()
    data_corrupted.iloc[test_start:test_end] *= 2.0 # Drastic change
    
    X_corr = engineer_features_df(data_corrupted, test_indices=test_indices)
    
    # 5. Check training samples AFTER the test block
    # Test block is [400, 500]. 
    # Any window that includes [400, 500] will be NaN and dropped.
    # The largest window is 200 (dist_ma200).
    # So indices from 400 to 500+200=700 will be dropped.
    # Index 800 should be safe.
    check_idx = 800
    
    # Align indices
    common_idx = X_orig.index.intersection(X_corr.index)
    assert len(common_idx) > 0, "No common indices found between original and corrupted features."
    
    target_ts = data.index[check_idx]
    if target_ts not in common_idx:
        # Fallback to the first available index after the test block + window
        after_test_indices = common_idx[common_idx > data.index[700]]
        if len(after_test_indices) > 0:
            target_ts = after_test_indices[0]
        else:
            pytest.fail(f"No valid indices found after the test block and window. common_idx max: {common_idx.max()}")

    val_orig = X_orig.loc[target_ts]
    val_corr = X_corr.loc[target_ts]
    
    diff = (val_orig - val_corr).abs()
    
    # We expect this to fail (diff > 0) because features use rolling windows 
    # that span across the test block.
    assert (diff < 1e-9).all(), (
        f"ROLLING WINDOW LEAKAGE DETECTED: Features at {target_ts} (index {check_idx}) "
        f"changed when data in the 'test' range [400, 500] was modified. "
        f"Differences:\n{diff[diff > 1e-9]}"
    )

def test_global_standardization_leakage():
    """
    ASSERT: Standardization parameters are not calculated on the full dataset.
    """
    # Create data with two distinct regimes
    n = 500
    data1 = pd.DataFrame({
        'open': np.random.normal(10, 1, n),
        'high': np.random.normal(11, 1, n),
        'low': np.random.normal(9, 1, n),
        'close': np.random.normal(10, 1, n),
        'volume': np.random.randint(100, 200, n)
    }, index=pd.date_range("2020-01-01", periods=n, freq="h"))
    
    # Regime 2 has much higher values
    data2 = pd.DataFrame({
        'open': np.random.normal(100, 1, n),
        'high': np.random.normal(101, 1, n),
        'low': np.random.normal(99, 1, n),
        'close': np.random.normal(100, 1, n),
        'volume': np.random.randint(100, 200, n)
    }, index=pd.date_range("2020-02-01", periods=n, freq="h"))
    
    # Features for Regime 1 alone
    X1 = engineer_features_df(data1)
    
    # Features for Regime 1 when Regime 2 is present in the same dataframe
    combined_data = pd.concat([data1, data2])
    X_combined = engineer_features_df(combined_data)
    
    # Check a sample in the middle of Regime 1
    check_ts = data1.index[250]
    
    if check_ts in X1.index and check_ts in X_combined.index:
        val1 = X1.loc[check_ts]
        val_comb = X_combined.loc[check_ts]
        
        diff = (val1 - val_comb).abs()
        
        assert (diff < 1e-9).all(), (
            f"GLOBAL LEAKAGE DETECTED: Features at {check_ts} changed when "
            f"future data was appended to the dataset. "
            f"Differences:\n{diff[diff > 1e-9]}"
        )

if __name__ == "__main__":
    pytest.main([__file__])

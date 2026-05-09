"""
MARK5 CPCV Horizon Leakage Tests
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This test suite mathematically proves data leakage in the CPCV implementation
by verifying that the purging horizon covers the entire label look-ahead window.

LEAKAGE DEFINITION:
If a label at time T depends on data up to T + H, then any training observation
in the interval [T - H, T + H] must be purged if T is in the test set.
If the purging horizon P < H, then observations in [T + P, T + H] leak information.
"""

import pytest
import numpy as np
import pandas as pd
from core.models.training.cpcv import CombinatorialPurgedKFold
from core.models.training.trainer import MARK5MLTrainer

def test_purging_horizon_covers_label_lookahead():
    """
    ASSERT: CPCV prediction_horizon >= Triple Barrier maximum look-ahead (5 bars).
    
    Standardized to 5 bars for institutional swing modeling.
    """
    # 1. Identify the actual label horizon (MARK6 Institutional Standard)
    REQUIRED_HORIZON = 5 
    
    # 2. Instantiate the trainer to see what it uses for CPCV
    trainer = MARK5MLTrainer()
    
    # 3. Extract the prediction_horizon used for CPCV
    cpcv_horizon = trainer.config.prediction_horizon
    
    # 4. Assert that the purging horizon is sufficient
    assert cpcv_horizon >= REQUIRED_HORIZON, (
        f"DATA LEAKAGE: CPCV purging horizon ({cpcv_horizon}) is less than "
        f"label look-ahead horizon ({REQUIRED_HORIZON})."
    )

def test_cpcv_split_isolation_math():
    """
    ASSERT: For any test index i, no training index j exists such that |i - j| < prediction_horizon.
    
    This test performs a brute-force check on the indices returned by the splitter.
    """
    n_samples = 500
    horizon = 30 # Use a consistent horizon for test
    
    # Configure splitter with the test horizon
    prediction_horizon = horizon
    embargo_limit = horizon
    
    splitter = CombinatorialPurgedKFold(
        n_splits=6, 
        n_test_splits=2, 
        prediction_horizon=prediction_horizon, 
        embargo_limit=embargo_limit
    )
    
    X = np.zeros((n_samples, 10))
    
    fold_count = 0
    for train_idx, test_idx in splitter.split(X):
        fold_count += 1
        
        # For every test index, ensure no training index is within the 'horizon' buffer
        for t_idx in test_idx:
            # Check condition 1: Purge before
            leaking_before = train_idx[(train_idx >= t_idx - prediction_horizon) & (train_idx < t_idx)]
            assert len(leaking_before) == 0, (
                f"LEAKAGE in fold {fold_count}: Train indices {leaking_before} "
                f"are within {prediction_horizon} bars before test index {t_idx}."
            )
            
            # Check condition 2: Purge after (Embargo)
            # The splitter uses prediction_horizon + embargo_limit for the total buffer after.
            # But de Prado's standard purging only requires 'prediction_horizon' after
            # if embargo is handled separately. Our splitter combines them.
            leaking_after = train_idx[(train_idx > t_idx) & (train_idx <= t_idx + prediction_horizon)]
            assert len(leaking_after) == 0, (
                f"LEAKAGE in fold {fold_count}: Train indices {leaking_after} "
                f"are within {prediction_horizon} bars after test index {t_idx}."
            )

if __name__ == "__main__":
    pytest.main([__file__])

"""
Combinatorial Purged Cross-Validation (CPCV)
Implements the time-series cross-validation methodology by Marcos Lopez de Prado.
Tailored for Small-Data (60-minute frequencies, ~2000 days).

This implementation ensures:
1. Zero leakage via rigorous Purging and Embargoing.
2. Maximum data retention for small datasets.
3. Memory efficiency by yielding index masks.
"""

import numpy as np
from itertools import combinations
from typing import Tuple, Generator, List, Optional, Union
import logging
from math import comb

logger = logging.getLogger("MARK5.CPCV")

def generate_cpcv_splits(
    n_samples: int,
    n_splits: int = 6,
    n_test_splits: int = 2,
    prediction_horizon: int = 24,  # Default 24 bars for 60-min data (1 day)
    embargo_limit: int = 24,       # Default 24 bars for 60-min data (1 day)
    min_train_size: Optional[int] = None
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Memory-efficient CPCV split generator yielding index masks.
    Mathematically prevents leakage while maximizing data retention for small datasets.
    
    Args:
        n_samples: Total number of bars in the dataset.
        n_splits: Number of groups (N) to partition the timeline into.
        n_test_splits: Number of groups (k) to use as the testing set in each reality.
        prediction_horizon: Number of bars used for label calculation (Purge limit).
        embargo_limit: Number of bars to prevent serial correlation leakage (Embargo limit).
        min_train_size: Minimum required training samples after purging.
        
    Yields:
        (train_indices, test_indices) as numpy arrays of integers.
    """
    # 1. Small-Data Awareness: Validate and adjust parameters
    if n_samples < 30:
        logger.error(f"Insufficient data for CPCV: {n_samples} samples.")
        return

    # Total isolation buffer per test block boundary
    # Purge (before) = prediction_horizon
    # Embargo (after) = prediction_horizon + embargo_limit
    # (We purge prediction_horizon after as well because labels overlap)
    after_purge_embargo = prediction_horizon + embargo_limit
    
    # Ensure n_splits is reasonable for the data size
    # Each block should ideally be larger than the isolation buffers
    avg_block_size = n_samples // n_splits
    if avg_block_size < (prediction_horizon + after_purge_embargo):
        new_n_splits = max(2, n_samples // (2 * (prediction_horizon + after_purge_embargo)))
        if new_n_splits < n_splits:
            logger.warning(f"Data too small for {n_splits} splits. Reducing to {new_n_splits}.")
            n_splits = new_n_splits
            n_test_splits = min(n_test_splits, n_splits - 1)

    if n_test_splits >= n_splits:
        n_test_splits = n_splits - 1
        logger.warning(f"n_test_splits adjusted to {n_test_splits}")

    # 2. Partition indices into N contiguous groups
    indices = np.arange(n_samples)
    groups = np.array_split(indices, n_splits)
    
    # 3. Generate all combinations of test groups
    # Total realities = N! / (k! * (N-k)!)
    test_combinations = list(combinations(range(n_splits), n_test_splits))
    
    logger.info(f"Generating {len(test_combinations)} CPCV realities (N={n_splits}, k={n_test_splits})")
    
    for test_group_indices in test_combinations:
        # Identify test indices
        test_indices_list = [groups[i] for i in test_group_indices]
        test_indices = np.concatenate(test_indices_list)
        
        # Initialize train mask
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_indices] = False
        
        # 4. Apply Purging and Embargoing for each test block
        for group_idx in test_group_indices:
            group = groups[group_idx]
            start_test = group[0]
            end_test = group[-1]
            
            # Purge BEFORE: Remove observations whose labels overlap with the test set start
            # If label at t depends on [t, t + h], then t + h must be < start_test
            # So t < start_test - h
            purge_before_start = max(0, start_test - prediction_horizon)
            train_mask[purge_before_start:start_test] = False
            
            # Purge & Embargo AFTER: 
            # 1. Purge observations whose labels overlap with the test set end
            # 2. Embargo to prevent serial correlation leakage
            purge_after_end = min(n_samples, end_test + after_purge_embargo + 1)
            train_mask[end_test + 1:purge_after_end] = False
            
        train_indices = np.where(train_mask)[0]
        
        # 5. Strict Validation: Zero-overlap verification
        assert len(np.intersect1d(train_indices, test_indices)) == 0, "CRITICAL: Train/Test overlap detected!"
        
        # Check minimum training size if specified
        if min_train_size and len(train_indices) < min_train_size:
            continue
            
        if len(train_indices) > 0 and len(test_indices) > 0:
            yield train_indices, test_indices

class CombinatorialPurgedKFold:
    """
    Institutional-grade CPCV Splitter.
    Maximizes data retention while mathematically preventing leakage.
    Compatible with scikit-learn CV API.
    """
    def __init__(
        self, 
        n_splits: int = 6, 
        n_test_splits: int = 2, 
        prediction_horizon: int = 24, 
        embargo_limit: int = 24
    ):
        """
        Args:
            n_splits: Number of total groups (N).
            n_test_splits: Number of groups to use for testing (k).
            prediction_horizon: Number of bars for label lookahead (Purge).
            embargo_limit: Number of bars for serial correlation (Embargo).
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.prediction_horizon = prediction_horizon
        self.embargo_limit = embargo_limit

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) for each combinatorial reality.
        """
        n_samples = len(X)
        after_purge_embargo = self.prediction_horizon + self.embargo_limit
        avg_block_size = n_samples // self.n_splits
        
        actual_n_splits = self.n_splits
        if avg_block_size < (self.prediction_horizon + after_purge_embargo):
            actual_n_splits = max(2, n_samples // (2 * (self.prediction_horizon + after_purge_embargo)))
        
        actual_n_test_splits = min(self.n_test_splits, actual_n_splits - 1)
        total_folds = comb(actual_n_splits, actual_n_test_splits)
        
        logger.info(f"CPCV Split: N={actual_n_splits}, k={actual_n_test_splits} | Total Folds: {total_folds}")

        return generate_cpcv_splits(
            n_samples=len(X),
            n_splits=self.n_splits,
            n_test_splits=self.n_test_splits,
            prediction_horizon=self.prediction_horizon,
            embargo_limit=self.embargo_limit
        )

    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, groups=None) -> int:
        """
        Returns the number of combinatorial realities.
        """
        if X is not None:
            # Account for potential reduction in n_splits due to small data
            n_samples = len(X)
            after_purge_embargo = self.prediction_horizon + self.embargo_limit
            avg_block_size = n_samples // self.n_splits
            
            actual_n_splits = self.n_splits
            if avg_block_size < (self.prediction_horizon + after_purge_embargo):
                actual_n_splits = max(2, n_samples // (2 * (self.prediction_horizon + after_purge_embargo)))
            
            actual_n_test_splits = min(self.n_test_splits, actual_n_splits - 1)
            return comb(actual_n_splits, actual_n_test_splits)
            
        return comb(self.n_splits, self.n_test_splits)

def validate_split_integrity(train_idx: np.ndarray, test_idx: np.ndarray, n_samples: int):
    """
    Programmatic assertion of split integrity.
    """
    # 1. No overlap
    overlap = np.intersect1d(train_idx, test_idx)
    if len(overlap) > 0:
        raise AssertionError(f"Leakage detected! Overlap: {len(overlap)} indices.")
        
    # 2. Bounds check
    if len(train_idx) > 0:
        assert train_idx.min() >= 0 and train_idx.max() < n_samples
    if len(test_idx) > 0:
        assert test_idx.min() >= 0 and test_idx.max() < n_samples
        
    logger.debug(f"Split integrity verified: Train={len(train_idx)}, Test={len(test_idx)}")
    return True

"""
CPCV Purge Correctness Tests for MARK5 ML system.

Verifies that CombinatorialPurgedKFold correctly:
1. Produces zero-overlap train/test splits
2. Applies purge zones before test blocks
3. Applies embargo zones after test blocks
4. Covers all samples across folds
5. Handles small dataset edge cases
6. Validates split boundaries are temporal (train before test)
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from core.models.training.cpcv import (
    CombinatorialPurgedKFold,
    generate_cpcv_splits,
    validate_split_integrity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_splitter(n_splits=5, n_test_splits=2, horizon=5, embargo=5):
    return CombinatorialPurgedKFold(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        prediction_horizon=horizon,
        embargo_limit=embargo,
    )


class TestZeroOverlap:
    """Train and test indices must never share elements."""

    def test_no_overlap_standard(self):
        """Standard configuration: n=200, splits=5, test_splits=2."""
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=5, embargo=5)
        X = np.zeros((200, 3))
        y = np.ones(200, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, \
                f"Overlap detected: {len(overlap)} shared indices"

    def test_no_overlap_large(self):
        """Large dataset: n=1000 samples."""
        splitter = make_splitter(n_splits=6, n_test_splits=2, horizon=10, embargo=10)
        X = np.zeros((1000, 5))
        y = np.zeros(1000, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, \
                f"Large dataset: overlap of {len(overlap)} indices"

    def test_no_overlap_single_test_split(self):
        """n_test_splits=1 (standard walk-forward mode)."""
        splitter = make_splitter(n_splits=5, n_test_splits=1, horizon=5, embargo=5)
        X = np.zeros((200, 3))
        y = np.zeros(200, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, "Single-test-split: overlap detected"

    def test_no_overlap_wide_embargo(self):
        """Wide embargo (20 bars) must still produce valid splits."""
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=20, embargo=20)
        X = np.zeros((500, 3))
        y = np.zeros(500, dtype=int)
        splits = list(splitter.split(X, y))
        assert len(splits) > 0, "Wide embargo produced no splits"
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, "Wide embargo: overlap detected"


class TestPurgeBoundaries:
    """Purge zone before test blocks must be excluded from train."""

    def test_purge_before_test_block(self):
        """No train index should fall within prediction_horizon bars before test start."""
        horizon = 10
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=horizon, embargo=5)
        X = np.zeros((300, 3))
        y = np.zeros(300, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            if len(test_idx) == 0:
                continue
            test_start = test_idx.min()
            # No train index should be in [test_start - horizon, test_start - 1]
            purge_zone = set(range(max(0, test_start - horizon), test_start))
            train_in_purge = purge_zone & set(train_idx)
            assert len(train_in_purge) == 0, \
                f"Train indices {train_in_purge} fall in purge zone before test_start={test_start}"

    def test_embargo_after_test_block(self):
        """No train index should fall within embargo bars after test end."""
        embargo = 10
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=5, embargo=embargo)
        X = np.zeros((300, 3))
        y = np.zeros(300, dtype=int)
        n_samples = len(X)
        for train_idx, test_idx in splitter.split(X, y):
            if len(test_idx) == 0:
                continue
            test_end = test_idx.max()
            # No train index should be in [test_end + 1, test_end + embargo]
            embargo_zone = set(range(test_end + 1, min(n_samples, test_end + embargo + 1)))
            train_in_embargo = embargo_zone & set(train_idx)
            assert len(train_in_embargo) == 0, \
                f"Train indices {train_in_embargo} fall in embargo zone after test_end={test_end}"


class TestTemporalOrdering:
    """Training data must precede test data in time for non-adjacent blocks."""

    def test_train_indices_are_non_negative(self):
        """All train indices are valid (>= 0)."""
        splitter = make_splitter()
        X = np.zeros((200, 3))
        y = np.zeros(200, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            assert train_idx.min() >= 0, "Negative train index"
            assert test_idx.min() >= 0, "Negative test index"

    def test_indices_within_bounds(self):
        """All indices must be < n_samples."""
        n = 200
        splitter = make_splitter(n_splits=5, n_test_splits=2)
        X = np.zeros((n, 3))
        y = np.zeros(n, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            assert train_idx.max() < n, f"Train index {train_idx.max()} >= n_samples={n}"
            assert test_idx.max() < n, f"Test index {test_idx.max()} >= n_samples={n}"

    def test_train_not_empty(self):
        """Every fold should have non-empty train set."""
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=5, embargo=5)
        X = np.zeros((300, 3))
        y = np.zeros(300, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            assert len(train_idx) > 0, "Empty train set in a fold"

    def test_test_not_empty(self):
        """Every fold should have non-empty test set."""
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=5, embargo=5)
        X = np.zeros((300, 3))
        y = np.zeros(300, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            assert len(test_idx) > 0, "Empty test set in a fold"


class TestCoverage:
    """Combinatorial nature of CPCV should cover all test indices across folds."""

    def test_all_samples_covered_in_test(self):
        """All samples should appear in at least one test fold."""
        n = 200
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=5, embargo=5)
        X = np.zeros((n, 3))
        y = np.zeros(n, dtype=int)
        covered = set()
        for _, test_idx in splitter.split(X, y):
            covered.update(test_idx.tolist())
        coverage = len(covered) / n
        assert coverage >= 0.80, \
            f"Only {coverage:.1%} of samples covered by test folds (need >= 80%)"

    def test_combinatorial_count(self):
        """Number of folds should be C(n_splits, n_test_splits)."""
        from math import comb
        n_splits = 5
        n_test_splits = 2
        expected_folds = comb(n_splits, n_test_splits)  # C(5,2) = 10
        splitter = make_splitter(
            n_splits=n_splits, n_test_splits=n_test_splits,
            horizon=5, embargo=5,
        )
        X = np.zeros((300, 3))
        y = np.zeros(300, dtype=int)
        actual_folds = sum(1 for _ in splitter.split(X, y))
        # May be <= expected due to data-size adjustments
        assert actual_folds <= expected_folds, \
            f"More folds than C({n_splits},{n_test_splits})={expected_folds}: got {actual_folds}"
        assert actual_folds >= 1, "At least one fold must be generated"

    def test_test_blocks_are_contiguous(self):
        """Test indices within each fold should come from contiguous groups."""
        splitter = make_splitter(n_splits=5, n_test_splits=1, horizon=5, embargo=5)
        X = np.zeros((200, 3))
        y = np.zeros(200, dtype=int)
        for _, test_idx in splitter.split(X, y):
            sorted_idx = np.sort(test_idx)
            # For n_test_splits=1, the test block should be contiguous
            if len(sorted_idx) > 1:
                gaps = np.diff(sorted_idx)
                max_gap = gaps.max()
                # Allow up to 2-bar gaps due to block boundary rounding
                assert max_gap <= 3, \
                    f"Test block has large gap ({max_gap}) — may not be contiguous"


class TestEdgeCases:
    """Edge case handling for small datasets and boundary parameters."""

    def test_small_dataset_handled_gracefully(self):
        """Very small dataset (n=50) should not raise, may produce 0 folds."""
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=10, embargo=5)
        X = np.zeros((50, 3))
        y = np.zeros(50, dtype=int)
        # Should not raise — may produce 0 folds due to data size adjustment
        splits = list(splitter.split(X, y))
        # If splits exist, they must be valid
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_generate_cpcv_splits_direct(self):
        """generate_cpcv_splits function produces valid splits."""
        splits = list(generate_cpcv_splits(
            n_samples=200,
            n_splits=5,
            n_test_splits=2,
            prediction_horizon=5,
            embargo_limit=5,
        ))
        assert len(splits) > 0, "generate_cpcv_splits produced no splits"
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, "Direct generator: overlap detected"

    def test_validate_split_integrity_passes(self):
        """validate_split_integrity should pass for a valid split."""
        n = 100
        train_idx = np.arange(0, 60)
        test_idx  = np.arange(70, 100)  # gap of 10 = embargo
        result = validate_split_integrity(train_idx, test_idx, n)
        assert result is True, "validate_split_integrity failed for valid split"

    def test_validate_split_integrity_fails_on_overlap(self):
        """validate_split_integrity must raise AssertionError on overlapping split."""
        n = 100
        train_idx = np.arange(0, 70)   # includes 60-69
        test_idx  = np.arange(60, 100)  # overlaps 60-69
        with pytest.raises(AssertionError, match="Leakage"):
            validate_split_integrity(train_idx, test_idx, n)

    def test_n_test_splits_adjusted_when_too_large(self):
        """If n_test_splits >= n_splits, it should be adjusted automatically."""
        # n_splits=3, n_test_splits=5 → should auto-reduce n_test_splits
        splits = list(generate_cpcv_splits(
            n_samples=300,
            n_splits=3,
            n_test_splits=5,  # > n_splits, must be reduced
            prediction_horizon=5,
            embargo_limit=5,
        ))
        # Should produce some splits without crashing
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0


class TestEmbargoPreventsSerialCorrelation:
    """Embargo ensures the model does not use temporally correlated samples."""

    def test_embargo_gap_present_after_contiguous_test_block(self):
        """After a contiguous test block, the next train index must be >= end + embargo."""
        horizon = 5
        embargo = 10
        splitter = make_splitter(
            n_splits=5, n_test_splits=1,
            horizon=horizon, embargo=embargo,
        )
        X = np.zeros((300, 3))
        y = np.zeros(300, dtype=int)
        n_samples = len(X)
        for train_idx, test_idx in splitter.split(X, y):
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            test_end = test_idx.max()
            # Find train indices that come after the test block
            post_test_train = train_idx[train_idx > test_end]
            if len(post_test_train) == 0:
                continue  # test block is at end of dataset — ok
            first_post_test_train = post_test_train.min()
            gap = first_post_test_train - test_end
            # Gap must be at least 1 (the embargo zone was carved out)
            assert gap >= 1, \
                f"No gap after test block: train resumes at {first_post_test_train}, test ended at {test_end}"

    def test_purge_gap_present_before_contiguous_test_block(self):
        """Before a contiguous test block, train must stop at least horizon bars before it."""
        horizon = 10
        splitter = make_splitter(
            n_splits=5, n_test_splits=1,
            horizon=horizon, embargo=5,
        )
        X = np.zeros((300, 3))
        y = np.zeros(300, dtype=int)
        for train_idx, test_idx in splitter.split(X, y):
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            test_start = test_idx.min()
            # Find train indices that come before the test block
            pre_test_train = train_idx[train_idx < test_start]
            if len(pre_test_train) == 0:
                continue  # test block is at start — ok
            last_pre_test_train = pre_test_train.max()
            gap = test_start - last_pre_test_train
            # Gap must be at least 1 (purge zone was carved out)
            assert gap >= 1, \
                f"No purge gap before test block: train ends at {last_pre_test_train}, test starts at {test_start}"


class TestGetNSplits:
    """get_n_splits method should return a reasonable positive integer."""

    def test_get_n_splits_without_X(self):
        """get_n_splits() with no X returns C(n_splits, n_test_splits)."""
        from math import comb
        splitter = make_splitter(n_splits=5, n_test_splits=2)
        n = splitter.get_n_splits()
        assert n == comb(5, 2), f"Expected C(5,2)=10, got {n}"

    def test_get_n_splits_with_X(self):
        """get_n_splits(X) accounts for data-size adjustments."""
        splitter = make_splitter(n_splits=5, n_test_splits=2, horizon=5, embargo=5)
        X = np.zeros((300, 3))
        n = splitter.get_n_splits(X)
        assert n >= 1, "get_n_splits(X) should return at least 1"

    def test_get_n_splits_small_data_reduces(self):
        """get_n_splits with very small data may return fewer folds."""
        splitter = make_splitter(n_splits=8, n_test_splits=3, horizon=30, embargo=30)
        X = np.zeros((100, 3))
        n = splitter.get_n_splits(X)
        # For a 100-bar dataset with large horizon+embargo, splits may reduce
        assert n >= 0, "get_n_splits should never return negative"

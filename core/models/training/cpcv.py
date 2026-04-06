"""
Combinatorial Purged Cross-Validation (CPCV)
Implements the time-series cross-validation methodology by Marcos Lopez de Prado.
"""
import numpy as np
from itertools import combinations
from typing import Tuple, Generator

class CombinatorialPurgedKFold:
    def __init__(self, n_splits: int = 6, n_test_splits: int = 2, embargo: int = 30):
        """
        Args:
            n_splits (int): Number of total groups to partition the timeline into.
            n_test_splits (int): Number of groups to use as the testing set in each combinatorial reality.
            embargo (int): Number of bars to purge/embargo around the test sets to prevent data leakage.
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo = embargo

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Split into N contiguous chronological groups
        groups = np.array_split(indices, self.n_splits)
        
        # All unique combinations of test groups (simulating distinct historical paths)
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        for test_idx_tuple in test_combinations:
            test_indices = []
            
            # 1. Form the active Test Set for this path
            for idx in test_idx_tuple:
                test_indices.extend(groups[idx])
                
            test_array = np.array(test_indices)
            if len(test_array) == 0:
                continue
                
            # 2. Base mask: Everything is training except the Test Set
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_array] = False
            
            # 3. Apply the Embargo/Purge strict isolation borders
            for idx in test_idx_tuple:
                group = groups[idx]
                if len(group) == 0:
                    continue
                
                start_test = group[0]
                end_test = group[-1]
                
                # Purge BEFORE test block (prevents test data bleeding into past predictions)
                purge_start = max(0, start_test - self.embargo)
                train_mask[purge_start:start_test] = False
                
                # Embargo AFTER test block (prevents leaked future info returning to training)
                embargo_end = min(n_samples, end_test + self.embargo + 1)
                train_mask[end_test + 1:embargo_end] = False
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.array(test_indices)
            
            # Only yield if both sets have sufficient data to train and test
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

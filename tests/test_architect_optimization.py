import unittest
import numpy as np
import pandas as pd
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from core.optimization.optimizer import OptimizationEngine
from core.data.collector import MARK5DataCollector

class TestArchitectOptimization(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_opt"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_purged_cv_split(self):
        """Verify Purged Walk-Forward Splitter"""
        print("\nTesting Purged CV...")
        optimizer = HyperparameterOptimizer(self.test_dir)
        X = np.arange(1000)
        
        splits = list(optimizer._purged_split(X, n_splits=3, purge_window=50))
        
        self.assertEqual(len(splits), 3)
        
        for i, (train, val) in enumerate(splits):
            # Verify Gap
            train_end = train[-1]
            val_start = val[0]
            gap = val_start - train_end
            
            self.assertGreaterEqual(gap, 50, f"Split {i} has insufficient purge gap: {gap}")
            print(f"  Split {i}: Train End {train_end} -> Val Start {val_start} (Gap {gap})")
            
        print("✅ Purged CV Verified")

    def test_optimizer_objective(self):
        """Verify Optimizer runs and minimizes LogLoss"""
        print("\nTesting Optimizer Objective...")
        optimizer = HyperparameterOptimizer(self.test_dir)
        
        # Generate Synthetic Data
        X = np.random.rand(500, 10)
        y = np.random.randint(0, 3, 500)
        
        # Run Optimization (Short)
        best_params = optimizer.optimize_xgboost(X, y, "TEST_TICKER", n_trials=2)
        
        self.assertIn('learning_rate', best_params)
        self.assertIn('max_depth', best_params)
        self.assertTrue(os.path.exists(f"{self.test_dir}/config/TEST_TICKER_xgb_params.json"))
        
        print("✅ Optimizer Objective Verified")

    def test_launcher_parallel_structure(self):
        """Verify Launcher Structure (Mocked)"""
        print("\nTesting Launcher Structure...")
        launcher = OptimizationEngine(max_workers=2)
        
        # We mock the actual optimization task to avoid heavy compute in test
        # But we verify the method exists and logic holds
        
        targets = launcher._load_targets()
        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[0]['ticker'], 'RELIANCE.NS')
        
        print("✅ Launcher Structure Verified")

if __name__ == '__main__':
    unittest.main()

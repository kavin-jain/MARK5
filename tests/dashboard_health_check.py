import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np
import importlib.util

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load dashboard.py as a module
DASHBOARD_PATH = os.path.join(PROJECT_ROOT, "dashboard.py")
spec = importlib.util.spec_from_file_location("dashboard_cli", DASHBOARD_PATH)
dashboard_cli = importlib.util.module_from_spec(spec)

# Mock missing dependencies
for mod in ["polars", "kiteconnect", "nsepython", "pyarrow", "joblib", "catboost", "xgboost", "lightgbm", "sklearn", "sklearn.linear_model", "sklearn.metrics", "sklearn.utils.class_weight", "scipy.stats", "scipy.special", "scipy.optimize"]:
    sys.modules[mod] = MagicMock()

class DashboardHealthCheck(unittest.TestCase):
    def setUp(self):
        # Mock Data
        # Need at least 60 + 250 = 310 bars for backtesting
        self.mock_df = pd.DataFrame({
            'open': np.random.rand(500),
            'high': np.random.rand(500),
            'low': np.random.rand(500),
            'close': np.random.rand(500),
            'volume': np.random.rand(500)
        }, index=pd.date_range(start='2023-01-01', periods=500))
        
        self.mock_metrics = {
            'Total Return (%)': 15.5,
            'Sharpe Ratio': 2.1,
            'Total Trades': 10,
            'Win Rate (%)': 60.0
        }

    def test_dashboard_full_flow(self):
        # Execute the module first to populate its namespace
        sys.modules["dashboard_cli"] = dashboard_cli
        spec.loader.exec_module(dashboard_cli)

        # We need to patch before calling main()
        with patch('rich.console.Console'), \
             patch('rich.prompt.IntPrompt.ask') as mock_int_prompt, \
             patch('rich.prompt.Prompt.ask') as mock_prompt, \
             patch('rich.prompt.Confirm.ask') as mock_confirm, \
             patch('rich.console.Console.clear'), \
             patch('builtins.input', return_value=''), \
             patch.object(dashboard_cli, 'MARK5MLTrainer') as mock_trainer, \
             patch.object(dashboard_cli, 'MARK5Predictor') as mock_predictor, \
             patch.object(dashboard_cli, 'RobustBacktester') as mock_backtester, \
             patch.object(dashboard_cli, 'DataPipeline') as mock_pipeline, \
             patch.object(dashboard_cli, 'RobustModelRegistry') as mock_registry:
            
            # Setup Mocks
            mock_trainer_inst = mock_trainer.return_value
            mock_trainer_inst.train_model.return_value = {'status': 'success'}
            mock_trainer_inst.fetch_data_for_training.return_value = self.mock_df
            
            mock_predictor_inst = mock_predictor.return_value
            mock_predictor_inst._container = True
            mock_predictor_inst.predict.return_value = {'signal': 'BUY'}
            
            mock_backtester_inst = mock_backtester.return_value
            mock_backtester_inst.run_simulation.return_value = (None, self.mock_metrics)
            
            mock_pipeline_inst = mock_pipeline.return_value
            mock_pipeline_inst.get_trending_stocks.return_value = pd.DataFrame({'symbol': ['SBIN'], 'change': [1.2]})
            mock_pipeline_inst.get_most_active.return_value = pd.DataFrame({'symbol': ['RELIANCE'], 'volume': [1000000]})
            mock_pipeline_inst.get_price_shockers.return_value = pd.DataFrame({'symbol': ['TCS'], 'shock': [2.5]})
            mock_pipeline_inst.get_fundamental_data.return_value = pd.DataFrame({'metric': ['PE'], 'value': [20]})
            
            mock_registry_inst = mock_registry.return_value
            mock_registry_inst.registry = {'SBIN.NS': {}}

            # Mock Universe to be smaller for faster testing
            dashboard_cli._DEFAULT_UNIVERSE = ["SBIN.NS"]

            # Define input sequence
            mock_int_prompt.side_effect = [
                1, 1, 2, 3, # Training: Single, All, Back
                2, 1, 60, 2, 60, 3, # Backtesting: Single, All, Back
                3, 1, 2, 3, 4, 5, 6, # ISE flow
                4, 5 # Status and Exit
            ]
            
            mock_prompt.side_effect = [
                "SBIN.NS", # Training Ticker
                "SBIN.NS", # Backtesting Ticker
                "RELIANCE" # ISE Ticker
            ]
            
            mock_confirm.return_value = True

            # Run main
            with self.assertRaises(SystemExit) as cm:
                dashboard_cli.main()
            
            self.assertEqual(cm.exception.code, 0)
            
            # Verify some calls
            mock_trainer_inst.train_model.assert_called()
            mock_backtester_inst.run_simulation.assert_called()
            mock_pipeline_inst.get_trending_stocks.assert_called()

if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()

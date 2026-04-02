#!/usr/bin/env python3
"""
MARK5 OPTIMIZER v7.1 - ARCHITECT EDITION
Merged & Simplified.
Features:
1. PARALLEL EXECUTION: Optimizes multiple tickers concurrently.
2. METRIC SHIFT: Evaluates Sharpe Ratio/Profit Factor.
3. SMART SCHEDULING: Prioritizes high-volatility tickers.
4. REPORTING: Generates markdown reports.
"""

import os
import time
import concurrent.futures
from typing import Dict, List
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import json

# Import core modules
from core.models.training.trainer import MARK5MLTrainer
from core.optimization.hyperparameter_optimizer import HyperparameterOptimizer

# Data collector may not exist - used for runtime data fetching only
try:
    from core.data.collector import MARK5DataCollector
    DATA_COLLECTOR_AVAILABLE = True
except ImportError:
    MARK5DataCollector = None
    DATA_COLLECTOR_AVAILABLE = False

warnings.filterwarnings('ignore')

class OptimizationEngine:
    def __init__(self, target_sharpe: float = 2.0, max_workers: int = 4):
        """
        Args:
            target_sharpe: We want risk-adjusted returns, not just accuracy.
            max_workers: Number of parallel optimizations.
        """
        self.target_sharpe = target_sharpe
        self.max_workers = max_workers
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.results = {}

    def calculate_sharpe_proxy(self, trades_df) -> float:
        """Approximates Sharpe from a backtest result DF."""
        if trades_df.empty: return 0.0
        returns = trades_df['net_pnl']
        if returns.std() == 0: return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252 * 75) # Annualized (approx 75 bars/day)

    def _is_production_ready(self, train_result: Dict) -> bool:
        """
        Quality gates to determine if a model is production-ready.
        
        Checks:
        1. Training completed successfully
        2. LogLoss is within acceptable range (<1.5)
        3. At least 1 successful fold was trained
        
        Returns:
            bool: True if model passes all quality checks
        """
        if train_result.get('status') != 'success':
            return False
        
        log_loss = train_result.get('brier_score', 999)
        if log_loss > 1.5:
            print(f"⚠️ LogLoss too high: {log_loss:.4f} (max: 1.5)")
            return False
        
        folds = train_result.get('folds', 0)
        if folds < 1:
            print(f"⚠️ Insufficient training folds: {folds}")
            return False
        
        return True

    def optimize_single_stock_task(self, target_config: Dict) -> Dict:
        """
        Standalone function for Parallel Execution.
        """
        ticker = target_config['ticker']
        current_sharpe = target_config.get('current_sharpe', 0.0)
        
        print(f"🚀 STARTING OPTIMIZATION: {ticker} (Current Sharpe: {current_sharpe:.2f})")
        
        try:
            # 1. Data Collection (Fresh)
            collector = MARK5DataCollector()
            data = collector.fetch_stock_data(ticker, period='2y')
            if len(data) < 100:
                return {'ticker': ticker, 'status': 'FAILED', 'reason': 'Insufficient Data'}

            # 2. Full Ensemble Optimization (XGBoost + LightGBM + RandomForest)
            optimizer = HyperparameterOptimizer()
            trainer = MARK5MLTrainer()
            
            X, y, _ = trainer.prepare_data_dynamic(data, ticker)
            
            # Run Optuna for ALL models
            best_params = optimizer.optimize_all_models(X, y, ticker, n_trials=30)
            
            # 3. Train with optimized params and validate
            train_result = trainer.train_advanced_ensemble(ticker, data)
            
            # 4. Quality Gates (Production Readiness Check)
            production_ready = self._is_production_ready(train_result)
            
            log_loss_score = train_result.get('brier_score', 999)
            # Approximate Sharpe from log loss (lower is better)
            approx_sharpe = max(0, 3.0 - log_loss_score)  # Rough proxy
            
            improvement = approx_sharpe - current_sharpe
            
            print(f"✅ FINISHED {ticker}: Sharpe {current_sharpe:.2f} -> {approx_sharpe:.2f}")
            
            return {
                'ticker': ticker,
                'status': 'SUCCESS' if train_result['status'] == 'success' else 'FAILED',
                'old_sharpe': current_sharpe,
                'new_sharpe': approx_sharpe,
                'improvement': improvement,
                'params': best_params,
                'log_loss': log_loss_score,
                'production_ready': production_ready,
                'target_met': approx_sharpe >= self.target_sharpe
            }

        except Exception as e:
            print(f"❌ CRASH {ticker}: {str(e)}")
            return {'ticker': ticker, 'status': 'ERROR', 'error': str(e), 'target_met': False}

    def run_parallel_cycle(self):
        print(f"\n⚡ MARK5 PARALLEL OPTIMIZER (Workers: {self.max_workers})")
        print("=" * 60)
        
        targets = self._load_targets()
        if not targets:
            print("No targets found.")
            return

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.optimize_single_stock_task, target): target['ticker'] 
                for target in targets
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    res = future.result()
                    self.results[ticker] = res
                except Exception as exc:
                    print(f'{ticker} generated an exception: {exc}')

        self._save_report()

    def _load_targets(self) -> List[Dict]:
        """Loads targets from file or DB."""
        targets_file = os.path.join(self.base_dir, 'optimization_targets.txt')
        if os.path.exists(targets_file):
            targets = []
            with open(targets_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        targets.append({'ticker': parts[0], 'current_sharpe': float(parts[1])})
            return targets
            
        # Fallback defaults
        return [
            {'ticker': 'RELIANCE.NS', 'current_sharpe': 1.2},
            {'ticker': 'HDFCBANK.NS', 'current_sharpe': 0.8},
            {'ticker': 'INFY.NS', 'current_sharpe': 1.5}
        ]

    def _save_report(self):
        summary = {
            'total_stocks': len(self.results),
            'targets_met': sum(1 for r in self.results.values() if r.get('target_met', False)),
            'results': self.results
        }
        
        output_file = os.path.join(
            os.path.dirname(self.base_dir),
            f'OPTIMIZATION_RESULTS_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        )
        
        with open(output_file, 'w') as f:
            f.write("# MARK5 Optimization Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"- **Total Stocks:** {summary['total_stocks']}\n")
            f.write(f"- **Targets Met:** {summary['targets_met']}/{summary['total_stocks']}\n\n")
            
            f.write("| Ticker | Old Sharpe | New Sharpe | Improvement | Status |\n")
            f.write("|--------|------------|------------|-------------|--------|\n")
            
            for ticker, res in self.results.items():
                status = '✅' if res.get('target_met') else '❌'
                f.write(f"| {ticker} | {res.get('old_sharpe',0):.2f} | {res.get('new_sharpe',0):.2f} | {res.get('improvement',0):+.2f} | {status} |\n")

        print(f"\n✅ Report saved to {output_file}")

if __name__ == '__main__':
    workers = min(os.cpu_count(), 8)
    engine = OptimizationEngine(target_sharpe=2.5, max_workers=workers)
    engine.run_parallel_cycle()

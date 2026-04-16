import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

from core.data.data_pipeline import DataPipeline
from core.models.predictor import MARK5Predictor
from core.models.tcn.backtester import RobustBacktester
from core.models.registry import RobustModelRegistry

logger = logging.getLogger("MARK5.UniverseOptimizer")

class UniverseOptimizer:
    """
    Smart Universe Optimizer.
    Scans a broad universe, trains missing models in isolated subprocesses,
    runs backtests, and selects elite stocks (High Return, Positive Sharpe).
    """

    def __init__(self):
        self.pipeline = DataPipeline()
        self.registry = RobustModelRegistry()
        
        # Hardcoded fallback list if API fails
        self.fallback_universe = [
            'COFORGE.NS', 'HAL.NS', 'IDFCFIRSTB.NS', 'RELIANCE.NS', 'HDFCBANK.NS',
            'INFY.NS', 'TCS.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBIN.NS',
            'TATAMOTORS.NS', 'ITC.NS', 'LT.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS',
            'HINDUNILVR.NS', 'KOTAKBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
            'ULTRACEMCO.NS', 'TITAN.NS', 'ADANIENT.NS', 'TRENT.NS', 'DIXON.NS',
            'HINDALCO.NS', 'BAJAJ-AUTO.NS', 'M&M.NS', 'NTPC.NS', 'ONGC.NS'
        ]

    def get_candidate_universe(self, limit: int = 50) -> List[str]:
        """Fetches highly liquid candidate stocks from NSE Most Active."""
        logger.info("Fetching NSE Most Active stocks as candidates...")
        try:
            df = self.pipeline.get_most_active('NSE')
            if not df.empty and 'ticker' in df.columns:
                # E.g. 'RELIANCE' -> 'RELIANCE.NS'
                tickers = df['ticker'].tolist()
                valid_tickers = []
                for t in tickers:
                    if not str(t).endswith('.NS'):
                        t = f"{t}.NS"
                    valid_tickers.append(t)
                logger.info(f"Fetched {len(valid_tickers)} active tickers from ISE API.")
                return valid_tickers[:limit]
        except Exception as e:
            logger.warning(f"Failed to fetch active universe from ISE: {e}. Using fallback.")
            
        return self.fallback_universe[:limit]

    def optimize_universe(self, min_return_pct: float = 15.0, min_sharpe: float = 0.5, limit: int = 50, days: int = 365):
        """
        The main runner logic that trains, backtests, filters, and saves.
        """
        candidates = self.get_candidate_universe(limit)
        logger.info(f"Starting Smart Universe Optimization on {len(candidates)} candidates.")
        
        elite_stocks = []
        
        for ticker in candidates:
            logger.info(f"--- Optimizing {ticker} ---")
            
            # 1. Train missing models via subprocess to prevent memory leaks
            if not self._check_model_ready(ticker):
                logger.info(f"Model missing or stale for {ticker}. Spawning training process...")
                try:
                    cmd = [sys.executable, "core/models/training/trainer.py", "--symbols", ticker.replace('.NS', ''), "--years", "3"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"Training failed for {ticker}. Skipping.")
                        continue
                    else:
                        logger.info(f"Training successful for {ticker}.")
                except Exception as e:
                    logger.error(f"Failed to run subprocess for {ticker}: {e}")
                    continue
            
            # 2. Run Backtest
            try:
                metrics = self._run_backtest(ticker, days=days)
            except Exception as e:
                logger.error(f"Backtest failed for {ticker}: {e}")
                continue
                
            if "error" in metrics:
                logger.warning(f"Backtest error for {ticker}: {metrics['error']}")
                continue
                
            # 3. Apply Filters
            ret = metrics.get('Total Return %', 0.0)
            sharpe = metrics.get('Sharpe Ratio', 0.0)
            trades = metrics.get('Total Trades', 0)
            
            logger.info(f"Results for {ticker} - Return: {ret:.2f}%, Sharpe: {sharpe:.2f}, Trades: {trades}")
            
            if ret >= min_return_pct and sharpe >= min_sharpe and trades > 0:
                logger.info(f"⭐ {ticker} selected! Meets elite criteria.")
                elite_stocks.append(ticker)
            else:
                logger.info(f"❌ {ticker} rejected. Did not meet criteria.")

        # 4. Save winning symbols
        if not elite_stocks:
            logger.warning("No stocks met the elite criteria. Using default fallback to ensure system operates.")
            elite_stocks = self.fallback_universe[:5]
            
        self._save_universe(elite_stocks)
        logger.info(f"Optimization complete. Saved {len(elite_stocks)} elite stocks to configuration.")
        return elite_stocks

    def _check_model_ready(self, ticker: str) -> bool:
        try:
            predictor = MARK5Predictor(ticker, allow_shadow=True)
            if predictor._container is not None:
                return True
        except Exception as e:
            logger.warning(f"Model check failed for {ticker}: {e}")
        return False

    def _run_backtest(self, ticker: str, days: int) -> Dict[str, Any]:
        df = self.pipeline.get_market_data(ticker, source='kite', period='2y')
        if df.empty or len(df) < days + 250:
            return {"error": "Insufficient data"}
            
        predictor = MARK5Predictor(ticker, allow_shadow=True)
        if not predictor._container:
             return {"error": "No model container available."}
             
        buffer_days = 250
        test_df = df.tail(days + buffer_days)
        start_idx = len(test_df) - days

        full_signals = [0] * len(test_df)
        for i in range(start_idx, len(test_df)):
            subset = test_df.iloc[:i+1]
            res = predictor.predict(subset)
            sig = res.get('signal', 'HOLD')
            full_signals[i] = 1 if 'BUY' in sig else (-1 if 'SELL' in sig else 0)

        signals_series = pd.Series(full_signals, index=test_df.index)
        backtester = RobustBacktester(segment='EQUITY_INTRADAY')
        _, metrics = backtester.run_simulation(test_df, signals_series, symbol=ticker)
        return metrics

    def _save_universe(self, universe: List[str]):
        config_dir = Path(_PROJECT_ROOT) / "config"
        config_dir.mkdir(exist_ok=True, parents=True)
        file_path = config_dir / "universe.json"
        
        with open(file_path, "w") as f:
            json.dump({"active_universe": universe, "updated_at": datetime.now().isoformat()}, f, indent=4)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

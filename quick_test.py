import os
import sys
import pandas as pd
from core.data.data_pipeline import DataPipeline
from core.models.predictor import MARK5Predictor
from core.models.tcn.backtester import RobustBacktester
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def train_ticker(ticker):
    print(f"Training {ticker} with new Donchian logic...")
    cmd = [
        sys.executable, "core/models/training/trainer.py", 
        "--symbols", ticker.replace('.NS', ''), 
        "--years", "3", 
        "--cutoff", "2025-03-31"
    ]
    subprocess.run(cmd)

def test_ticker(ticker):
    print(f"Testing {ticker} OOS...")
    train_ticker(ticker)
    
    predictor = MARK5Predictor(ticker, allow_shadow=True)
    
    base_sym = ticker.replace('.NS', '')
    spot_path = f"data/cache/{base_sym}_NS_1d.parquet"
    if not os.path.exists(spot_path):
        spot_path = f"data/cache/{base_sym.replace('.', '_')}_NS_1d.parquet"
        
    df = pd.read_parquet(spot_path)
    
    oos_start = pd.Timestamp("2025-04-01")
    test_df = df[df.index >= oos_start - pd.Timedelta(days=365)].copy()
    
    full_signals = [0] * len(test_df)
    start_idx = test_df.index.get_indexer([oos_start], method='bfill')[0]
    
    for i in range(start_idx, len(test_df)):
        subset = test_df.iloc[:i+1]
        res = predictor.predict(subset)
        sig = res.get('signal', 'HOLD')
        full_signals[i] = 1 if 'BUY' in sig else (-1 if 'SELL' in sig else 0)

    signals_series = pd.Series(full_signals, index=test_df.index)
    
    backtester = RobustBacktester(segment='EQUITY_DELIVERY')
    _, metrics = backtester.run_simulation(test_df, signals_series, symbol=ticker)
    
    print(f"Results for {ticker}:")
    print(f"Return: {metrics.get('Total Return %', 0):.2f}%")
    print(f"Win Rate: {metrics.get('Win Rate %', 0):.2f}%")
    print(f"Trades: {metrics.get('Total Trades', 0)}")
    return metrics

if __name__ == "__main__":
    test_ticker("HCLTECH.NS")
    test_ticker("ABB.NS")

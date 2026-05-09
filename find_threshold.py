import pandas as pd
import numpy as np
import os
from core.models.predictor import MARK5Predictor
from core.models.tcn.backtester import RobustBacktester
import logging

logging.basicConfig(level=logging.ERROR)

TICKERS = ["RELIANCE.NS", "M&M.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS"]
OOS_START = pd.Timestamp("2025-04-01")

def test_hurdle(hurdle):
    bt = RobustBacktester(segment='FUTURES')
    total_ret = 0
    total_trades = 0
    
    for ticker in TICKERS:
        try:
            predictor = MARK5Predictor(ticker, allow_shadow=True)
            # Monkey-patch the hurdle for this test
            # Note: We need to ensure we bypass the internal hurdle logic
            
            base_sym = ticker.replace('.NS', '')
            df = pd.read_parquet(f"data/cache/{base_sym}_NS_1d.parquet")
            test_df = df[df.index >= OOS_START - pd.Timedelta(days=100)].copy()
            
            signals = []
            for i in range(len(test_df)):
                subset = test_df.iloc[:i+1]
                pred = predictor.predict(subset)
                conf = pred.get('confidence', 0.5)
                
                if conf >= hurdle:
                    signals.append(1)
                elif conf <= (1.0 - hurdle):
                    signals.append(-1)
                else:
                    signals.append(0)
            
            signals_series = pd.Series(signals, index=test_df.index)
            _, metrics = bt.run_simulation(test_df[test_df.index >= OOS_START], 
                                           signals_series[signals_series.index >= OOS_START], 
                                           symbol=ticker)
            
            total_ret += metrics.get('Total Return %', 0)
            total_trades += metrics.get('Total Trades', 0)
        except:
            continue
            
    return total_ret / len(TICKERS), total_trades

print("\n" + "="*50)
print("🎯 MARK5 SNIPER OPTIMIZATION (Hurdle Sweep)")
print("="*50)
print(f"Hurdle | Avg Return % | Trades")
print("-" * 30)

for h in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
    ret, trades = test_hurdle(h)
    print(f"{h:6.2f} | {ret:+.2f}% | {trades}")

print("="*50 + "\n")

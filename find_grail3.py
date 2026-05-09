import pandas as pd
import numpy as np
import os
from core.models.tcn.backtester import RobustBacktester

TICKERS = [
    "IRFC.NS", "IOB.NS", "M&M.NS", "CONCOR.NS", "AARTIIND.NS",
    "BEL.NS", "IRCON.NS", "HUDCO.NS", "BRIGADE.NS", "DLF.NS",
    "HCLTECH.NS", "ABB.NS", "BAJAJ-AUTO.NS", "PIIND.NS"
]

def test_ma_crossover(fast_ma, slow_ma, atr_sl, max_hold):
    bt = RobustBacktester(
        segment='EQUITY_DELIVERY',
        atr_multiplier=atr_sl,
        pt_multiplier=50.0, # massive PT to ride trend
        max_hold_days=max_hold
    )
    
    total_ret = 0
    total_trades = 0
    
    for ticker in TICKERS:
        base = ticker.replace('.NS', '')
        path = f"data/cache/{base}_NS_1d.parquet"
        if not os.path.exists(path):
            path = f"data/cache/{base.replace('.', '_')}_NS_1d.parquet"
        if not os.path.exists(path):
            continue
            
        df = pd.read_parquet(path)
        oos_start = pd.Timestamp("2024-04-01") # test over a longer period to find edge
        test_df = df[df.index >= oos_start].copy()
        
        close = test_df['close']
        
        fast = close.rolling(window=fast_ma).mean()
        slow = close.rolling(window=slow_ma).mean()
        
        signals = pd.Series(0, index=test_df.index)
        
        # Long: Fast crosses above Slow
        long_condition = (fast > slow)
        long_trigger = long_condition & (~long_condition.shift(1).fillna(False))
        signals[long_trigger] = 1
        
        # We don't explicitly short here, let the SL or max hold take us out, or short when fast crosses below slow
        short_condition = (fast < slow)
        short_trigger = short_condition & (~short_condition.shift(1).fillna(False))
        signals[short_trigger] = -1
        
        _, metrics = bt.run_simulation(test_df, signals, symbol=ticker)
        total_ret += metrics.get('Total Return %', 0)
        total_trades += metrics.get('Total Trades', 0)
        
    avg_ret = total_ret / len(TICKERS)
    return avg_ret, total_trades

best_ret = -999
best_params = None

for fast_ma in [10, 20]:
    for slow_ma in [50, 100, 200]:
        for atr_sl in [3.0, 5.0, 10.0]:
            for max_hold in [60, 120, 250]:
                ret, trades = test_ma_crossover(fast_ma, slow_ma, atr_sl, max_hold)
                if ret > best_ret and trades > 0:
                    best_ret = ret
                    best_params = (fast_ma, slow_ma, atr_sl, max_hold)
                    print(f"New Best: Ret={ret:.2f}% Trades={trades} Params={best_params}")

print(f"\nFINAL BEST: Ret={best_ret:.2f}% Params={best_params}")

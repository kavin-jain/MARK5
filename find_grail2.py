import pandas as pd
import numpy as np
import os
from core.models.tcn.backtester import RobustBacktester

TICKERS = [
    "IRFC.NS", "IOB.NS", "M&M.NS", "CONCOR.NS", "AARTIIND.NS",
    "BEL.NS", "IRCON.NS", "HUDCO.NS", "BRIGADE.NS", "DLF.NS"
]

def test_donchian(entry_window, exit_window, atr_sl):
    bt = RobustBacktester(
        segment='EQUITY_DELIVERY',
        atr_multiplier=atr_sl,
        pt_multiplier=10.0, # no PT, ride the trend
        max_hold_days=120
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
        high = test_df['high']
        low = test_df['low']
        
        rolling_high = high.rolling(window=entry_window).max().shift(1)
        rolling_low = low.rolling(window=exit_window).min().shift(1)
        
        signals = pd.Series(0, index=test_df.index)
        
        # Long: Breakout above entry_window high
        long_condition = (close > rolling_high)
        long_trigger = long_condition & (~long_condition.shift(1).fillna(False))
        signals[long_trigger] = 1
        
        # Sell: Close below exit_window low
        short_condition = (close < rolling_low)
        short_trigger = short_condition & (~short_condition.shift(1).fillna(False))
        signals[short_trigger] = -1 # to exit
        
        _, metrics = bt.run_simulation(test_df, signals, symbol=ticker)
        total_ret += metrics.get('Total Return %', 0)
        total_trades += metrics.get('Total Trades', 0)
        
    avg_ret = total_ret / len(TICKERS)
    return avg_ret, total_trades

best_ret = -999
best_params = None

for entry_w in [20, 50, 100]:
    for exit_w in [10, 20, 50]:
        for atr_sl in [2.0, 3.0, 5.0]:
            ret, trades = test_donchian(entry_w, exit_w, atr_sl)
            if ret > best_ret and trades > 0:
                best_ret = ret
                best_params = (entry_w, exit_w, atr_sl)
                print(f"New Best: Ret={ret:.2f}% Trades={trades} Params={best_params}")

print(f"\nFINAL BEST: Ret={best_ret:.2f}% Params={best_params}")

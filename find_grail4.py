import pandas as pd
import numpy as np
import os
from core.models.tcn.backtester import RobustBacktester

TICKERS = [
    "IRFC.NS", "IOB.NS", "M&M.NS", "CONCOR.NS", "AARTIIND.NS",
    "BEL.NS", "IRCON.NS", "HUDCO.NS", "BRIGADE.NS", "DLF.NS"
]

def test_mr(rsi_len, rsi_buy, rsi_sell, bb_std, atr_sl, atr_pt, max_hold):
    bt = RobustBacktester(
        segment='FUTURES',
        atr_multiplier=atr_sl,
        pt_multiplier=atr_pt,
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
        oos_start = pd.Timestamp("2024-04-01")
        test_df = df[df.index >= oos_start].copy()
        
        close = test_df['close']
        
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        upper_bb = sma + (bb_std * std)
        lower_bb = sma - (bb_std * std)
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=test_df.index)
        
        long_condition = (close < lower_bb) & (rsi < rsi_buy)
        long_trigger = long_condition & (~long_condition.shift(1).fillna(False))
        signals[long_trigger] = 1
        
        short_condition = (close > upper_bb) & (rsi > rsi_sell)
        short_trigger = short_condition & (~short_condition.shift(1).fillna(False))
        signals[short_trigger] = -1
        
        _, metrics = bt.run_simulation(test_df, signals, symbol=ticker)
        total_ret += metrics.get('Total Return %', 0)
        total_trades += metrics.get('Total Trades', 0)
        
    avg_ret = total_ret / len(TICKERS)
    return avg_ret, total_trades

best_ret = -999
best_params = None

for rsi_buy in [25, 30, 40]:
    for rsi_sell in [60, 70, 75]:
        for bb_std in [1.5, 2.0]:
            for atr_sl in [2.0, 3.0, 5.0]:
                for atr_pt in [4.0, 6.0]:
                    ret, trades = test_mr(14, rsi_buy, rsi_sell, bb_std, atr_sl, atr_pt, 15)
                    if ret > best_ret and trades > 0:
                        best_ret = ret
                        best_params = (rsi_buy, rsi_sell, bb_std, atr_sl, atr_pt)
                        print(f"New Best: Ret={ret:.2f}% Trades={trades} Params={best_params}")

print(f"\nFINAL BEST: Ret={best_ret:.2f}% Params={best_params}")

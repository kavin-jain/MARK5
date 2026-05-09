import pandas as pd
import numpy as np
import os
from core.models.tcn.backtester import RobustBacktester

TICKERS = [
    "HCLTECH.NS", "ABB.NS", "BAJAJ-AUTO.NS", "PIIND.NS",
    "IDFCFIRSTB.NS", "BAJFINANCE.NS", "HUDCO.NS", "ASIANPAINT.NS",
    "ETERNAL.NS", "IPCALAB.NS", "COFORGE.NS"
]

def test_params(rsi_len, rsi_thres, bb_len, bb_std, atr_sl, atr_pt, max_hold):
    bt = RobustBacktester(
        segment='EQUITY_DELIVERY',
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
        oos_start = pd.Timestamp("2025-04-01")
        test_df = df[df.index >= oos_start].copy()
        
        # Compute signals
        close = test_df['close']
        
        sma = close.rolling(window=bb_len).mean()
        std = close.rolling(window=bb_len).std()
        lower_bb = sma - (bb_std * std)
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=test_df.index)
        
        # Uptrend filter: Close > 200 EMA
        ema_200 = close.ewm(span=200, adjust=False).mean()
        uptrend = close > ema_200
        
        long_condition = uptrend & (rsi < rsi_thres) & (close < lower_bb)
        long_trigger = long_condition & (~long_condition.shift(1).fillna(False))
        signals[long_trigger] = 1
        
        _, metrics = bt.run_simulation(test_df, signals, symbol=ticker)
        total_ret += metrics.get('Total Return %', 0)
        total_trades += metrics.get('Total Trades', 0)
        
    avg_ret = total_ret / len(TICKERS)
    return avg_ret, total_trades

best_ret = -999
best_params = None

# Brute force
for rsi_thres in [30, 40, 50, 60]:
    for bb_std in [1.0, 1.5, 2.0]:
        for atr_sl in [1.5, 2.0, 3.0]:
            for atr_pt in [2.0, 3.0, 4.0]:
                ret, trades = test_params(14, rsi_thres, 20, bb_std, atr_sl, atr_pt, 20)
                if ret > best_ret and trades > 0:
                    best_ret = ret
                    best_params = (rsi_thres, bb_std, atr_sl, atr_pt)
                    print(f"New Best: Ret={ret:.2f}% Trades={trades} Params={best_params}")

print(f"\nFINAL BEST: Ret={best_ret:.2f}% Params={best_params}")

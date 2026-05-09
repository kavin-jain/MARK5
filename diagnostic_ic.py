import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MARK5.ICAudit")

TICKERS = [
    "IRFC.NS", "IOB.NS", "M&M.NS", "CONCOR.NS", "AARTIIND.NS",
    "BEL.NS", "IRCON.NS", "HUDCO.NS", "BRIGADE.NS", "DLF.NS"
]

def calculate_ic(series, target):
    mask = ~series.isna() & ~target.isna()
    if mask.sum() < 30: return 0.0
    ic, _ = spearmanr(series[mask], target[mask])
    return ic

def run_audit():
    all_ic_results = []
    
    for ticker in TICKERS:
        logger.info(f"Auditing {ticker}...")
        base_sym = ticker.replace('.NS', '')
        path = f"data/cache/{base_sym}_NS_1d.parquet"
        if not os.path.exists(path):
            path = f"data/cache/{base_sym.replace('.', '_')}_NS_1d.parquet"
        if not os.path.exists(path): continue
        
        df = pd.read_parquet(path)
        # Target: 10-day forward return
        df['target_10d'] = df['close'].pct_change(10).shift(-10)
        
        candidates = {}
        c = df['close']
        h = df['high']
        l = df['low']
        v = df['volume']
        
        # --- MOMENTUM ---
        candidates['rsi_14'] = (c.diff().clip(lower=0).rolling(14).mean() / (c.diff().abs().rolling(14).mean() + 1e-9))
        candidates['roc_10'] = c.pct_change(10)
        candidates['williams_r'] = (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min() + 1e-9)
        
        # --- VOLATILITY ---
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        candidates['bb_width'] = (h.rolling(20).max() - l.rolling(20).min()) / c
        candidates['atr_vol'] = atr / c
        candidates['vol_zscore'] = (v - v.rolling(60).mean()) / v.rolling(60).std()
        
        # --- MICROSTRUCTURE ---
        candidates['amihud'] = c.pct_change().abs() / (v * c + 1e-9)
        candidates['gap'] = (df['open'] - c.shift(1)) / c.shift(1)
        candidates['range_z'] = (h - l) / c
        
        # --- TREND ---
        candidates['dist_ma50'] = (c - c.rolling(50).mean()) / c.rolling(50).mean()
        candidates['dist_ma200'] = (c - c.rolling(200).mean()) / c.rolling(200).mean()
        
        ticker_ics = {}
        for name, series in candidates.items():
            ticker_ics[name] = calculate_ic(series, df['target_10d'])
            
        all_ic_results.append(ticker_ics)
        
    ic_df = pd.DataFrame(all_ic_results)
    mean_ic = ic_df.mean().sort_values(key=abs, ascending=False)
    
    print("\n" + "="*50)
    print("💎 GOLDEN FEATURE AUDIT (Information Coefficients)")
    print("="*50)
    for name, val in mean_ic.items():
        print(f"{name:15} | IC: {val:+.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_audit()

"""
MARK6 Indicator Brainstorming & IC Audit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script implements new candidate indicators and evaluates their 
Information Coefficient (IC) against 10-day forward returns.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MARK6.Brainstorm")

TICKERS = [
    "IRFC.NS", "IOB.NS", "M&M.NS", "CONCOR.NS", "AARTIIND.NS",
    "BEL.NS", "IRCON.NS", "HUDCO.NS", "BRIGADE.NS", "DLF.NS",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AXISBANK.NS"
]

def calculate_ic(series, target):
    mask = ~series.isna() & ~target.isna() & np.isfinite(series) & np.isfinite(target)
    if mask.sum() < 60: return 0.0
    ic, _ = spearmanr(series[mask], target[mask])
    return ic

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def run_brainstorm():
    all_ic_results = []
    
    # Load Nifty 50 for Relative Strength
    nifty_path = "data/cache/NIFTY50_1d.parquet"
    if os.path.exists(nifty_path):
        nifty_df = pd.read_parquet(nifty_path)
        nifty_ret = nifty_df['close'].pct_change()
    else:
        logger.warning("NIFTY50_1d.parquet not found. RS will be skipped.")
        nifty_ret = None

    for ticker in TICKERS:
        base_sym = ticker.replace('.NS', '')
        path = f"data/cache/{base_sym}_NS_1d.parquet"
        if not os.path.exists(path):
            path = f"data/cache/{base_sym.replace('.', '_')}_NS_1d.parquet"
        if not os.path.exists(path): continue
        
        df = pd.read_parquet(path)
        df['target_10d'] = df['close'].pct_change(10).shift(-10)
        
        candidates = {}
        c = df['close']
        h = df['high']
        l = df['low']
        v = df['volume']
        tp = (h + l + c) / 3
        
        # --- BRAINSTORMED INDICATORS ---
        
        # 1. Relative Strength vs Nifty
        if nifty_ret is not None:
            stock_ret = c.pct_change(10)
            n_ret = nifty_ret.reindex(c.index).pct_change(10)
            candidates['rel_strength'] = stock_ret - n_ret
            
        # 2. Volume-Weighted RSI (VWRSI)
        v_norm = v / v.rolling(20).mean()
        delta = c.diff()
        gain = (delta.clip(lower=0) * v_norm).rolling(14).mean()
        loss = ((-delta).clip(lower=0) * v_norm).rolling(14).mean()
        candidates['vwrsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        # 3. MFI Divergence Proxy
        mfi_period = 14
        mf = tp * v
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(mfi_period).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(mfi_period).sum()
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))
        candidates['mfi_div'] = c.pct_change(mfi_period) - mfi.pct_change(mfi_period)

        # 4. Trend Intensity Index (TII)
        ma_60 = c.rolling(60).mean()
        dev = c - ma_60
        pos_dev = dev.clip(lower=0).rolling(60).sum()
        neg_dev = (-dev).clip(lower=0).rolling(60).sum()
        candidates['tii_60'] = 100 * (pos_dev / (pos_dev + neg_dev + 1e-9))
        
        # 5. Volatility-Adjusted Momentum
        candidates['vol_adj_mom'] = c.pct_change(20) / (c.pct_change().rolling(20).std() * np.sqrt(20) + 1e-9)

        # --- BASELINE GOLDEN 8 (for comparison) ---
        candidates['amihud'] = c.pct_change().abs() / (v * c + 1e-9)
        candidates['bb_width'] = (h.rolling(20).max() - l.rolling(20).min()) / (c + 1e-9)
        candidates['atr_vol'] = (pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1).rolling(14).mean()) / (c + 1e-9)
        
        ticker_ics = {}
        for name, series in candidates.items():
            ticker_ics[name] = calculate_ic(series, df['target_10d'])
            
        all_ic_results.append(ticker_ics)
        
    ic_df = pd.DataFrame(all_ic_results)
    mean_ic = ic_df.mean().sort_values(key=abs, ascending=False)
    
    print("\n" + "="*60)
    print("🚀 MARK6 BRAINSTORMED FEATURE AUDIT (Information Coefficients)")
    print("="*60)
    for name, val in mean_ic.items():
        print(f"{name:20} | IC: {val:+.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_brainstorm()

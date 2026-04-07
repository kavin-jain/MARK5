import pandas as pd
import numpy as np
from pathlib import Path
import json

# Survivors from user summary
SURVIVORS = [
    "HINDALCO.NS", "BSE.NS", "ADANIPORTS.NS", "COFORGE.NS", "MCX.NS", 
    "RECLTD.NS", "APOLLOHOSP.NS", "MARICO.NS", "SHRIRAMFIN.NS", 
    "MUTHOOTFIN.NS", "TORNTPHARM.NS", "BAJAJFINSV.NS", "AUBANK.NS", 
    "NTPC.NS", "SBIN.NS", "BAJAJ-AUTO.NS", "LT.NS", "JSWSTEEL.NS"
]

CACHE_DIR = Path("data/cache")

def run_correlation_audit():
    print("="*60)
    print("MARK5 SURVIVOR CORRELATION AUDIT (Rule 20)")
    print("="*60)
    
    price_data = {}
    for sym in SURVIVORS:
        f = CACHE_DIR / f"{sym.replace('.NS', '')}_60m.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            # Use daily resampling for 60-day rolling calculation (Rule 20)
            daily = df['close'].resample('D').last().dropna()
            price_data[sym] = daily
            
    if not price_data:
        print("No cache data found for survivors.")
        return

    df_prices = pd.DataFrame(price_data).ffill()
    df_returns = df_prices.pct_change().dropna()
    
    # Calculate 60-session rolling correlation (using last 250 days as sample)
    corr_matrix = df_returns.tail(250).corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            c1 = corr_matrix.columns[i]
            c2 = corr_matrix.columns[j]
            val = corr_matrix.iloc[i, j]
            if val > 0.70:
                high_corr_pairs.append((c1, c2, val))
                
    if not high_corr_pairs:
        print("✅ Rule 20 Passed: No pairs exceed 0.70 correlation.")
    else:
        print(f"⚠️  Rule 20 Violation: Found {len(high_corr_pairs)} high-correlation pairs (>0.70)")
        print("-" * 60)
        sorted_pairs = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)
        for c1, c2, val in sorted_pairs:
            print(f"{c1:15} <-> {c2:15} | Correlation: {val:.3f}")
    
    print("="*60)

if __name__ == "__main__":
    run_correlation_audit()

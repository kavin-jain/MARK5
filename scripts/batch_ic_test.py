import pandas as pd
import numpy as np
import logging
import argparse
import time
import matplotlib.pyplot as plt

# Import your working IC function
from scripts.feature_ic import calculate_feature_ic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MARK5.BatchIC")

# A stratified 20-stock universe covering every major Indian sector
STRATIFIED_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "SBIN.NS", "TATAMOTORS.NS",
    "SUNPHARMA.NS", "HINDUNILVR.NS", "BHARTIARTL.NS", "ADANIENT.NS", "ICICIBANK.NS",
    "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS", "ASIANPAINT.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "TRENT.NS", "DIXON.NS"
]

def run_batch_ic(universe=STRATIFIED_UNIVERSE):
    logger.info(f"Initiating Batch IC Analysis across {len(universe)} sectors...")
    
    all_results = []
    
    for symbol in universe:
        try:
            # We already hardcoded EXPECTED_FEATURE_COUNT = 8 in your engine
            res_df = calculate_feature_ic(symbol, top_n=8)
            res_df['symbol'] = symbol
            all_results.append(res_df)
            
            # Close plots to prevent memory leak/noise in batch runs
            plt.close('all')
            
            time.sleep(1) # Be polite to Yahoo Finance API
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {str(e)}")
            
    if not all_results:
        logger.error("All symbols failed. Exiting.")
        return
        
    # Combine all individual reports
    master_df = pd.concat(all_results, ignore_index=True)
    
    # ── Calculate Universe Mean IC and IC Information Ratio (ICIR) ──
    summary = master_df.groupby('feature').agg(
        mean_ic=('ic_spearman', 'mean'),
        std_ic=('ic_spearman', 'std'),
        mean_mi=('mi_score', 'mean'),
        win_rate_p_val=('p_value', lambda x: (x < 0.05).mean() * 100) # % of stocks where feature is statistically significant
    ).reset_index()
    
    # IC Information Ratio = Mean / StdDev. Higher is more consistent.
    summary['icir'] = summary['mean_ic'] / (summary['std_ic'] + 1e-9)
    
    # Sort by the most consistent predictive power
    summary = summary.sort_values(by='mean_ic', ascending=False).round(4)
    
    print("\n" + "="*80)
    print(" 🚀 MARK5 UNIVERSE FEATURE VALIDATION REPORT (10 SECTORS)")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
    
    # Save for documentation
    if not os.path.exists("reports"): os.makedirs("reports")
    summary.to_csv("reports/universe_feature_ic_summary.csv", index=False)
    logger.info("Saved final Universe IC Summary to reports/universe_feature_ic_summary.csv")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run on full 105 stock universe (Takes time)")
    args = parser.parse_args()
    
    # If you want to run all 105 stocks later, you can pass a full list here
    run_batch_ic()

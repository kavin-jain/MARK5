import os
import sys
import pandas as pd
import numpy as np
from core.models.predictor import MARK5Predictor
from core.models.training.financial_engineer import FinancialEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MARK5.Diagnostic")

# Use two of our most liquid stocks for the test
TICKERS = ["M&M.NS", "RELIANCE.NS"]
OOS_START = pd.Timestamp("2025-04-01")

def run_diagnostic():
    results = []
    
    for ticker in TICKERS:
        logger.info(f"🔍 Running Separation Diagnostic for {ticker}...")
        
        # 1. Load Predictor (Ensuring we use the trained model)
        try:
            predictor = MARK5Predictor(ticker, allow_shadow=True)
        except Exception as e:
            logger.error(f"Failed to load predictor for {ticker}: {e}")
            continue
            
        # 2. Load Data
        base_sym = ticker.replace('.NS', '')
        path = f"data/cache/{base_sym}_NS_1d.parquet"
        if not os.path.exists(path):
            path = f"data/cache/{base_sym.replace('.', '_')}_NS_1d.parquet"
        
        df = pd.read_parquet(path)
        # We need a buffer for features
        test_df = df[df.index >= OOS_START - pd.Timedelta(days=365)].copy()
        
        # 3. Find all primary signals in OOS
        fe = FinancialEngineer()
        primary_signals = fe.get_primary_signals(test_df)
        oos_signals = primary_signals[primary_signals.index >= OOS_START]
        active_events = oos_signals[oos_signals != 0]
        
        logger.info(f"Found {len(active_events)} primary signals in OOS period.")
        
        # 4. Analyze each event
        for timestamp in active_events.index:
            # Subset data up to this point
            idx = test_df.index.get_loc(timestamp)
            subset = test_df.iloc[:idx+1]
            
            # Get ML Confidence
            pred = predictor.predict(subset)
            conf = pred.get('confidence', 0.0)
            
            # Get actual outcome (10-day forward return)
            # Find the index 10 bars ahead
            try:
                future_idx = idx + 10
                if future_idx < len(test_df):
                    p_start = test_df['close'].iloc[idx]
                    p_end = test_df['close'].iloc[future_idx]
                    actual_ret = (p_end / p_start - 1) * active_events.loc[timestamp]
                else:
                    actual_ret = np.nan
            except:
                actual_ret = np.nan
                
            results.append({
                "ticker": ticker,
                "timestamp": timestamp,
                "signal": active_events.loc[timestamp],
                "ml_confidence": conf,
                "actual_10d_ret": actual_ret
            })

    # 5. Statistical Analysis
    diag_df = pd.DataFrame(results).dropna()
    if diag_df.empty:
        print("❌ No valid OOS events found to analyze.")
        return

    # Calculate separation: Average return of High Confidence (>0.55) vs Low Confidence (<0.45)
    high_conf = diag_df[diag_df['ml_confidence'] >= 0.55]
    low_conf = diag_df[diag_df['ml_confidence'] <= 0.45]
    mid_conf = diag_df[(diag_df['ml_confidence'] > 0.45) & (diag_df['ml_confidence'] < 0.55)]

    print("\n" + "="*50)
    print("📈 MARK5 SEPARATION DIAGNOSTIC (OOS)")
    print("="*50)
    print(f"Group | Count | Avg 10d Return")
    print(f"High Conf (>=55%) | {len(high_conf)} | {high_conf['actual_10d_ret'].mean()*100:+.2f}%")
    print(f"Mid Conf          | {len(mid_conf)} | {mid_conf['actual_10d_ret'].mean()*100:+.2f}%")
    print(f"Low Conf  (<=45%) | {len(low_conf)} | {low_conf['actual_10d_ret'].mean()*100:+.2f}%")
    print("="*50)
    
    correlation = diag_df['ml_confidence'].corr(diag_df['actual_10d_ret'])
    print(f"Correlation (Conf vs Ret): {correlation:.4f}")
    
    if correlation > 0.1:
        print("✅ VERDICT: ML Brain has significant separation edge.")
    elif correlation > 0:
        print("🟡 VERDICT: ML Brain has weak separation edge.")
    else:
        print("❌ VERDICT: ML Brain has NO edge (Random noise).")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_diagnostic()

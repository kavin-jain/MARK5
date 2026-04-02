import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Add root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.downloader import YFinanceDownloader
from core.models.features import AdvancedFeatureEngine

def test_frac_diff_d(symbol, start_date="2020-01-01"):
    downloader = YFinanceDownloader()
    data = downloader.download_historical_data([symbol], interval='1d', period='5y')
    
    if symbol not in data:
        print(f"Failed to download data for {symbol}")
        return
        
    df = data[symbol]
    close = df['close']
    
    fe = AdvancedFeatureEngine()
    
    results = []
    
    # Test d values from 0.0 to 1.0 (0 = no diff, 1 = standard return)
    d_vals = np.linspace(0.0, 1.0, 21)
    
    for d in d_vals:
        diffed = fe._frac_diff_ffd(close, d=d, thres=1e-3) # faster computation
        diffed = diffed.dropna()
        diffed = diffed[diffed != 0.0] # Remove 0s from padding
        
        if len(diffed) > 100:
            # Calculate ADF p-value (stationarity)
            try:
                adf_stat, p_value, _, _, _, _ = adfuller(diffed, maxlag=1, regression='c', autolag=None)
            except Exception as e:
                p_value = 1.0
                
            # Calculate memory (correlation to original price)
            # Align indices
            aligned = pd.concat([close, diffed], axis=1).dropna()
            corr = np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
            
            results.append({
                'd': d,
                'p_value': p_value,
                'correlation': corr,
                'stationary': p_value < 0.05
            })
            print(f"d={d:.2f} | ADF p-value={p_value:.4f} | Memory Corr={corr:.4f}")
            
    res_df = pd.DataFrame(results)
    print("\nOptimal d ranges (Stationary & Highest Memory):")
    valid = res_df[res_df['stationary'] == True]
    if not valid.empty:
        best = valid.sort_values(by='correlation', ascending=False).iloc[0]
        print(best)
    else:
        print("No d value achieved stationarity (p < 0.05).")

if __name__ == "__main__":
    test_frac_diff_d("RELIANCE.NS")

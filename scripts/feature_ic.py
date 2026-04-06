import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import yfinance as yf
from datetime import datetime
import logging
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MARK5.FeatureIC")

# Local imports
from core.data.adapters.kite_adapter import KiteFeedAdapter
from core.models.features import AdvancedFeatureEngine
from core.models.training.financial_engineer import FinancialEngineer
from dotenv import load_dotenv

# Load .env
load_dotenv()

def calculate_feature_ic(symbol, top_n=20):
    logger.info(f"🚀 Calculating 60m Feature IC for {symbol}...")
    
    # Load data via Kite
    config = {
        "api_key": os.getenv("KITE_API_KEY"),
        "api_secret": os.getenv("KITE_API_SECRET"),
        "access_token": os.getenv("KITE_ACCESS_TOKEN")
    }
    adapter = KiteFeedAdapter(config)
    if not adapter.connect():
        logger.error("Kite connection failed.")
        return pd.DataFrame()
        
    df = adapter.fetch_ohlcv(symbol, period="3y", interval="60m")
    adapter.disconnect()
    
    if df.empty:
        logger.error(f"No data for {symbol}")
        return pd.DataFrame()
    
    # 1. Generate Features
    logger.info("  Generating features...")
    fe_engine = AdvancedFeatureEngine()
    features_df = fe_engine.engineer_all_features(df)
    
    # 2. Generate Labels (ALIGNED WITH 60m UPGRADE)
    logger.info("  Generating labels (70-bar horizon)...")
    fe = FinancialEngineer()
    labels_df = fe.get_labels(df, run_bars=70, pt_sl=[2.5, 2.0])
    
    # Check if a continuous return column exists for accurate Spearman IC
    continuous_target_col = None
    for candidate in ['ret', 'fwd_ret', 'max_ret']:
        if candidate in labels_df.columns:
            continuous_target_col = candidate
            break
            
    if not continuous_target_col:
        logger.warning("No continuous return column found in labels! Spearman IC will act as Point-Biserial. Defaulting to 'bin'.")
        continuous_target_col = 'bin'

    # 3. Align and Join
    # Ensure both are timezone-naive to match AdvancedFeatureEngine logic
    if features_df.index.tz is not None:
        features_df.index = features_df.index.tz_localize(None)
    if labels_df.index.tz is not None:
        labels_df.index = labels_df.index.tz_localize(None)
        
    aligned_df = features_df.join(labels_df[['bin', continuous_target_col]], how='inner').dropna()
    
    # Exclude non-feature columns
    exclude = ['date', 'open', 'high', 'low', 'close', 'volume', 'bin', continuous_target_col]
    feature_cols = [c for c in aligned_df.columns if c not in exclude]
    
    # --- FIX: ROBUST DATA CLEANING ---
    # Replace inf with NaN, forward-fill valid historical data, then fill any remaining with 0
    X = aligned_df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    y_binary = aligned_df['bin']
    y_continuous = aligned_df[continuous_target_col]
    
    logger.info(f"  Calculating IC for {len(feature_cols)} features over {len(aligned_df)} samples...")
    
    results = []
    
    # 3. Calculate Mutual Information (against BINARY target to see non-linear classification power)
    mi_scores = mutual_info_classif(X, y_binary, discrete_features=False)
    
    for i, col in enumerate(feature_cols):
        feature_data = X[col]
        
        # Spearman correlation (Information Coefficient) against CONTINUOUS target
        ic_corr, p_val = stats.spearmanr(feature_data, y_continuous)
        
        # Pearson correlation
        pearson_corr, _ = stats.pearsonr(feature_data, y_continuous)
        
        # --- FIX: DIRECTIONALITY PRESERVED ---
        raw_ic = ic_corr if not np.isnan(ic_corr) else 0
        
        results.append({
            'feature': col,
            'ic_spearman': raw_ic,                 # Retains direction (+ or -)
            'abs_ic': abs(raw_ic),                 # Used strictly for ranking magnitude
            'pearson': pearson_corr if not np.isnan(pearson_corr) else 0,
            'p_value': p_val if not np.isnan(p_val) else 1,
            'mi_score': mi_scores[i]
        })
    
    # Sort by absolute IC to find the strongest predictors (regardless of direction)
    res_df = pd.DataFrame(results).sort_values('abs_ic', ascending=False)
    
    # Log Top N
    logger.info(f"\n📈 TOP {top_n} FEATURES BY MAGNITUDE (Spearman IC):\n{res_df.head(top_n)[['feature', 'ic_spearman', 'abs_ic', 'mi_score', 'p_value']]}")
    
    # 4. Plot Top N
    plt.figure(figsize=(12, 8))
    top_df = res_df.head(top_n)
    
    # Plotting using the absolute IC for magnitude ranking
    sns.barplot(x='abs_ic', y='feature', data=top_df, palette='viridis', hue='feature', legend=False)
    plt.axvline(x=0.02, color='r', linestyle='--', label='Min Threshold (|0.02|)')
    plt.title(f"Top {top_n} Features by IC Magnitude (Spearman) - {symbol}")
    plt.xlabel("Absolute Spearman Correlation |IC|")
    plt.legend()
    
    if not os.path.exists("reports"): 
        os.makedirs("reports")
        
    plt.savefig(f"reports/feature_ic_{symbol}.png", bbox_inches='tight')
    plt.close() # Free memory
    
    # 5. Save CSV for audit
    res_df.to_csv(f"reports/feature_ic_{symbol}.csv", index=False)
    logger.info(f"📊 Full IC report saved to reports/feature_ic_{symbol}.csv")
    
    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Feature IC for a symbol")
    parser.add_argument("--symbol", default="SBIN.NS", help="Symbol to analyze")
    args = parser.parse_args()
    
    res = calculate_feature_ic(args.symbol)
    if not res.empty:
        print(f"\n💾 Execution complete. Full IC report saved to reports/feature_ic_{args.symbol}.csv")
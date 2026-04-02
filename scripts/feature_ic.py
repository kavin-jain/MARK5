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

from core.models.features import AdvancedFeatureEngine
from core.models.training.financial_engineer import FinancialEngineer

def calculate_feature_ic(symbol, top_n=20):
    logger.info(f"🚀 Calculating Feature IC for {symbol}...")
    
    # Load data
    df = yf.download(symbol, period="3y", interval="1d")
    
    # Handle MultiIndex columns if present (yfinance >= 0.2.x)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Generate Features
    logger.info("  Generating features...")
    fe_engine = AdvancedFeatureEngine()
    features_df = fe_engine.engineer_all_features(df)
    
    # 2. Generate Labels
    logger.info("  Generating labels...")
    fe = FinancialEngineer()
    labels_df = fe.get_labels(df, pt_multiplier=1.5, sl_multiplier=1.0, max_holding=20)
    
    # Join
    aligned_df = features_df.join(labels_df[['bin']], how='inner').dropna()
    
    # Exclude non-feature columns
    exclude = ['date', 'open', 'high', 'low', 'close', 'volume', 'bin']
    feature_cols = [c for c in aligned_df.columns if c not in exclude]
    
    X = aligned_df[feature_cols]
    y = aligned_df['bin']
    
    logger.info(f"  Calculating IC for {len(feature_cols)} features over {len(aligned_df)} samples...")
    
    results = []
    
    # 3. Calculate IC (Spearman correlation) and Mutual Information
    mi_scores = mutual_info_classif(X.replace([np.inf, -np.inf], 0).fillna(0), y, discrete_features=False)
    
    for i, col in enumerate(feature_cols):
        # Spearman correlation (Information Coefficient)
        ic_corr, p_val = stats.spearmanr(X[col], y)
        
        # Pearson correlation (for linear relationship)
        pearson_corr, _ = stats.pearsonr(X[col], y)
        
        results.append({
            'feature': col,
            'ic_spearman': abs(ic_corr),
            'pearson': pearson_corr,
            'p_value': p_val,
            'mi_score': mi_scores[i]
        })
    
    res_df = pd.DataFrame(results).sort_values('ic_spearman', ascending=False)
    
    # Log Top N
    logger.info(f"\n📈 TOP {top_n} FEATURES BY SPEARMAN IC:\n{res_df.head(top_n)[['feature', 'ic_spearman', 'mi_score', 'p_value']]}")
    
    # 4. Plot Top N
    plt.figure(figsize=(12, 8))
    top_df = res_df.head(top_n)
    sns.barplot(x='ic_spearman', y='feature', data=top_df, palette='viridis')
    plt.axvline(x=0.02, color='r', linestyle='--', label='Min Threshold (0.02)')
    plt.title(f"Top {top_n} Features by Information Coefficient (Spearman) - {symbol}")
    plt.xlabel("Absolute Spearman Correlation")
    plt.legend()
    
    if not os.path.exists("reports"): os.makedirs("reports")
    plt.savefig(f"reports/feature_ic_{symbol}.png")
    
    # 5. Save CSV for audit
    res_df.to_csv(f"reports/feature_ic_{symbol}.csv", index=False)
    logger.info(f"📊 Full IC report saved to reports/feature_ic_{symbol}.csv")
    
    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="SBIN.NS")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()
    
    calculate_feature_ic(args.symbol, args.top_n)

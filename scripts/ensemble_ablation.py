import pandas as pd
import numpy as np
import argparse
import os
import yfinance as yf
import logging
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MARK5.EnsembleAudit")

from core.models.features import AdvancedFeatureEngine
from core.models.training.financial_engineer import FinancialEngineer

def run_ablation_audit(symbol):
    logger.info(f"🚀 Starting Ensemble Ablation Audit for {symbol}...")
    
    # 1. Prepare Data
    df = yf.download(symbol, period="4y", interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    
    fe_engine = AdvancedFeatureEngine()
    features_df = fe_engine.engineer_all_features(df)
    
    fe = FinancialEngineer()
    labels_df = fe.get_labels(df, pt_multiplier=1.5, sl_multiplier=1.0, max_holding=20)
    
    # Meta-label (binary) and Raw return (regression)
    aligned_df = features_df.join(labels_df[['bin', 'ret']], how='inner').dropna()
    
    X = aligned_df.drop(columns=['bin', 'ret'])
    y_bin = aligned_df['bin']
    y_ret = aligned_df['ret']
    
    # Time Series Split (Walk-Forward)
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_bin, y_test_bin = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
        y_train_ret, y_test_ret = y_ret.iloc[train_idx], y_ret.iloc[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        fold_res = {'fold': fold}
        
        # --- PART A: CLASSIFICATION ABLATION ---
        # XGBoost
        xgb_c = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42)
        xgb_c.fit(X_train_scaled, y_train_bin)
        fold_res['xgb_auc'] = roc_auc_score(y_test_bin, xgb_c.predict_proba(X_test_scaled)[:, 1])
        
        # LightGBM
        lgb_c = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
        lgb_c.fit(X_train_scaled, y_train_bin)
        fold_res['lgb_auc'] = roc_auc_score(y_test_bin, lgb_c.predict_proba(X_test_scaled)[:, 1])
        
        # RandomForest
        rf_c = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
        rf_c.fit(X_train_scaled, y_train_bin)
        fold_res['rf_auc'] = roc_auc_score(y_test_bin, rf_c.predict_proba(X_test_scaled)[:, 1])
        
        # Simple Ensemble (Mean of Top 2)
        aucs = {'xgb': fold_res['xgb_auc'], 'lgb': fold_res['lgb_auc'], 'rf': fold_res['rf_auc']}
        top_2 = sorted(aucs, key=aucs.get, reverse=True)[:2]
        
        preds = []
        if 'xgb' in top_2: preds.append(xgb_c.predict_proba(X_test_scaled)[:, 1])
        if 'lgb' in top_2: preds.append(lgb_c.predict_proba(X_test_scaled)[:, 1])
        if 'rf' in top_2: preds.append(rf_c.predict_proba(X_test_scaled)[:, 1])
        
        ens_pred = np.mean(preds, axis=0)
        fold_res['top2_ens_auc'] = roc_auc_score(y_test_bin, ens_pred)
        
        # --- PART B: REGRESSION VS CLASSIFICATION ---
        # Train Regressor (predict raw returns)
        xgb_r = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42)
        xgb_r.fit(X_train_scaled, y_train_ret)
        reg_preds = xgb_r.predict(X_test_scaled)
        
        # Convert regression output to "probability of PT hit" proxy using correlation
        # or just check AUC if we treat reg_preds as a score
        fold_res['reg_as_clf_auc'] = roc_auc_score(y_test_bin, reg_preds)
        
        results.append(fold_res)
    
    res_df = pd.DataFrame(results)
    avg_res = res_df.mean()
    
    logger.info("\n" + "="*50)
    logger.info(f"ENSEMBLE ABLATION RESULTS: {symbol}")
    logger.info("="*50)
    logger.info(f"Avg XGB AUC:        {avg_res['xgb_auc']:.3f}")
    logger.info(f"Avg LGB AUC:        {avg_res['lgb_auc']:.3f}")
    logger.info(f"Avg RF AUC:         {avg_res['rf_auc']:.3f}")
    logger.info(f"Avg Top-2 Ens AUC:  {avg_res['top2_ens_auc']:.3f}")
    logger.info(f"Avg Reg-as-Clf AUC: {avg_res['reg_as_clf_auc']:.3f}")
    logger.info("="*50)
    
    if avg_res['reg_as_clf_auc'] > avg_res['top2_ens_auc']:
        logger.warning("💡 Regression significantly outperforms Classification for this target!")
    
    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="SBIN.NS")
    args = parser.parse_args()
    
    run_ablation_audit(args.symbol)

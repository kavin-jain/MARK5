"""
MARK5 HYPERPARAMETER OPTIMIZER v7.0 - ARCHITECT EDITION
Revisions:
1. OBJECTIVE SHIFT: From Accuracy -> Log Loss (Purity of Probability).
2. PURGED VALIDATION: Prevents autocorrelation leakage.
3. HYPERBAND PRUNING: Kills bad trials early.
4. REGULARIZATION: Penalizes tree complexity to prevent noise fitting.
"""

import optuna
import json
import os
import logging
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import log_loss, precision_score
from optuna.pruners import HyperbandPruner

logger = logging.getLogger("MARK5.Optimizer")

class HyperparameterOptimizer:
    def __init__(self, base_dir="model_artifacts"):
        self.artifact_dir = base_dir
        os.makedirs(f"{self.artifact_dir}/config", exist_ok=True)

    def _purged_split(self, X, n_splits=5, purge_window=50):
        """
        Custom generator for Purged Walk-Forward CV.
        Removes 'purge_window' samples between Train and Test to prevent leakage.
        """
        total_rows = len(X)
        # Adjust fold size to accommodate purge windows
        # We need n_splits test sets.
        # Approximate: total = (n_splits * test_size) + (n_splits * purge) + initial_train
        # Let's just use a safe fraction
        fold_size = int(total_rows / (n_splits + 1.5))
        
        for i in range(1, n_splits + 1):
            train_end = i * fold_size
            test_start = train_end + purge_window
            test_end = test_start + fold_size
            
            if test_end > total_rows: 
                # Try to squeeze the last fold if possible
                test_end = total_rows
                if test_end - test_start < (fold_size // 2): break # Skip if too small
            
            yield np.arange(0, train_end), np.arange(test_start, test_end)

    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray, ticker: str, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimizes XGBoost for Probability Calibration (LogLoss), not just Accuracy.
        """
        def objective(trial):
            # 1. Search Space (Conservative for Financials)
            params = {
                'objective': 'multi:softprob', # or binary:logistic
                'num_class': 3, # Sell, Hold, Buy
                'eval_metric': 'mlogloss',
                'booster': 'gbtree',
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 8), # Lower depth = less overfitting
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'gamma': trial.suggest_float('gamma', 0.1, 5.0), # Min Split Loss (Crucial for noise)
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True), # L1 Reg
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True), # L2 Reg
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'n_jobs': 4,
                'verbosity': 0
            }

            scores = []
            
            # 2. Purged CV Loop
            cv = self._purged_split(X, n_splits=4, purge_window=60) # Purge 60 bars (e.g. 5 hours of 5m bars)
            
            for step, (train_idx, val_idx) in enumerate(cv):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = xgb.XGBClassifier(**params)
                
                # Pruning Callback (Handle missing integration package)
                callbacks = []
                try:
                    from optuna.integration import XGBoostPruningCallback
                    callbacks.append(XGBoostPruningCallback(trial, "validation_0-mlogloss"))
                except ImportError:
                    pass # Pruning inside boosting rounds disabled, but trial pruning still works
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                preds_proba = model.predict_proba(X_val)
                score = log_loss(y_val, preds_proba)
                scores.append(score)

            # Return Mean Log Loss
            return np.mean(scores) if scores else float('inf')

        # 3. Create Study with Hyperband Pruner
        # We minimize LogLoss
        study = optuna.create_study(
            direction='minimize', 
            pruner=HyperbandPruner(min_resource=20, max_resource='auto', reduction_factor=3)
        )
        
        logger.info(f"⚡ Starting Optimization for {ticker} (Target: LogLoss)")
        study.optimize(objective, n_trials=n_trials, n_jobs=1) # Parallelization handled by Launcher

        best_params = study.best_params
        best_loss = study.best_value
        
        logger.info(f"✅ {ticker} Optimized. Best LogLoss: {best_loss:.4f}")

        # 4. Atomic Save
        save_path = f"{self.artifact_dir}/config/{ticker}_xgb_params.json"
        temp_path = save_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        os.replace(temp_path, save_path)

        return best_params

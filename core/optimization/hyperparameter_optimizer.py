"""
MARK5 HYPERPARAMETER OPTIMIZER v9.0 - MULTICLASS EDITION
C-01 FIX: All objectives now match trainer's 3-class multiclass targets.
Revisions from v8.0:
1. OBJECTIVE SHIFT: Binary → Multiclass (multi:softprob, multiclass, MultiClass).
2. PURGED VALIDATION: Prevents autocorrelation leakage.
3. HYPERBAND PRUNING: Kills bad trials early.
4. MULTI-MODEL SUPPORT: XGBoost, LightGBM, RandomForest, CatBoost.
"""

import optuna
import json
import os
import logging
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from optuna.pruners import HyperbandPruner
import catboost as cb

logger = logging.getLogger("MARK5.Optimizer")

class HyperparameterOptimizer:
    """
    Multi-model hyperparameter optimizer using Optuna.
    C-01 FIX: All objectives are now MULTICLASS to match the trainer.
    Supports: XGBoost, LightGBM, RandomForest, CatBoost.
    """
    
    def __init__(self, base_dir="model_artifacts"):
        self.artifact_dir = base_dir
        os.makedirs(f"{self.artifact_dir}/config", exist_ok=True)

    def _purged_split(self, X, n_splits=5, purge_window=50):
        """
        Custom generator for Purged Walk-Forward CV.
        Removes 'purge_window' samples between Train and Test to prevent leakage.
        """
        total_rows = len(X)
        fold_size = int(total_rows / (n_splits + 1.5))
        
        for i in range(1, n_splits + 1):
            train_end = i * fold_size
            test_start = train_end + purge_window
            test_end = test_start + fold_size
            
            if test_end > total_rows: 
                test_end = total_rows
                if test_end - test_start < (fold_size // 2): break
            
            yield np.arange(0, train_end), np.arange(test_start, test_end)

    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray, ticker: str, n_trials: int = 100) -> Dict[str, Any]:
        """Optimizes XGBoost for MULTICLASS Log Loss with purged walk-forward CV."""
        num_class = len(np.unique(y))
        if num_class < 3:
            num_class = 3

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
                'gamma': trial.suggest_float('gamma', 0.1, 3.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
                # C-01 FIX: multiclass objectives matching trainer
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': num_class,
                'tree_method': 'hist',
                'n_jobs': 4,
                'verbosity': 0
            }

            scores = []
            cv = self._purged_split(X, n_splits=4, purge_window=60)
            
            for step, (train_idx, val_idx) in enumerate(cv):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                preds_proba = model.predict_proba(X_val)
                loss = log_loss(y_val, preds_proba, labels=list(range(num_class)))
                scores.append(loss)

            return np.mean(scores) if scores else float('inf')

        study = optuna.create_study(
            direction='minimize', 
            pruner=HyperbandPruner(min_resource=20, max_resource='auto', reduction_factor=3)
        )
        
        logger.info(f"⚡ Starting XGBoost Optimization for {ticker} (Target: MultiClass LogLoss)")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_params = study.best_params
        best_loss = study.best_value
        
        logger.info(f"✅ {ticker} XGBoost Optimized. Best LogLoss: {best_loss:.4f}")
        self._save_params(ticker, 'xgb', best_params)
        return best_params

    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray, ticker: str, n_trials: int = 100) -> Dict[str, Any]:
        """Optimizes LightGBM for MULTICLASS LogLoss."""
        num_class = len(np.unique(y))
        if num_class < 3:
            num_class = 3

        def objective(trial):
            params = {
                # C-01 FIX: multiclass objectives matching trainer
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': num_class,
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'class_weight': 'balanced',
                'n_jobs': 4,
                'verbose': -1
            }

            scores = []
            cv = self._purged_split(X, n_splits=4, purge_window=60)
            
            for step, (train_idx, val_idx) in enumerate(cv):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                
                preds_proba = model.predict_proba(X_val)
                loss = log_loss(y_val, preds_proba, labels=list(range(num_class)))
                scores.append(loss)

            return np.mean(scores) if scores else float('inf')

        study = optuna.create_study(
            direction='minimize', 
            pruner=HyperbandPruner(min_resource=20, max_resource='auto', reduction_factor=3)
        )
        
        logger.info(f"⚡ Starting LightGBM Optimization for {ticker} (Target: MultiClass LogLoss)")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_params = study.best_params
        best_loss = study.best_value
        
        logger.info(f"✅ {ticker} LightGBM Optimized. Best LogLoss: {best_loss:.4f}")
        self._save_params(ticker, 'lgb', best_params)
        return best_params

    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray, ticker: str, n_trials: int = 50) -> Dict[str, Any]:
        """Optimizes RandomForest for MULTICLASS classification."""
        num_class = len(np.unique(y))
        if num_class < 3:
            num_class = 3

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': 'balanced_subsample',
                'n_jobs': 4,
                'random_state': 42
            }

            scores = []
            cv = self._purged_split(X, n_splits=4, purge_window=60)
            
            for step, (train_idx, val_idx) in enumerate(cv):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                
                preds_proba = model.predict_proba(X_val)
                loss = log_loss(y_val, preds_proba, labels=list(range(num_class)))
                scores.append(loss)

            return np.mean(scores) if scores else float('inf')

        study = optuna.create_study(
            direction='minimize', 
            pruner=HyperbandPruner(min_resource=10, max_resource='auto', reduction_factor=3)
        )
        
        logger.info(f"⚡ Starting RandomForest Optimization for {ticker} (Target: MultiClass LogLoss)")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_params = study.best_params
        best_loss = study.best_value
        
        logger.info(f"✅ {ticker} RandomForest Optimized. Best LogLoss: {best_loss:.4f}")
        self._save_params(ticker, 'rf', best_params)
        return best_params

    def optimize_catboost(self, X: np.ndarray, y: np.ndarray, ticker: str, n_trials: int = 50) -> Dict[str, Any]:
        """Optimizes CatBoost for MULTICLASS LogLoss with ordered boosting."""

        def objective(trial):
            # Compute per-class weights
            classes = np.unique(y)
            total = len(y)
            class_weights = [total / (len(classes) * max(np.sum(y == c), 1)) for c in range(3)]

            params = {
                'iterations': trial.suggest_int('iterations', 200, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 3, 6),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.9),
                # C-01 FIX: multiclass objectives matching trainer
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'class_weights': class_weights,
                'random_seed': 42,
                'verbose': 0,
                'early_stopping_rounds': 50,
                'boosting_type': 'Ordered',
            }

            scores = []
            cv = self._purged_split(X, n_splits=4, purge_window=60)
            
            for step, (train_idx, val_idx) in enumerate(cv):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = cb.CatBoostClassifier(**params)
                eval_pool = cb.Pool(X_val, y_val)
                model.fit(X_train, y_train, eval_set=eval_pool)
                
                preds_proba = model.predict_proba(X_val)
                loss = log_loss(y_val, preds_proba, labels=list(range(3)))
                scores.append(loss)

            return np.mean(scores) if scores else float('inf')

        study = optuna.create_study(
            direction='minimize', 
            pruner=HyperbandPruner(min_resource=10, max_resource='auto', reduction_factor=3)
        )
        
        logger.info(f"⚡ Starting CatBoost Optimization for {ticker} (Target: MultiClass LogLoss)")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_params = study.best_params
        best_loss = study.best_value
        
        logger.info(f"✅ {ticker} CatBoost Optimized. Best LogLoss: {best_loss:.4f}")
        self._save_params(ticker, 'catboost', best_params)
        return best_params

    def optimize_all_models(self, X: np.ndarray, y: np.ndarray, ticker: str, 
                            n_trials: int = 50) -> Dict[str, Dict]:
        """
        Optimizes all ensemble models (XGBoost, LightGBM, RandomForest, CatBoost).
        
        Returns:
            Dict with model names as keys containing best params.
        """
        logger.info(f"🚀 Starting Full Ensemble Optimization for {ticker}")
        
        results = {}
        results['xgboost'] = self.optimize_xgboost(X, y, ticker, n_trials=n_trials)
        results['lightgbm'] = self.optimize_lightgbm(X, y, ticker, n_trials=n_trials)
        results['random_forest'] = self.optimize_random_forest(X, y, ticker, n_trials=max(30, n_trials//2))
        results['catboost'] = self.optimize_catboost(X, y, ticker, n_trials=max(30, n_trials//2))
        
        logger.info(f"✅ Full Ensemble Optimization Complete for {ticker}")
        return results

    def _save_params(self, ticker: str, model_type: str, params: Dict):
        """Atomic save of optimized parameters."""
        save_path = f"{self.artifact_dir}/config/{ticker}_{model_type}_params.json"
        temp_path = save_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(params, f, indent=2)
        os.replace(temp_path, save_path)

    def load_params(self, ticker: str, model_type: str) -> Dict:
        """Load previously optimized parameters."""
        load_path = f"{self.artifact_dir}/config/{ticker}_{model_type}_params.json"
        if os.path.exists(load_path):
            with open(load_path, 'r') as f:
                return json.load(f)
        return {}

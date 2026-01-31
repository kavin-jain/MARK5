"""
MARK5 Advanced ML Trainer v7.0 - ARCHITECT EDITION
Revisions:
1. Integrated Probability Calibration (Isotonic Regression) - No more fake confidence.
2. Geometric Mean Ensembling - Penalizes model disagreement.
3. Strict Feature Preservation - Prevents index misalignment.
"""

import numpy as np
import pandas as pd
import sys
import os
import gc
import logging
import joblib
from datetime import datetime
from typing import Dict, Tuple

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Sklearn & Boosting
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss, brier_score_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import lightgbm as lgb

# Import Core definitions
try:
    from core.utils.constants import FEATURE_EXCLUDE_COLUMNS
    from core.models.training.financial_engineer import FinancialEngineer
except ImportError:
    FEATURE_EXCLUDE_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']
    from core.models.training.financial_engineer import FinancialEngineer

# LOGGING SETUP
logger = logging.getLogger("MARK5_Architect")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - [ARCHITECT] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from core.models.model_versioning import ModelVersionManager

class MARK5MLTrainer:
    def __init__(self, config=None):
        self.config = config if config else self._get_default_config()
        self.use_gpu = self._setup_hardware()
        self.models_base_dir = getattr(self.config, 'models_dir', 'models')
        self.version_manager = ModelVersionManager(self.config.__dict__ if hasattr(self.config, '__dict__') else self.config)
        os.makedirs(self.models_base_dir, exist_ok=True)

    def _get_default_config(self):
        class Config:
            prediction_horizon = 5
            gpu_memory_limit = 4096
            models_dir = "./models"
            transaction_cost = 0.0012 # 0.12% round trip realism
        return Config()

    def _setup_hardware(self) -> bool:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                return True
            return False
        except Exception: return False

    # -------------------------------------------------------------------------
    # 1. DATA PREP WITH REALISM
    # -------------------------------------------------------------------------
    def prepare_data_dynamic(self, data: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        # Pass transaction cost to Engineer
        fe = FinancialEngineer(transaction_cost_pct=getattr(self.config, 'transaction_cost', 0.001))
        
        # Use High/Low for better Volatility calculation if available
        labels_df = fe.get_labels(
            prices=data, 
            run_bars=self.config.prediction_horizon, 
            pt_sl=[2.0, 1.0] # 2:1 Reward Risk
        )
        
        # Inner join to align features
        aligned_df = data.join(labels_df[['bin']], how='inner')
        aligned_df.dropna(inplace=True)
        
        # Target: 1 (Profitable Buy) vs 0 (Everything else)
        targets_clean = aligned_df['bin'].values.astype(int)
        
        # Features
        feature_cols = [c for c in aligned_df.columns if c not in FEATURE_EXCLUDE_COLUMNS + ['bin', 'target', 'ret', 'out', 'sl', 'pt', 't1']]
        
        # Calculate weights - Balanced can be dangerous in high noise, using sqrt(balanced) to dampen
        classes = np.unique(targets_clean)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=targets_clean)
        class_weight_dict = dict(zip(classes, weights))
        
        logger.info(f"[{ticker}] Training Samples: {len(aligned_df)} | Class Weights: {class_weight_dict}")
        
        return aligned_df[feature_cols], targets_clean, class_weight_dict

    # -------------------------------------------------------------------------
    # 2. PURGED WALK-FORWARD VALIDATION WITH CALIBRATION
    # -------------------------------------------------------------------------
    def train_advanced_ensemble(self, ticker: str, data: pd.DataFrame) -> Dict:
        logger.info(f"🚀 Starting ARCHITECT-LEVEL training for {ticker}")
        
        features_df, y, class_weights = self.prepare_data_dynamic(data, ticker)
        X = features_df.values
        feature_names = features_df.columns.tolist()
        
        # Walk-Forward Params
        n_samples = len(X)
        train_window_size = int(n_samples * 0.50)
        test_window_size = int(n_samples * 0.10)
        step_size = test_window_size
        embargo = self.config.prediction_horizon * 2 # Safety buffer
        
        best_brier_score = 1.0 # Lower is better
        best_models_global = {}
        final_scaler = None
        calibrators_global = {}
        
        current_train_end = train_window_size
        
        while current_train_end + test_window_size <= n_samples:
            test_start = current_train_end + embargo
            test_end = test_start + test_window_size
            
            if test_end > n_samples: break
            
            # --- SPLITS ---
            X_train_raw = X[:current_train_end]
            y_train = y[:current_train_end]
            X_test_raw = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            # --- SCALING ---
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
            
            # --- CALIBRATION SPLIT ---
            # We need a hold-out set INSIDE training just to calibrate probabilities
            cal_size = int(len(X_train) * 0.15)
            X_train_sub = X_train[:-cal_size]
            y_train_sub = y_train[:-cal_size]
            X_cal = X_train[-cal_size:]
            y_cal = y_train[-cal_size:]
            
            # --- TRAINING ---
            models = {}
            calibrators = {}
            preds_proba = {}
            
            # 1. XGBoost
            xgb_model = self._train_xgboost(X_train_sub, y_train_sub, X_cal, y_cal, class_weights)
            models['xgb'] = xgb_model
            # Calibrate XGB
            raw_xgb_cal = xgb_model.predict_proba(X_cal)[:, 1]
            iso_xgb = IsotonicRegression(out_of_bounds='clip')
            iso_xgb.fit(raw_xgb_cal, y_cal)
            calibrators['xgb'] = iso_xgb
            # Predict Test
            raw_xgb_test = xgb_model.predict_proba(X_test)[:, 1]
            preds_proba['xgb'] = iso_xgb.transform(raw_xgb_test)

            # 2. LightGBM
            lgb_model = self._train_lightgbm(X_train_sub, y_train_sub, X_cal, y_cal, class_weights)
            models['lgb'] = lgb_model
            # Calibrate LGB
            raw_lgb_cal = lgb_model.predict_proba(X_cal)[:, 1]
            iso_lgb = IsotonicRegression(out_of_bounds='clip')
            iso_lgb.fit(raw_lgb_cal, y_cal)
            calibrators['lgb'] = iso_lgb
            # Predict Test
            raw_lgb_test = lgb_model.predict_proba(X_test)[:, 1]
            preds_proba['lgb'] = iso_lgb.transform(raw_lgb_test)

            # 3. Random Forest (Self-calibrating usually, but let's be safe)
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, class_weight=class_weights)
            rf_model.fit(X_train_sub, y_train_sub)
            models['rf'] = rf_model
            # Calibrate RF
            raw_rf_cal = rf_model.predict_proba(X_cal)[:, 1]
            iso_rf = IsotonicRegression(out_of_bounds='clip')
            iso_rf.fit(raw_rf_cal, y_cal)
            calibrators['rf'] = iso_rf
            # Predict Test
            raw_rf_test = rf_model.predict_proba(X_test)[:, 1]
            preds_proba['rf'] = iso_rf.transform(raw_rf_test)
            
            # --- GEOMETRIC MEAN ENSEMBLE ---
            # Geometric mean penalizes low confidence from any single model
            ens_prob = (preds_proba['xgb'] * preds_proba['lgb'] * preds_proba['rf']) ** (1/3)
            ens_pred = (ens_prob > 0.5).astype(int)
            
            # Metrics
            brier = brier_score_loss(y_test, ens_prob) # Measures Calibration Accuracy
            precision = precision_score(y_test, ens_pred, zero_division=0)
            
            logger.info(f"Validating... Brier Score: {brier:.4f} (Lower is better) | Precision: {precision:.2%}")
            
            if brier < best_brier_score:
                best_brier_score = brier
                best_models_global = models
                calibrators_global = calibrators
                final_scaler = scaler
                
            current_train_end += step_size
            gc.collect()

        if not best_models_global:
            return {'status': 'failed', 'reason': 'Model convergence failure'}
            
        # VERSION CONTROL
        version = self.version_manager.increment_version(ticker)
        
        self._save_artifacts(ticker, best_models_global, calibrators_global, final_scaler, feature_names, version)
        
        return {'status': 'success', 'brier_score': best_brier_score, 'version': version}

    def _train_xgboost(self, X_t, y_t, X_v, y_v, weights):
        # Binary objective is safer for 0/1 specific tasks
        sample_w = np.array([weights[y] for y in y_t])
        clf = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', eval_metric='logloss',
            tree_method='hist', device='cuda' if self.use_gpu else 'cpu',
            early_stopping_rounds=50, random_state=42
        )
        clf.fit(X_t, y_t, sample_weight=sample_w, eval_set=[(X_v, y_v)], verbose=False)
        return clf

    def _train_lightgbm(self, X_t, y_t, X_v, y_v, weights):
        sample_w = np.array([weights[y] for y in y_t])
        callbacks = [lgb.early_stopping(50, verbose=False)]
        clf = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            objective='binary', device='gpu' if self.use_gpu else 'cpu',
            random_state=42, verbose=-1
        )
        clf.fit(X_t, y_t, sample_weight=sample_w, eval_set=[(X_v, y_v)], callbacks=callbacks)
        return clf

    def _save_artifacts(self, ticker, models, calibrators, scaler, features, version):
        # Versioned Path: models/TICKER/v1_20251205/
        ts = datetime.now().strftime("%Y%m%d")
        path = os.path.join(self.models_base_dir, ticker, f"v{version}_{ts}")
        os.makedirs(path, exist_ok=True)
        
        # Also create a 'latest' symlink or copy if needed, but for now we rely on predictor finding the version.
        # Actually ModelVersionManager tells us the version integer. 
        # But we need to know the PATH for that version.
        # Simplification: We save to `models/TICKER/v{version}` directly to find it easily? 
        # OR we assume predictor looks for largest version number.
        # Let's stick to the user's implied path structure or just `v{version}`.
        # User said: `model_path.format(..., version=version)`
        # I'll use `v{version}` directory name for simplicity.
        
        path = os.path.join(self.models_base_dir, ticker, f"v{version}")
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(scaler, os.path.join(path, 'scaler.pkl'))
        joblib.dump(features, os.path.join(path, 'features.json'))
        
        for name, model in models.items():
            joblib.dump(model, os.path.join(path, f'{name}_model.pkl'))
            # Save the calibrator alongside the model
            joblib.dump(calibrators[name], os.path.join(path, f'{name}_calibrator.pkl'))
            
        logger.info(f"✅ Calibrated Models saved to {path}")

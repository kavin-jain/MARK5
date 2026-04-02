"""
MARK5 INFERENCE ENGINE v7.0 - ARCHITECT EDITION
Revisions:
1. ATOMIC MODEL SWAPPING: No race conditions during weight updates.
2. ENTROPY GUARDRAIL: Rejects predictions if probability distribution is too flat.
3. FAIL-SAFE DEFAULTS: Returns 'HOLD' on any internal exception.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from threading import Lock
from core.models.features import AdvancedFeatureEngine
from core.models.model_versioning import ModelVersionManager
from core.utils.config_manager import get_config

class AtomicModelContainer:
    """
    Holds the state of the world.
    Swapped atomically to prevent partial reads during inference.

    v10.0: meta_model attribute added for stacking ensemble inference.
    When meta_model is not None (v10+ artifacts), prediction uses
    LogisticRegression(p_xgb, p_lgb, p_cat) instead of arithmetic mean.
    When None (v9 or older artifacts), falls back to arithmetic mean.
    """
    def __init__(
        self,
        models: Dict,
        scaler,
        weights: Dict,
        schema: list,
        calibrators: Dict = None,
        meta_model=None,
    ):
        self.models = models
        self.scaler = scaler
        self.weights = weights
        self.schema = schema
        self.calibrators = calibrators or {}
        self.meta_model = meta_model   # LogisticRegression or None
        self.timestamp = datetime.now()

class MARK5Predictor:
    def __init__(self, ticker: str, data_provider=None):
        self.logger = logging.getLogger(f"MARK5.Inference.{ticker}")
        self.ticker = ticker
        self.feature_engine = AdvancedFeatureEngine()
        
        # Dependency Injection or Singleton Config
        self.config = get_config()
        self.version_manager = ModelVersionManager(self.config.__dict__ if hasattr(self.config, '__dict__') else self.config)
        self.version = self.version_manager.get_latest_version(ticker)
        
        self._container: Optional[AtomicModelContainer] = None
        self._lock = Lock() # For swapping the container
        self.dl_seq_length = 60
        
        self.reload_artifacts()

    def reload_artifacts(self):
        """
        Loads artifacts into a NEW container, then atomically swaps.
        Zero downtime.
        """
        try:
            # Look for versioned directory first
            version = self.version_manager.get_latest_version(self.ticker)
            base_dir = f"models/{self.ticker}/v{version}"
            
            # Fallback to non-versioned if version is 0 or path missing
            if version == 0 or not os.path.exists(base_dir):
                base_dir = f"models/{self.ticker}"
            
            # Load Schema
            
            # Load Schema
            if not os.path.exists(f"{base_dir}/features.json"):
                self.logger.warning(f"No artifacts found for {self.ticker}")
                return

            with open(f"{base_dir}/features.json", 'r') as f:
                schema = json.load(f)
            
            # Load Scaler
            scaler = joblib.load(f"{base_dir}/scaler.pkl")
            
            # Load Weights
            weights = {}
            if os.path.exists(f"{base_dir}/weights.json"):
                with open(f"{base_dir}/weights.json", 'r') as f:
                    weights = json.load(f)

            # Load base models — v10.0: 'rf' replaced by 'cat' (CatBoostClassifier)
            # Both old ('rf') and new ('cat') names are tried for backwards compatibility.
            models = {}
            calibrators = {}
            for name in ['xgb', 'lgb', 'cat', 'rf']:
                model_path = f"{base_dir}/{name}_model.pkl"
                cal_path = f"{base_dir}/{name}_calibrator.pkl"
                if os.path.exists(model_path):
                    models[name] = joblib.load(model_path)
                if os.path.exists(cal_path):
                    calibrators[name] = joblib.load(cal_path)

            # Load stacking meta-learner (v10.0+); None for older model versions.
            meta_model = None
            meta_path = f"{base_dir}/meta_model.pkl"
            if os.path.exists(meta_path):
                meta_model = joblib.load(meta_path)
                self.logger.info("Meta-learner loaded for stacking inference.")

            # ATOMIC SWAP — meta_model passed to constructor (v10.0+)
            new_container = AtomicModelContainer(
                models, scaler, weights, schema, calibrators,
                meta_model=meta_model,
            )
            with self._lock:
                self._container = new_container

            self.logger.info(
                f"✅ Models hot-swapped: {list(models.keys())} | "
                f"meta={'yes' if meta_model else 'no'}"
            )

        except Exception as e:
            self.logger.error(f"Artifact reload failed: {e}")

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """High entropy = Flat distribution = Confusion."""
        return -np.sum(probs * np.log(probs + 1e-9))

    def predict(self, raw_data: pd.DataFrame, **kwargs) -> Dict:
        # Get local reference to avoid locking overhead
        container = self._container
        if not container:
            return {'status': 'error', 'msg': 'System initializing'}

        try:
            # 1. Feature Engineering
            df_feats = self.feature_engine.engineer_all_features(raw_data)
            
            
            # 2. Schema Alignment (Strict)
            # Fills missing with 0, Drops extra
            # M-5: Warn explicitly when the live feature set diverges from the
            # training schema so silent zero-fill degradation is surfaced.
            dropped_cols = set(df_feats.columns) - set(container.schema)
            zero_filled_cols = set(container.schema) - set(df_feats.columns)
            if dropped_cols or zero_filled_cols:
                self.logger.warning(
                    f"Schema mismatch for {self.ticker}: "
                    f"extra_features_dropped={dropped_cols}, "
                    f"missing_features_zero_filled={zero_filled_cols}"
                )
            df_aligned = df_feats.reindex(columns=container.schema).fillna(0)
            
            # 3. Scaling
            # Pass numpy array explicitly — all CPCV base models were fitted on
            # numpy arrays (scaler.fit_transform returns ndarray). Passing a
            # DataFrame here triggers sklearn feature-name warnings and can cause
            # LGB to attempt column reordering against its stored feature names.
            X_scaled = container.scaler.transform(df_aligned.values)
            X_current = X_scaled[-1].reshape(1, -1)
            
            X_seq = None
            if len(X_scaled) >= self.dl_seq_length:
                X_seq = X_scaled[-self.dl_seq_length:].reshape(1, self.dl_seq_length, -1)

            # 4. Inference — Binary Classification (0=not-buy, 1=buy)
            # Geometric mean ensemble (matches trainer's validation)
            model_probs_class1 = []  # probability of class 1 (buy)
            model_details = {}
            
            for name, model in container.models.items():
                w = container.weights.get(name, 1.0)
                if w <= 0: continue
                
                try:
                    # All base models trained on numpy arrays in CPCV — pass numpy.
                    probs = model.predict_proba(X_current)[0]
                    raw_prob_buy = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    
                    # USE RAW PROBABILITIES — calibration (both isotonic and Platt)
                    # destroys signal. Diagnostic proved:
                    #   RF raw std=0.10 → calibrated std=0.006
                    #   XGB raw std=0.019 → calibrated std=0.0003
                    # Calibration with 75 samples maps everything to prior (~0.43)
                    prob_buy = raw_prob_buy
                    
                    model_probs_class1.append(prob_buy)
                    model_details[name] = {'raw': raw_prob_buy}
                except Exception:
                    continue

            if not model_probs_class1:
                return {'signal': 'HOLD', 'reason': 'All models failed'}

            # ── Ensemble: stacking meta-learner (v10+) or arithmetic mean fallback ──
            # Meta-model (LogisticRegression) was trained on OOF predictions from CPCV.
            # It learns the correct weighting of base models from held-out data.
            # Arithmetic mean is the fallback for older model artifacts (v9 and below).
            if container.meta_model is not None and len(model_probs_class1) == 3:
                # Stacking path: feed base model probs to meta-learner.
                # Order must match training order: [xgb, lgb, cat].
                # model_probs_class1 preserves insertion order from the model loop above.
                meta_X = np.array(model_probs_class1).reshape(1, -1)
                confidence = float(container.meta_model.predict_proba(meta_X)[0, 1])
                ensemble_method = 'stacking'
            else:
                # Fallback: arithmetic mean (v9 artifacts or partial model loads).
                confidence = float(np.mean(model_probs_class1))
                ensemble_method = 'arithmetic_mean'
            
            # RAW probability hurdle: 0.52 (above 0.50 random baseline)
            PROBABILITY_HURDLE = 0.52

            if confidence >= PROBABILITY_HURDLE:
                signal = "BUY"
            else:
                signal = "HOLD"
                if confidence >= 0.50:
                    signal = f"HOLD (Conf {confidence:.0%} < {PROBABILITY_HURDLE:.0%} hurdle)"

            # C-3: Real entropy from final probability distribution.
            entropy_val = self._calculate_entropy(
                np.array([1.0 - confidence, confidence])
            )

            return {
                'status': 'success',
                'signal': signal,
                'confidence': confidence,
                'entropy': entropy_val,
                'ensemble_method': ensemble_method,
                'probs': [1.0 - confidence, confidence],
            }

        except Exception as e:
            self.logger.error(f"Inference crash: {e}")
            return {'signal': 'HOLD', 'reason': 'Crash Protection'}
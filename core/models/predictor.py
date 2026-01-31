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
    """
    def __init__(self, models: Dict, scaler, weights: Dict, schema: list):
        self.models = models
        self.scaler = scaler
        self.weights = weights
        self.schema = schema
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

            # Load Models
            models = {}
            for name in ['xgboost', 'lightgbm', 'lstm', 'tcn']:
                path = f"{base_dir}/{name}_model.pkl"
                if os.path.exists(path):
                    models[name] = joblib.load(path)

            # ATOMIC SWAP
            new_container = AtomicModelContainer(models, scaler, weights, schema)
            with self._lock:
                self._container = new_container
                
            self.logger.info("✅ Models hot-swapped successfully.")

        except Exception as e:
            self.logger.error(f"Artifact reload failed: {e}")

    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """High entropy = Flat distribution = Confusion."""
        return -np.sum(probs * np.log(probs + 1e-9))

    def predict(self, raw_data: pd.DataFrame) -> Dict:
        # Get local reference to avoid locking overhead
        container = self._container
        if not container:
            return {'status': 'error', 'msg': 'System initializing'}

        try:
            # 1. Feature Engineering
            df_feats = self.feature_engine.engineer_all_features(raw_data)
            
            # 2. Schema Alignment (Strict)
            # Fills missing with 0, Drops extra
            df_aligned = df_feats.reindex(columns=container.schema).fillna(0)
            
            # 3. Scaling
            X_scaled = container.scaler.transform(df_aligned)
            X_current = X_scaled[-1].reshape(1, -1)
            
            X_seq = None
            if len(X_scaled) >= self.dl_seq_length:
                X_seq = X_scaled[-self.dl_seq_length:].reshape(1, self.dl_seq_length, -1)

            # 4. Inference
            weighted_probs = np.zeros(3) # Sell, Hold, Buy
            total_weight = 0.0
            
            model_details = {}
            
            for name, model in container.models.items():
                w = container.weights.get(name, 1.0)
                if w <= 0: continue
                
                try:
                    if name in ['lstm', 'tcn']:
                        if X_seq is None: continue
                        probs = model.predict_proba(X_seq)[0]
                    else:
                        probs = model.predict_proba(X_current)[0]
                    
                    weighted_probs += probs * w
                    total_weight += w
                    model_details[name] = probs.tolist()
                    
                except Exception:
                    continue

            if total_weight == 0:
                return {'signal': 'HOLD', 'reason': 'All models failed'}

            final_probs = weighted_probs / total_weight
            
            # 5. Entropy Guardrail
            # Max entropy for 3 classes is ~1.09 (33/33/33). 
            # If entropy > 1.0, the models are confused.
            entropy = self._calculate_entropy(final_probs)
            
            final_class = int(np.argmax(final_probs))
            confidence = float(final_probs[final_class])
            
            signal = {0: "SELL", 1: "HOLD", 2: "BUY"}[final_class]
            
            if entropy > 1.0:
                signal = "HOLD (High Entropy)"
            elif confidence < 0.55:
                signal = "HOLD (Low Conf)"

            return {
                'status': 'success',
                'signal': signal,
                'confidence': confidence,
                'entropy': round(entropy, 3),
                'probs': final_probs.tolist()
            }

        except Exception as e:
            self.logger.error(f"Inference crash: {e}")
            return {'signal': 'HOLD', 'reason': 'Crash Protection'}

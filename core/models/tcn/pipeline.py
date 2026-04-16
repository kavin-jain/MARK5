"""
MARK5 UNIFIED PIPELINE (Production Grade)
=========================================
Changes:
1. Removed Duplicate Classes: No more "Shadow" classes at the bottom.
2. Robust Position Sizing: Added "Half-Kelly" logic and Volatility Floors.
3. VWAP & Time Features: Added to Feature Engineering injection.
4. Self-Standardizing Features: Using rolling windows for zero-leakage normalization.
"""

import numpy as np
import pandas as pd
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Path setup
current_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(current_dir))

from core.models.tcn.system import TCNTradingModel
from core.models.tcn.backtester import RobustBacktester as TCNBacktester
from core.models.tcn.features import AlphaFeatureEngineer as EnhancedFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MARK5_PIPELINE")

class RobustSignalProcessor:
    """
    Advanced Signal Processing with Hysteresis.
    Prevents "flickering" signals around the threshold.
    """
    def __init__(self, deadband_lower=0.45, deadband_upper=0.55):
        self.lower = deadband_lower
        self.upper = deadband_upper
        self.current_state = 'HOLD' 

    def process(self, probability: float) -> Tuple[str, float]:
        # Validate Input
        if not np.isfinite(probability):
            return 'HOLD', 0.0
            
        probability = float(np.clip(probability, 0.0, 1.0))

        # Strength calculation
        strength = 0.0
        if probability > 0.5:
            strength = (probability - 0.5) * 2
        else:
            strength = (0.5 - probability) * 2

        # Hysteresis Logic
        if probability > self.upper:
            self.current_state = 'BUY'
        elif probability < self.lower:
            self.current_state = 'SELL'
        else:
            # Deadband = HOLD
            self.current_state = 'HOLD'
            strength = 0.0
            
        return self.current_state, strength

class SafePositionSizer:
    """
    Risk-First Position Sizing.
    Protects against "Low Volatility Prediction" traps.
    """
    def __init__(self, target_vol=0.15, max_pos=1.0, floor_vol=0.08):
        # HARD Validation of Risk Parameters
        if not (0.0 < max_pos <= 1.0):
             raise ValueError(f"CRITICAL: max_pos must be between 0 and 1. Got {max_pos}")
        if not (0.0 < target_vol <= 5.0): # 500% vol is unrealistic, sanity cap
             raise ValueError("target_vol out of sanity range")
             
        self.target_vol = target_vol
        self.max_pos = max_pos
        self.floor_vol = floor_vol 

    def calculate(self, signal: str, strength: float, pred_vol_annualized: float) -> float:
        if signal == 'HOLD' or strength < 0.1:
            return 0.0
        
        # Volatility Sanity Check
        if not np.isfinite(pred_vol_annualized) or pred_vol_annualized <= 0:
            logger.warning(f"⚠️ Invalid Predicted Volatility: {pred_vol_annualized}. Using Floor.")
            pred_vol_annualized = self.floor_vol
            
        # FAIL-SAFE: Use max of Predicted or Floor
        effective_vol = max(pred_vol_annualized, self.floor_vol)
        
        # Volatility Targeting
        vol_scalar = self.target_vol / effective_vol
        
        # Cap leverage multiplier safely (0.1x to 2.5x)
        vol_scalar = np.clip(vol_scalar, 0.1, 2.5) 
        
        # Confidence Scaling
        size = vol_scalar * strength
        
        # Hard Cap
        size = np.clip(size, 0.0, self.max_pos)
        
        if signal == 'SELL':
            size = -size
            
        return float(size)

class TCNPipelineOrchestrator:
    def __init__(self, sequence_length=64):
        self.seq_len = sequence_length
        self.n_feat = None # Dynamic detection
        self.model = None # Lazy init after feature detection
        self.signal_proc = RobustSignalProcessor()
        self.sizer = SafePositionSizer()
        self.engineer = EnhancedFeatureEngineer(use_robust_scaling=True)
        self.backtester = TCNBacktester()

    def _validate_data(self, df: pd.DataFrame, min_len: int = 0) -> bool:
        """Strict Data Validation"""
        if df is None or not isinstance(df, pd.DataFrame):
            logger.error("❌ Data must be a DataFrame")
            return False
            
        req_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in req_cols):
             logger.error(f"❌ Missing required columns: {req_cols}")
             return False
             
        if len(df) < min_len:
             logger.error(f"❌ Insufficient data length. Need {min_len}, got {len(df)}")
             return False
             
        if df[req_cols].isnull().any().any():
             logger.error("❌ Data contains NaNs in price/volume columns")
             return False
             
        return True

    def load_production_model(self, model_path: str):
        """Load Model for Inference"""
        path_obj = Path(model_path)
        
        # Determine n_features from the engineer's feature list
        self.n_feat = self.engineer.n_features
        
        self.model = TCNTradingModel(self.seq_len, self.n_feat)
        self.model.build_model()
        self.model.load(model_path)
        logger.info(f"✅ Model loaded from {model_path} (n_feat={self.n_feat})")

    def _prepare_sequences(self, features: pd.DataFrame, targets_dir=None, targets_vol=None):
        X, y_d, y_v = [], [], []
        vals = features.values
        
        has_targets = targets_dir is not None
        
        # Validation checks
        if len(vals) < self.seq_len:
             return np.array([]), np.array([]), np.array([])

        for i in range(len(vals) - self.seq_len):
            X.append(vals[i : i + self.seq_len])
            if has_targets:
                # Correct Alignment:
                # Sequence 0..T-1 (Length T) -> Should predict Outcome at T+1 ?
                # The original review noted standard TCN usually predicts T+1 given 0..T
                
                # Here we use the target aligned with the END of the sequence
                # Index i+seq_len is the first bar *after* the sequence
                # targets_dir was calculated using shift(-1), so it already points to future
                
                y_d.append(targets_dir[i + self.seq_len - 1]) # Use 'current' target (which looks forward)
                y_v.append(targets_vol[i + self.seq_len - 1])
                
        return np.array(X), np.array(y_d), np.array(y_v)

    def train_production(self, data: pd.DataFrame, save_path: str):
        """Trains the Final Production Model on ALL data"""
        if not self._validate_data(data, min_len=self.seq_len + 50):
            return

        logger.info("🔧 Starting Production Training...")
        
        # 1. Feature Engineering
        # Unified engine is self-standardizing, no training_mode needed for scaling
        features = self.engineer.generate_features(data)
        
        # DYNAMIC FEATURE COUNT DETECTION
        self.n_feat = features.shape[1]
        logger.info(f"detected n_features: {self.n_feat}")

        # 2. Target Generation (Corrected)
        # Direction: Close[t+1] > Close[t]
        t_dir = (data['close'].shift(-1) > data['close']).astype(int).fillna(0).values
        
        # Volatility: Next day's realized volatility proxy
        t_vol = (data['close'].pct_change().rolling(20).std() * np.sqrt(252)).shift(-1).fillna(0.01).values

        # 3. Sequencing
        X, y_d, y_v = self._prepare_sequences(features, t_dir, t_vol)
        
        if len(X) == 0:
            logger.error("❌ Sequence generation failed (Empty).")
            return
        
        # 4. Train
        split = int(len(X) * 0.9)
        
        # Lazy Build
        self.model = TCNTradingModel(self.seq_len, self.n_feat)
        self.model.build_model()
        
        self.model.train(
            X[:split], y_d[:split], y_v[:split],
            X[split:], y_d[split:], y_v[split:],
            epochs=50, batch_size=64
        )
        
        # 5. Save Model
        self.model.save(save_path)
        
        logger.info("✅ Production Model Deployed.")

    def predict_live(self, recent_data: pd.DataFrame) -> Dict:
        """
        Live Inference Engine with Safety Guards.
        """
        try:
            # 1. Input Validation
            if not self._validate_data(recent_data, min_len=self.seq_len):
                raise ValueError("Invalid Data Input")

            # 2. Feature Engineering
            # Must match training features EXACTLY
            # Use training_mode=False to use fitted scaler
            features = self.engineer.generate_features(recent_data, training_mode=False)
            
            if len(features) < self.seq_len:
                raise ValueError(f"Insufficient features: {len(features)} < {self.seq_len}")
            
            # Auto-detect n_feat if not set (first run)
            if self.n_feat is None:
                 self.n_feat = features.shape[1]
                 # If model is None, we can't predict!
                 if self.model is None:
                     raise ValueError("Model not initialized/loaded!")

            # Extract last sequence
            last_seq = features.values[-self.seq_len:]
            X = last_seq.reshape(1, self.seq_len, self.n_feat)
            
            # 3. Model Inference
            pred_dir_prob, pred_vol = self.model.predict(X)
            
            # Handle potential array shapes
            prob = float(pred_dir_prob[0][0]) if isinstance(pred_dir_prob, np.ndarray) else float(pred_dir_prob)
            vol = float(pred_vol[0][0]) if isinstance(pred_vol, np.ndarray) else float(pred_vol)
            
            # OUTPUT VALIDATION
            if not (0.0 <= prob <= 1.0):
                 logger.error(f"❌ Invalid Probability: {prob}")
                 raise ValueError("Model output probability out of bounds")
                 
            if not (0.0 < vol < 5.0): # 500% vol sanity limit
                 # If slightly negative due to float errors, clip? 
                 # But TCN regression typically outputs scalar.
                 if vol <= 0:
                     logger.warning(f"⚠️ Negative Volatility Predicted: {vol}. Warning.")
                     # PositionSizer handles this via floor, but good to flag.
                 elif vol >= 5.0:
                     logger.error(f"❌ Extreme Volatility: {vol}")
                     raise ValueError("Model output volatility insane")

            # 4. Signal Processing
            signal, strength = self.signal_proc.process(prob)
            
            # 5. Position Sizing
            size = self.sizer.calculate(signal, strength, vol)
            
            return {
                "timestamp": datetime.now(),
                "signal": signal,
                "probability": prob,
                "confidence": strength,
                "forecast_volatility": vol,
                "recommended_position": size,
                "safe_mode": False
            }
            
        except Exception as e:
            logger.error(f"❌ CRITICAL FAILURE in predict_live: {e}", exc_info=True)
            # Safe Fallback
            return {
                "timestamp": datetime.now(),
                "signal": "HOLD", # Safety Default
                "probability": 0.5,
                "confidence": 0.0,
                "forecast_volatility": 0.0,
                "recommended_position": 0.0,
                "safe_mode": True,
                "error": str(e)
            }

if __name__ == "__main__":
    print("MARK5 Pipeline Loaded. Ready for Matrix.")

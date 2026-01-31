"""
MARK5 UNIFIED PIPELINE (Production Grade)
=========================================
Changes:
1. Removed Duplicate Classes: No more "Shadow" classes at the bottom.
2. Robust Position Sizing: Added "Half-Kelly" logic and Volatility Floors.
3. VWAP & Time Features: Added to Feature Engineering injection.
4. Walk-Forward Leakage Fix: Scalers are now isolated per split.
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

from core.models.tcn.system import TCNTradingModel  # Using the Master Model
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
            # If we are in the deadband, we HOLD unless we want to implement trailing stops here.
            # For pure signal generation, deadband = HOLD.
            self.current_state = 'HOLD'
            strength = 0.0
            
        return self.current_state, strength

class SafePositionSizer:
    """
    Risk-First Position Sizing.
    Protects against "Low Volatility Prediction" traps.
    """
    def __init__(self, target_vol=0.15, max_pos=1.0, floor_vol=0.08):
        self.target_vol = target_vol
        self.max_pos = max_pos
        self.floor_vol = floor_vol # Minimum volatility assumption (8% annualized)

    def calculate(self, signal: str, strength: float, pred_vol_annualized: float) -> float:
        if signal == 'HOLD' or strength < 0.1:
            return 0.0
        
        # FAIL-SAFE: Never trust the model if it predicts unreasonably low volatility
        # Use the maximum of Predicted Volatility OR Floor Volatility
        effective_vol = max(pred_vol_annualized, self.floor_vol)
        
        # Volatility Targeting (Kelly-style sizing)
        # Size = Target_Vol / Effective_Vol
        vol_scalar = self.target_vol / effective_vol
        
        # Cap leverage multiplier to avoid "Flash Crash" suicide
        vol_scalar = np.clip(vol_scalar, 0.1, 2.5) 
        
        # Confidence Scaling (Half-Kelly heuristic)
        # We multiply by strength (confidence) to size down on weak signals
        size = vol_scalar * strength
        
        # Hard Cap
        size = np.clip(size, 0.0, self.max_pos)
        
        if signal == 'SELL':
            size = -size
            
        return size

class TCNPipelineOrchestrator:
    def __init__(self, sequence_length=64, n_features=54):
        self.seq_len = sequence_length
        self.n_feat = n_features
        self.model = TCNTradingModel(sequence_length, n_features)
        self.signal_proc = RobustSignalProcessor()
        self.sizer = SafePositionSizer()
        self.engineer = EnhancedFeatureEngineer(use_robust_scaling=True)
        self.backtester = TCNBacktester()

    def _prepare_sequences(self, features: pd.DataFrame, targets_dir=None, targets_vol=None):
        X, y_d, y_v = [], [], []
        vals = features.values
        
        has_targets = targets_dir is not None
        
        # Optimization: Use numpy striding for speed if possible, but loop is safer for clarity
        for i in range(len(vals) - self.seq_len):
            X.append(vals[i : i + self.seq_len])
            if has_targets:
                y_d.append(targets_dir[i + self.seq_len])
                y_v.append(targets_vol[i + self.seq_len])
                
        return np.array(X), np.array(y_d), np.array(y_v)

    def train_production(self, data: pd.DataFrame, save_path: str):
        """Trains the Final Production Model on ALL data"""
        logger.info("🔧 Injecting Indian Market Features (VWAP)...")
        
        # 1. Feature Engineering (Enhanced)
        if 'vwap' not in data.columns:
            # Simple VWAP calculation if missing
            data['vwap'] = (data['volume'] * (data['high']+data['low']+data['close'])/3).cumsum() / data['volume'].cumsum()
        
        features = self.engineer.engineer_features(data)
        
        # 2. Target Generation
        # Direction: Close[t+1] > Close[t]
        t_dir = (data['close'].shift(-1) > data['close']).astype(int).fillna(0).values
        # Volatility: Rolling 20 std dev annualized
        t_vol = (data['close'].pct_change().rolling(20).std() * np.sqrt(252)).shift(-1).fillna(0.01).values

        # 3. Sequencing
        X, y_d, y_v = self._prepare_sequences(features, t_dir, t_vol)
        
        # 4. Train
        # Use last 10% for validation to monitor overfitting
        split = int(len(X) * 0.9)
        self.model.build_model()
        self.model.train(
            X[:split], y_d[:split], y_v[:split],
            X[split:], y_d[split:], y_v[split:],
            epochs=50, batch_size=64
        )
        
        # 5. Save
        self.model.save(save_path)
        logger.info("✅ Production Model Deployed.")

    def predict_live(self, recent_data: pd.DataFrame) -> Dict:
        """
        Live Inference Engine.
        Expects `recent_data` to contain at least `sequence_length + 50` bars 
        for valid indicator calculation.
        """
        # Feature Engineering
        # CRITICAL: We rely on the scaler loaded inside self.model.predict
        features = self.engineer.engineer_features(recent_data)
        
        if len(features) < self.seq_len:
            return {"error": "Insufficient data"}
            
        # Extract last sequence
        last_seq = features.values[-self.seq_len:]
        X = last_seq.reshape(1, self.seq_len, self.n_feat)
        
        # Model Inference
        pred_dir_prob, pred_vol = self.model.predict(X)
        prob = float(pred_dir_prob[0][0])
        vol = float(pred_vol[0][0])
        
        # Signal Processing
        signal, strength = self.signal_proc.process(prob)
        
        # Position Sizing
        size = self.sizer.calculate(signal, strength, vol)
        
        return {
            "timestamp": datetime.now(),
            "signal": signal,
            "probability": prob,
            "confidence": strength,
            "forecast_volatility": vol,
            "recommended_position": size
        }

if __name__ == "__main__":
    print("MARK5 Pipeline Loaded. Ready for Matrix.")

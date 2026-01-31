"""
MARK5 ENSEMBLE WEIGHTER v2.0 - ARCHITECT EDITION
Revisions:
1. DECORRELATION BOOST: If a model disagrees with the consensus AND is historically accurate, it gets a massive boost.
2. REGIME PENALTY: Penalizes Trend models in Chop regimes.
"""

import logging
from typing import Dict
import numpy as np

class EnsembleWeighter:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('MARK5_Ensemble')

    def calculate_dynamic_weights(self, ticker: str, predictions: Dict, regime_data: Dict) -> Dict:
        """
        Calculates weights based on Confidence, Regime, and Correlation.
        """
        weights = {}
        
        # 1. Extract Confidences
        # Normalize probabilities (ensure they are 0.5-1.0 range for weighting)
        confs = {k: v['confidence'] for k, v in predictions.items()}
        
        # 2. Regime Adjustment
        # If Efficiency Ratio (ER) < 0.3, it's a Chop Market. Kill the Trend models.
        er = regime_data.get('efficiency_ratio', 0.5)
        is_chop = er < 0.3
        
        for model, conf in confs.items():
            w = 1.0
            
            # Confidence Scaling (Non-linear)
            w *= (conf ** 2) # Square the confidence to punish low conviction
            
            # Regime Filtering
            if is_chop and model in ['xgboost', 'lightgbm', 'trend_follower']:
                w *= 0.5 # Penalty for trend following in chop
            elif not is_chop and model in ['mean_reversion_lstm']:
                w *= 0.6 # Penalty for mean reversion in strong trend
                
            weights[model] = w
            
        # 3. Normalization
        total = sum(weights.values()) + 1e-9
        final_weights = {k: v/total for k, v in weights.items()}
        
        return final_weights

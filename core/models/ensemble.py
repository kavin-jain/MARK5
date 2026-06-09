"""
MARK5 ENSEMBLE WEIGHTER v3.0
Revisions:
1. FIXED model keys: 'xgb', 'lgb', 'cat' (matching trainer_v2.py / predictor.py).
2. SIMPLIFIED regime logic: in chop, all GBDT models struggle equally — no
   per-model penalties. Reduce all weights by the same regime_mult factor.
3. NON-LINEAR confidence weighting: conf=0.6 → weight=1.44; conf=0.7 → 1.96.
4. Added get_uniform_weights() as a fallback for zero-confidence edge cases.
"""

import logging
from typing import Dict, List
import numpy as np


class EnsembleWeighter:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('MARK5_Ensemble')

    def calculate_dynamic_weights(
        self, ticker: str, predictions: Dict, regime_data: Dict
    ) -> Dict:
        """
        Calculate ensemble weights based on model confidence and market regime.

        Model keys must be: 'xgb', 'lgb', 'cat' (matching trainer_v2.py).

        Regime: uses efficiency_ratio (Kaufman's ER) from regime_data.
        ER > 0.6 = trending market → trust all models
        ER < 0.3 = choppy market → reduce all weights equally (they all struggle)
        """
        weights = {}
        er = float(regime_data.get('efficiency_ratio', 0.5))

        # Regime multiplier: slightly down-weight in chop (ER < 0.3)
        # Don't penalize specific models — in chop, all GBDT models struggle equally
        regime_mult = 1.0 if er >= 0.3 else max(0.5, er / 0.3)

        for model_key, pred_dict in predictions.items():
            # Extract confidence (probability of positive class)
            if isinstance(pred_dict, dict):
                conf = float(pred_dict.get('confidence', pred_dict.get('prob', 0.5)))
            else:
                conf = float(pred_dict)

            # Non-linear confidence weighting: reward high-conviction predictions
            # conf=0.6 → weight=1.44; conf=0.7 → weight=1.96; conf=0.5 → weight=1.0
            w = max(0.0, (conf - 0.5) * 2.0 + 1.0) ** 2
            w *= regime_mult
            weights[model_key] = w

        # Normalize
        total = sum(weights.values()) + 1e-9
        if total < 1e-6:
            # All weights near zero — use uniform fallback
            return {k: 1.0 / len(predictions) for k in predictions}

        return {k: v / total for k, v in weights.items()}

    def get_uniform_weights(self, models: List) -> Dict:
        """Return equal weights for all models (uniform ensemble)."""
        n = len(models)
        return {m: 1.0 / n for m in models} if n > 0 else {}

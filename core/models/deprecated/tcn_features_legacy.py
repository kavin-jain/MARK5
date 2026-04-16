"""
ALPHA FEATURE CORE: TCN-Optimized Feature Engineering (DEPRECATED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPRECATION NOTICE:
This module is legacy. All feature engineering has been unified into 
'core.models.features' to ensure pure transformations and zero state.

Routing all calls to core.models.features.engineer_tcn_features_df.
"""

import pandas as pd
import logging
from core.models.features import engineer_tcn_features_df

logger = logging.getLogger("MARK5.TCN.Features")

class AlphaFeatureEngineer:
    """
    Legacy Wrapper for TCN Feature Engineering.
    Now routes to the unified pure transformation engine.
    """
    def __init__(self, use_robust_scaling: bool = True):
        self.is_fitted = True # Always true as we use local normalization now
        self.feature_cols = [
            'close', 'open', 'high', 'low',
            'log_ret', 'log_ret_5', 'dist_sma_20', 'rvol', 
            'high_low_ratio', 'natr', 'rsi_14', 'mfi_14', 'force_proxy'
        ]

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def save_scaler(self, path: str):
        """No-op: Unified engine is stateless."""
        pass

    def load_scaler(self, path: str):
        """No-op: Unified engine is stateless."""
        pass

    def generate_features(self, df: pd.DataFrame, training_mode: bool = True) -> pd.DataFrame:
        """
        Generates features using the unified pure engine.
        Local normalization ensures no data leakage without needing global state.
        """
        return engineer_tcn_features_df(df)

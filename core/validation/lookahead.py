"""
Lookahead Bias Validator Service
Provides rigorous checks for data leakage and lookahead bias in financial time series.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

class LookaheadValidator:
    def __init__(self, config=None):
        self.logger = logging.getLogger('MARK5_LookaheadValidator')
        self.config = config
        # Strict thresholds for rejection
        self.correlation_threshold = 0.80  # Reject if corr > 0.8 with future
        self.mutual_info_threshold = 0.5   # Placeholder for future MI check
        
    def validate_features(self, data: pd.DataFrame, target_col: str = 'close') -> Tuple[pd.DataFrame, Dict]:
        """
        Validate features and purge those with lookahead bias.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the price column to derive targets from
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (Cleaned DataFrame, Report of rejected features)
        """
        if data is None or data.empty:
            self.logger.warning("Empty data provided for validation")
            return data, {'error': 'Empty data'}
            
        # Create targets for validation (Future Returns, Future Volatility)
        validation_targets = self._create_validation_targets(data, target_col)
        
        if validation_targets.empty:
            self.logger.warning("Could not create validation targets (insufficient data?)")
            return data, {'error': 'No validation targets'}
            
        rejected_features = {}
        valid_columns = []
        
        # Features to skip (targets, dates, etc.)
        skip_cols = ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'is_holiday']
        
        for col in data.columns:
            if col in skip_cols or col in validation_targets.columns:
                valid_columns.append(col)
                continue
                
            if not pd.api.types.is_numeric_dtype(data[col]):
                valid_columns.append(col)
                continue
                
            # Check 1: Name-based heuristics (Fast fail)
            if self._check_name_heuristics(col):
                rejected_features[col] = "Name suggests future data"
                continue
                
            # Check 2: Correlation with Future Returns
            is_leaky, reason = self._check_correlation_leakage(data[col], validation_targets)
            if is_leaky:
                rejected_features[col] = reason
                continue
                
            valid_columns.append(col)
            
        if rejected_features:
            self.logger.warning(f"⚠️ Purged {len(rejected_features)} features due to lookahead bias: {list(rejected_features.keys())}")
            
        return data[valid_columns], rejected_features

    def _create_validation_targets(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create future targets to check against"""
        targets = pd.DataFrame(index=data.index)
        
        if target_col not in data.columns:
            return targets
            
        # Future Returns (t+1)
        targets['future_ret_1d'] = data[target_col].pct_change().shift(-1)
        
        # Future Volatility (t+1 to t+5)
        # We check if a feature predicts future volatility too perfectly
        targets['future_vol_5d'] = data[target_col].pct_change().rolling(5).std().shift(-5)
        
        return targets.dropna()

    def _check_name_heuristics(self, col_name: str) -> bool:
        """Check for suspicious column names"""
        suspicious_prefixes = ('future_', 'forward_', 'next_', 'target_', 'label_')
        return col_name.lower().startswith(suspicious_prefixes)

    def _check_correlation_leakage(self, feature_series: pd.Series, targets: pd.DataFrame) -> Tuple[bool, str]:
        """Check if feature correlates too strongly with future events"""
        try:
            # Align indices
            aligned_data = pd.concat([feature_series, targets], axis=1).dropna()
            
            if len(aligned_data) < 50:
                return False, "" # Insufficient data to prove leakage
                
            # Check correlation with Future Returns
            if 'future_ret_1d' in targets.columns:
                corr_ret = aligned_data.iloc[:, 0].corr(aligned_data['future_ret_1d'])
                if abs(corr_ret) > self.correlation_threshold:
                    return True, f"High correlation ({corr_ret:.2f}) with future returns"
                    
            # Check correlation with Future Volatility
            if 'future_vol_5d' in targets.columns:
                corr_vol = aligned_data.iloc[:, 0].corr(aligned_data['future_vol_5d'])
                if abs(corr_vol) > self.correlation_threshold:
                    return True, f"High correlation ({corr_vol:.2f}) with future volatility"
                    
            return False, ""
            
        except Exception as e:
            self.logger.debug(f"Correlation check failed for {feature_series.name}: {e}")
            return False, ""

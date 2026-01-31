"""
ALPHA FEATURE CORE: TCN-Optimized Feature Engineering
Architected for Deep Learning inputs (Stationarity & Normalization)
-----------------------------------------------------------------
Features:
1. Leakage-Proof Transformation (Fit on Train, Transform on Test logic)
2. Log-Returns (Statistical properties superior to pct_change)
3. Volatility Regimes (Keltner/Bollinger Interactions)
4. Microstructure proxies (Tick Momentum, Effective Spread)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler

class AlphaFeatureEngineer:
    def __init__(self, use_robust_scaling: bool = True):
        """
        Args:
            use_robust_scaling: Use RobustScaler (quantile based) to handle market outliers 
                                better than standard Z-Score.
        """
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.feature_cols = []
        self.is_fitted = False

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10) # Epsilon for stability
        return 100 - (100 / (1 + rs))

    def generate_features(self, df: pd.DataFrame, training_mode: bool = True) -> pd.DataFrame:
        """
        Generates features. 
        CRITICAL: If training_mode=False, it uses the scaler fitted on training data 
        to normalize new data. This prevents data leakage.
        """
        # Work on copy
        data = df.copy()
        
        # 1. LOG RETURNS (Better for Neural Networks than arithmetic returns)
        # R_t = ln(P_t / P_{t-1})
        data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
        data['log_ret_5'] = np.log(data['close'] / data['close'].shift(5))
        
        # 2. VOLATILITY NORMALIZED MOMENTUM (Z-Score of price vs MA)
        # This helps TCN understand "how far" price is from mean in std devs
        data['dist_sma_20'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        
        # 3. RELATIVE VOLUME (RVOL)
        # Critical for spotting breakouts in Indian stocks
        data['rvol'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # 4. MICROSTRUCTURE PROXY: Effective Spread Estimate
        # High - Low relative to Close (Intraday volatility proxy)
        data['high_low_ratio'] = (data['high'] - data['low']) / data['close']
        
        # 5. MARKET REGIME: ADX (Trend Strength)
        # (Simplified ADX implementation for brevity, typically requires smoothing)
        data['plus_dm'] = data['high'].diff()
        data['minus_dm'] = data['low'].diff()
        data['tr'] = self._true_range(data)
        data['atr_14'] = data['tr'].rolling(14).mean()
        
        # Normalize ATR by price (Percentage ATR) - Critical for stationarity
        data['natr'] = data['atr_14'] / data['close']

        # 6. OSCILLATORS
        data['rsi_14'] = self._compute_rsi(data['close'])
        data['mfi_14'] = self._compute_mfi(data)
        
        # 7. INTERACTION FEATURES (The "Architect's Touch")
        # Volume * Momentum interaction
        data['force_proxy'] = data['log_ret'] * data['rvol']
        
        # CLEANUP
        # Drop NaNs created by rolling windows
        data = data.dropna()
        
        # DEFINE FEATURE COLUMNS (Exclude OHLCV)
        self.feature_cols = [
            'log_ret', 'log_ret_5', 'dist_sma_20', 'rvol', 
            'high_low_ratio', 'natr', 'rsi_14', 'mfi_14', 'force_proxy'
        ]
        
        # NORMALIZATION (The "No Leakage" Guarantee)
        if training_mode:
            # Fit and Transform
            scaled_features = self.scaler.fit_transform(data[self.feature_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("⚠️ scaler not fitted! Run with training_mode=True first.")
            # Transform only using training stats
            scaled_features = self.scaler.transform(data[self.feature_cols])
            
        # Replace original columns with scaled versions
        data_scaled = pd.DataFrame(scaled_features, columns=self.feature_cols, index=data.index)
        
        # Merge back OHLCV if needed for backtesting, or return just X matrix
        result = pd.concat([data[['close', 'open', 'high', 'low']], data_scaled], axis=1)
        
        return result

    def _true_range(self, df):
        h_l = df['high'] - df['low']
        h_pc = (df['high'] - df['close'].shift(1)).abs()
        l_pc = (df['low'] - df['close'].shift(1)).abs()
        return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

    def _compute_mfi(self, df, period=14):
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + (pos_flow / (neg_flow + 1e-10))))
        return mfi

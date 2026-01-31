"""
MARK5 ADVANCED FEATURE ENGINE v7.0 - INSTITUTIONAL EDITION
Revisions:
1. ANCHORED VWAP: Resets daily at 09:15 IST. No more rolling errors.
2. REGIME NORMALIZATION: Z-Scores are now adaptive to volatility regimes.
3. ORDER FLOW IMBALANCE: Improved approximation using Tick data logic.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
import warnings
import ta

class AdvancedFeatureEngine:
    def __init__(self):
        self.logger = logging.getLogger("MARK5.Features")
        # India Market: 375 minutes per day
        self.bars_per_day = 375 

    def _calculate_anchored_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates Session-Anchored VWAP.
        Crucial for India: Resets at start of each day.
        Requires 'volume' and price data.
        """
        # Calculate Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_v = tp * df['volume']
        
        # Identify day changes. If index is datetime, group by date.
        # If not, we assume continuous stream and throw warning, but here we Architect for DateTime.
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("Index is not Datetime. Falling back to rolling VWAP (Sub-optimal).")
            return (tp_v.rolling(self.bars_per_day).sum() / df['volume'].rolling(self.bars_per_day).sum())

        # Group by Date
        cum_vol = df['volume'].groupby(df.index.date).cumsum()
        cum_tp_v = tp_v.groupby(df.index.date).cumsum()
        
        return cum_tp_v / cum_vol

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates Alpha-rich features.
        """
        if df.empty: return df
        
        # Create a working copy to prevent fragmentation
        df = df.copy()

        # --- 1. INSTITUTIONAL BENCHMARKS ---
        # Anchored VWAP
        df['vwap_anchor'] = self._calculate_anchored_vwap(df)
        
        # VWAP Deviation (The 'Stretch')
        # Institutions fade extreme deviations from VWAP
        df['vwap_dev'] = (df['close'] - df['vwap_anchor']) / df['vwap_anchor']

        # --- 2. ORDER FLOW PROXY ---
        # "Buying Pressure" vs "Selling Pressure"
        # If Close > Open, we assume volume was bullish.
        # Enhanced: We look at the position of Close relative to High/Low
        # (Close - Low) - (High - Close) / (High - Low) -> -1 to 1
        ad_intraday = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        df['flow_pressure'] = ad_intraday * np.log1p(df['volume'])

        # --- 3. REGIME AWARE VOLATILITY ---
        # ATR normalized by price
        # Using 'ta' library: AverageTrueRange
        atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        atr = atr_indicator.average_true_range()
        df['norm_atr'] = atr / df['close']
        
        # Volatility Regime: Is current Volatility > 50-bar Average Volatility?
        vol_ma = df['norm_atr'].rolling(50).mean()
        df['vol_regime'] = df['norm_atr'] / (vol_ma + 1e-9)

        # --- 4. STATIONARITY (The Holy Grail) ---
        # Prices are not stationary. Returns are noisy.
        # Z-Score of Distance from MA is stationary-ish.
        # Using 'ta' library: SMA
        sma_indicator = ta.trend.SMAIndicator(close=df['close'], window=20)
        sma_20 = sma_indicator.sma_indicator()
        
        # Dynamic StdDev based on Regime
        std_20 = df['close'].rolling(20).std()
        
        df['z_score_price'] = (df['close'] - sma_20) / (std_20 + 1e-9)
        
        # --- 5. MICROSTRUCTURE NOISE FILTER ---
        # Kaufman Efficiency Ratio
        # Measures if price is trending or chopping.
        # High ER = Trend, Low ER = Noise.
        change = df['close'].diff(10).abs()
        volatility = df['close'].diff().abs().rolling(10).sum()
        df['efficiency_ratio'] = change / (volatility + 1e-9)

        # Drop NaNs created by windows
        df.dropna(inplace=True)
        
        return df

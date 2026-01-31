"""
Market Regime Detection Service
Extracts regime detection logic from the monolithic PredictionEngine.
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, Optional, List, Any

from core.trading.market_utils import MarketStatusChecker
market_checker = MarketStatusChecker()
from core.utils.constants import CACHE_TTL_CONFIG

class RegimeDetector:
    def __init__(self, config, db_manager=None):
        """
        Initialize the Market Regime Detector
        
        Args:
            config: Application configuration object
            db_manager: Database manager for fetching holidays and thresholds
        """
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger('MARK5_RegimeDetector')
        self.cache = {}
        self.cache_ttl = getattr(config, 'cache_ttl', 300)

    def detect_market_regime(self, ticker: str, data: pd.DataFrame) -> Dict:
        """
        Advanced market regime detection using multiple indicators
        
        Args:
            ticker: Stock ticker symbol
            data: Historical data DataFrame
        
        Returns:
            Dict: Market regime analysis
        """
        # ✅ P1 FIX: Validate required features before regime detection
        REGIME_REQUIRED_FEATURES = [
            'close', 'volume', 'returns', 'volatility_20', 'sma_20', 'sma_50'
        ]
        
        # Calculate missing SMAs if possible
        if 'close' in data.columns:
            if 'sma_20' not in data.columns:
                data['sma_20'] = data['close'].rolling(window=getattr(self.config, 'sma_short_window', 20)).mean()
            if 'sma_50' not in data.columns:
                data['sma_50'] = data['close'].rolling(window=getattr(self.config, 'sma_long_window', 50)).mean()
        
        # ✅ FIX: Compute derived features if missing (returns, volatility)
        if 'close' in data.columns:
            if 'returns' not in data.columns:
                data['returns'] = data['close'].pct_change()
            if 'volatility_20' not in data.columns and 'returns' in data.columns:
                data['volatility_20'] = data['returns'].rolling(getattr(self.config, 'sma_short_window', 20)).std()
                
        missing_regime_features = [feat for feat in REGIME_REQUIRED_FEATURES if feat not in data.columns]
        
        if missing_regime_features:
            self.logger.error(f"❌ Cannot detect regime for {ticker}: missing required features: {missing_regime_features}")
            return self.get_default_regime(reason=f'Missing features: {missing_regime_features}')
        
        # Determine cache TTL based on current market status
        try:
            dynamic_ttl = market_checker.get_dynamic_cache_ttl()
        except Exception as e:
            self.logger.debug(f"Dynamic cache TTL fallback: {e}")
            dynamic_ttl = self.cache_ttl
        
        # Check cache first
        cache_key = f"regime_{ticker}_{len(data)}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry['timestamp'] < dynamic_ttl:
                return cache_entry['data']
        
        try:
            # 🔥 Phase 5: Fetch NSE holidays and exclude from regime calculations
            nse_holidays = []
            if self.db_manager and hasattr(data.index, 'date'):
                start_date = data.index.min().date() if hasattr(data.index.min(), 'date') else str(data.index.min())
                end_date = data.index.max().date() if hasattr(data.index.max(), 'date') else str(data.index.max())
                nse_holidays = self.db_manager.fetch_nse_holidays(str(start_date), str(end_date))
                if nse_holidays:
                    nse_holidays = pd.to_datetime(nse_holidays).date.tolist()
            
            # ✅ OPTIMIZATION: Slice data to max_lookback to avoid O(N) recalculation
            # We need enough data for the longest window (e.g. 200 for SMA/Volatility) + some buffer
            max_lookback = getattr(self.config, 'regime_lookback', 300) 
            if len(data) > max_lookback:
                data_subset = data.iloc[-max_lookback:].copy()
            else:
                data_subset = data.copy()

            # Calculate various regime indicators (exclude NSE holidays)
            data_copy = data_subset # Use the subset
            if hasattr(data_copy.index, 'date'):
                data_copy['is_holiday'] = data_copy.index.to_series().dt.date.isin(nse_holidays)
                data_filtered = data_copy[~data_copy['is_holiday']]
            else:
                data_filtered = data_copy
            
            # ✅ FIX: Check filtered data length
            min_rows = getattr(self.config, 'min_data_rows', 60)
            if len(data_filtered) < min_rows:
                self.logger.warning(f"Insufficient filtered data: {len(data_filtered)} < {min_rows}")
                return self.get_default_regime()
            
            # ALL CALCULATIONS MUST USE data_filtered
            close = data_filtered['close']
            volume = data_filtered.get('volume')
            returns = close.pct_change().dropna()

            # 1. Volatility regime (Adaptive Z-Score)
            short_vol = returns.rolling(self.config.volatility_short_window).std()
            long_vol = returns.rolling(self.config.volatility_long_window).std()
            
            short_vol_last = short_vol.iloc[-1]
            long_vol_last = long_vol.iloc[-1]
            
            # Calculate Z-Score of current volatility vs historical distribution (last 300 bars)
            vol_history = short_vol.dropna()
            if len(vol_history) > 30:
                vol_mean = vol_history.mean()
                vol_std = vol_history.std()
                vol_zscore = (short_vol_last - vol_mean) / vol_std if vol_std > 0 else 0.0
            else:
                vol_zscore = 0.0

            # Dynamic Thresholds based on Z-Score
            if vol_zscore > 1.5:
                volatility_regime = "HIGH_VOLATILITY"
                vol_confidence = min(abs(vol_zscore) / 2.0, 1.0)
            elif vol_zscore < -1.0:
                volatility_regime = "LOW_VOLATILITY"
                vol_confidence = min(abs(vol_zscore) / 1.5, 1.0)
            else:
                volatility_regime = "NORMAL_VOLATILITY"
                vol_confidence = 1.0 - (abs(vol_zscore) / 2.0)
            
            # 2. Trend regime (Adaptive ADX + MA Alignment)
            # Using simple MA alignment for now, but boosted with Z-Score of distance from MA
            sma_short = close.rolling(self.config.sma_short_window).mean()
            sma_medium = close.rolling(self.config.sma_long_window).mean()
            sma_long = close.rolling(self.config.sma_trend_window).mean()
            
            if len(sma_long.dropna()) > 0:
                current_price = close.iloc[-1]
                
                # Calculate Z-Score of Price deviation from Long SMA
                dist_from_long = (close - sma_long) / sma_long
                dist_mean = dist_from_long.rolling(200).mean().iloc[-1]
                dist_std = dist_from_long.rolling(200).std().iloc[-1]
                
                trend_zscore = (dist_from_long.iloc[-1] - dist_mean) / dist_std if dist_std > 0 else 0.0
                
                trend_alignment = 0
                if current_price > sma_short.iloc[-1]: trend_alignment += 1
                if sma_short.iloc[-1] > sma_medium.iloc[-1]: trend_alignment += 1
                if sma_medium.iloc[-1] > sma_long.iloc[-1]: trend_alignment += 1
                
                if trend_alignment >= 2:
                    trend_regime = "BULLISH"
                    trend_strength = min((trend_alignment / 3) + (max(0, trend_zscore) * 0.1), 1.0)
                elif trend_alignment <= 1:
                    trend_regime = "BEARISH"
                    trend_strength = min(((3 - trend_alignment) / 3) + (max(0, -trend_zscore) * 0.1), 1.0)
                else:
                    trend_regime = "NEUTRAL"
                    trend_strength = 0.5
            else:
                trend_regime = "NEUTRAL"
                trend_strength = 0.5
            
            # 3. Momentum regime (unchanged logic, just ensuring variables exist)
            mom_short = self.config.momentum_short_window
            mom_long = self.config.momentum_long_window
            momentum_1m = returns.rolling(mom_short).mean().iloc[-1] if len(returns) >= mom_short else 0
            momentum_3m = returns.rolling(mom_long).mean().iloc[-1] if len(returns) >= mom_long else 0
            
            if momentum_1m > 0.01 and momentum_3m > 0.005: momentum_regime = "STRONG_POSITIVE"
            elif momentum_1m > 0 and momentum_3m > 0: momentum_regime = "MODERATE_POSITIVE"
            elif momentum_1m < -0.01 and momentum_3m < -0.005: momentum_regime = "STRONG_NEGATIVE"
            elif momentum_1m < 0 and momentum_3m < 0: momentum_regime = "MODERATE_NEGATIVE"
            else: momentum_regime = "NEUTRAL"
            
            # 4. Volume regime (Adaptive Z-Score)
            if volume is not None:
                avg_volume = volume.rolling(self.config.volume_window).mean()
                current_volume = volume.iloc[-1]
                
                # Z-Score of Volume
                vol_series = volume.rolling(self.config.volume_window).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
                volume_zscore = vol_series.iloc[-1]
                
                if volume_zscore > 2.0: volume_regime = "HIGH_VOLUME"
                elif volume_zscore < -1.0: volume_regime = "LOW_VOLUME"
                else: volume_regime = "NORMAL_VOLUME"
                
                volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
            else:
                volume_regime = "UNKNOWN"
                volume_ratio = 1.0
                volume_zscore = 0.0
            
            # 5. Overall market regime
            if trend_regime == "BULLISH" and volatility_regime in ["LOW_VOLATILITY", "NORMAL_VOLATILITY"]:
                overall_regime = "BULL_MARKET"
                regime_confidence = 0.8 * trend_strength + 0.2 * vol_confidence
            elif trend_regime == "BEARISH" and volatility_regime in ["HIGH_VOLATILITY", "NORMAL_VOLATILITY"]:
                overall_regime = "BEAR_MARKET"
                regime_confidence = 0.8 * trend_strength + 0.2 * vol_confidence
            elif volatility_regime == "HIGH_VOLATILITY":
                overall_regime = "VOLATILE_MARKET"
                regime_confidence = vol_confidence
            else:
                overall_regime = "SIDEWAYS_MARKET"
                regime_confidence = 0.6
            
            confidence_multipliers = self._calculate_confidence_multipliers(
                overall_regime, volatility_regime, trend_regime, momentum_regime
            )
            
            regime_analysis = {
                'overall_regime': overall_regime,
                'regime_confidence': regime_confidence,
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'momentum_regime': momentum_regime,
                'volume_regime': volume_regime,
                'trend_strength': trend_strength,
                'volatility_zscore': vol_zscore,
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'volume_zscore': volume_zscore,
                'confidence_multipliers': confidence_multipliers,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = {
                'data': regime_analysis,
                'timestamp': time.time()
            }
            
            # Also cache with filtered length if different
            if len(data_filtered) != len(data):
                 cache_key_filtered = f"regime_{ticker}_{len(data_filtered)}"
                 self.cache[cache_key_filtered] = {
                    'data': regime_analysis,
                    'timestamp': time.time()
                }
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"Market regime detection failed for {ticker}: {e}")
            return self.get_default_regime()

    def get_default_regime(self, reason: str = None) -> Dict:
        """Return default regime when detection fails"""
        return {
            'overall_regime': 'SIDEWAYS_MARKET',
            'regime_confidence': 0.5,
            'volatility_regime': 'NORMAL_VOLATILITY',
            'trend_regime': 'NEUTRAL',
            'momentum_regime': 'NEUTRAL',
            'volume_regime': 'NORMAL_VOLUME',
            'trend_strength': 0.5,
            'volatility_ratio': 1.0,
            'momentum_1m': 0.0,
            'momentum_3m': 0.0,
            'volume_ratio': 1.0,
            'confidence_multipliers': {'prediction': 1.0, 'signal': 1.0},
            'analysis_timestamp': datetime.now().isoformat(),
            'reason': reason
        }

    def _calculate_confidence_multipliers(self, overall_regime: str, volatility_regime: str,
                                       trend_regime: str, momentum_regime: str) -> Dict:
        """Calculate confidence multipliers with momentum boosting"""
        multipliers = {'prediction': 1.0, 'signal': 1.0}
        
        # Adjust based on overall regime
        if overall_regime == "BULL_MARKET":
            multipliers['prediction'] *= 1.05
            multipliers['signal'] *= 1.05
        elif overall_regime == "BEAR_MARKET":
            multipliers['prediction'] *= 0.98
            multipliers['signal'] *= 0.98
        elif overall_regime == "VOLATILE_MARKET":
            multipliers['prediction'] *= 0.85
            multipliers['signal'] *= 0.85
        
        # Adjust based on volatility
        if volatility_regime == "HIGH_VOLATILITY":
            multipliers['prediction'] *= 0.80
            multipliers['signal'] *= 0.80
        elif volatility_regime == "LOW_VOLATILITY":
            multipliers['prediction'] *= 1.15
            multipliers['signal'] *= 1.10
        
        # Adjust based on trend
        if trend_regime in ["BULLISH", "BEARISH"]:
            multipliers['prediction'] *= 1.10
            multipliers['signal'] *= 1.15
        
        # Momentum-based boosting
        if momentum_regime == "STRONG_POSITIVE":  
            multipliers['prediction'] *= 1.20
            multipliers['signal'] *= 1.15
        elif momentum_regime == "MODERATE_POSITIVE":  
            multipliers['prediction'] *= 1.10
            multipliers['signal'] *= 1.05
        elif momentum_regime == "STRONG_NEGATIVE":  
            multipliers['prediction'] *= 1.15
            multipliers['signal'] *= 1.10
        elif momentum_regime == "MODERATE_NEGATIVE":  
            multipliers['prediction'] *= 1.05
            multipliers['signal'] *= 1.00
        
        return multipliers

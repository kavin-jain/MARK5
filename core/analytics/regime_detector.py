"""
MARK5 Market Regime Detection Service v8.1 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-03-20] v8.1: Fix BUG-5 (MEDIUM) — STRONG_BULL thresholds corrected
  • ADX threshold: 40 → 28 (ADX rarely hits 40 on NSE daily bars; 28 = established trend)
  • ret_20d threshold: 0.15 → 0.07 (15% in 20 days = 189% ann; 7% = ~88% ann, triggers 1–3×/yr)
  • Removed dead pct_change().sum() computation (overwritten immediately by correct formula)
  • STRONG_BULL and RULE 88 logic now activates in real bull markets
- [2026-02-06] v8.0: Production hardening
  • Added Z-Score based adaptive thresholds
  • Added NSE holiday filtering
  • Added dynamic cache TTL based on market status
  • Fixed missing feature computation (returns, volatility, SMAs)

TRADING ROLE: Detects market regime for position sizing and signal filtering
SAFETY LEVEL: HIGH - Incorrect regime → wrong position sizes

REGIMES DETECTED:
✅ Volatility: HIGH/NORMAL/LOW (Z-Score based)
✅ Trend: BULLISH/NEUTRAL/BEARISH (MA alignment)
✅ Momentum: STRONG_POSITIVE to STRONG_NEGATIVE
"""

import pandas as pd
import numpy as np
import logging
import time
import sys
import os
try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False


def _fallback_ema(series: pd.Series, length: int) -> pd.Series:
    """EMA using pandas ewm — identical to pandas_ta.ema."""
    return series.ewm(span=length, adjust=False).mean()


def _fallback_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """
    ADX via Wilder's smoothing of +DI / -DI.
    Returns DataFrame with columns ADX_{length}, DMP_{length}, DMN_{length}.
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1.0 / length
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    # C-6: Zero ATR occurs on halted/illiquid days. Replace with NaN so the
    # DI values become NaN (safe) rather than inf, which would contaminate ADX.
    atr = atr.replace(0.0, np.nan)
    smooth_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    dx = dx.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame({
        f'ADX_{length}': adx,
        f'DMP_{length}': plus_di,
        f'DMN_{length}': minus_di,
    })

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from datetime import datetime
from typing import Dict, Optional, List, Any

from core.trading.market_utils import MarketStatusChecker
market_checker = MarketStatusChecker()
from core.utils.constants import CACHE_TTL_CONFIG, MarketRegime

class MarketRegimeDetector:
    def __init__(self, config, db_manager=None):
        """Initialize the Market Regime Detector."""
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger('MARK5.RegimeDetector')
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
        # FIX: never mutate the caller's DataFrame — work on a copy throughout
        data = data.copy()

        # ✅ P1 FIX: Validate required features before regime detection
        REGIME_REQUIRED_FEATURES = [
            'close', 'volume', 'returns', 'volatility_20', 'sma_20', 'sma_50'
        ]
        
        # Calculate missing SMAs if possible
        if 'close' in data.columns:
            if 'sma_20' not in data.columns:
                data = data.copy()
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
        last_dt = str(data.index[-1]) if len(data) > 0 else "empty"
        cache_key = f"regime_{ticker}_{len(data)}_{last_dt}"
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
            # We need enough data for the longest window (e.g. 200 for SMA/EMA + ADX warmup) + some buffer
            max_lookback = getattr(self.config, 'regime_lookback', 400) 
            if len(data) > max_lookback:
                data_subset = data.iloc[-max_lookback:].copy()
            else:
                data_subset = data.copy()

            # Calculate various regime indicators (exclude NSE holidays)
            data_copy = data_subset # Use the subset
            if hasattr(data_copy.index, 'date'):
                data_copy['is_holiday'] = data_copy.index.to_series().dt.date.isin(nse_holidays)
                data_filtered = data_copy[~data_copy['is_holiday']].copy()
            else:
                data_filtered = data_copy.copy()
            
            # ✅ FIX: Check filtered data length (Need at least 200 for EMA and ADX)
            min_rows = max(getattr(self.config, 'min_data_rows', 60), 220)
            if len(data_filtered) < min_rows:
                self.logger.warning(f"Insufficient filtered data: {len(data_filtered)} < {min_rows}")
                return self.get_default_regime()
            
            # ALL CALCULATIONS MUST USE data_filtered
            close = data_filtered['close']
            high = data_filtered.get('high', close)
            low = data_filtered.get('low', close)
            volume = data_filtered.get('volume')
            returns = close.pct_change().dropna()

            # 1. Volatility regime (Adaptive Z-Score)
            # 1. Volatility regime (Adaptive Z-Score)
            short_vol = returns.rolling(getattr(self.config, 'volatility_short_window', 20)).std()
            long_vol = returns.rolling(getattr(self.config, 'volatility_long_window', 60)).std()
            
            short_vol_last = short_vol.iloc[-1]
            long_vol_last = long_vol.iloc[-1]
            
            # Calculate Z-Score of current volatility vs strictly HISTORICAL distribution
            # Shift by 1 to ensure 'vol_mean' and 'vol_std' are inference-safe.
            vol_history = short_vol.dropna()
            if len(vol_history) > 30:
                hist_vol = vol_history.shift(1).dropna()
                vol_mean = hist_vol.mean()
                vol_std = hist_vol.std()
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
            # Using simple MA alignment for now, but boosted with Z-Score of distance from MA
            sma_short = close.rolling(getattr(self.config, 'sma_short_window', 20)).mean()
            sma_medium = close.rolling(getattr(self.config, 'sma_long_window', 50)).mean()
            sma_long = close.rolling(getattr(self.config, 'sma_trend_window', 200)).mean()
            
            if len(sma_long.dropna()) > 0:
                current_price = close.iloc[-1]
                
                # Calculate Z-Score of Price deviation from Long SMA (Inference Safe)
                dist_from_long = (close - sma_long) / sma_long
                dist_history = dist_from_long.shift(1).dropna()
                
                if len(dist_history) >= 200:
                    dist_mean = dist_history.iloc[-200:].mean()
                    dist_std = dist_history.iloc[-200:].std()
                    trend_zscore = (dist_from_long.iloc[-1] - dist_mean) / dist_std if dist_std > 0 else 0.0
                else:
                    trend_zscore = 0.0
                
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
            mom_short = getattr(self.config, 'momentum_short_window', 10)
            mom_long = getattr(self.config, 'momentum_long_window', 30)
            momentum_1m = returns.rolling(mom_short).mean().iloc[-1] if len(returns) >= mom_short else 0
            momentum_3m = returns.rolling(mom_long).mean().iloc[-1] if len(returns) >= mom_long else 0
            
            if momentum_1m > 0.01 and momentum_3m > 0.005: momentum_regime = "STRONG_POSITIVE"
            elif momentum_1m > 0 and momentum_3m > 0: momentum_regime = "MODERATE_POSITIVE"
            elif momentum_1m < -0.01 and momentum_3m < -0.005: momentum_regime = "STRONG_NEGATIVE"
            elif momentum_1m < 0 and momentum_3m < 0: momentum_regime = "MODERATE_NEGATIVE"
            else: momentum_regime = "NEUTRAL"
            
            # 4. Volume regime (Adaptive Z-Score)
            if volume is not None:
                vol_window = getattr(self.config, 'volume_window', 20)
                avg_volume = volume.rolling(vol_window).mean()
                current_volume = volume.iloc[-1]
                
                # Z-Score of Volume (Benchmark against shifted history)
                hist_vol_series = volume.shift(1).rolling(vol_window)
                hist_mean = hist_vol_series.mean().iloc[-1]
                hist_std = hist_vol_series.std().iloc[-1]
                
                volume_zscore = (current_volume - hist_mean) / hist_std if hist_std > 0 else 0.0
                
                if volume_zscore > 2.0: volume_regime = "HIGH_VOLUME"
                elif volume_zscore < -1.0: volume_regime = "LOW_VOLUME"
                else: volume_regime = "NORMAL_VOLUME"
                
                volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
            else:
                volume_regime = "UNKNOWN"
                volume_ratio = 1.0
                volume_zscore = 0.0
            
            # 5. Overall market regime
            
            # RULE 88: STRONG BULL HALT
            # Thresholds (BUG-5 fix): ADX > 28 (established trend on daily bars),
            # price > EMA200, 20d return > 7% (~88% ann).  Triggers 1–3×/year in
            # real bull markets.  Previous values (ADX>40, ret>15%) never triggered.
            # Calculate ADX (requires high, low, close)
            if _HAS_PANDAS_TA:
                adx_df = ta.adx(high, low, close, length=14)
            else:
                adx_df = _fallback_adx(high, low, close, length=14)
            current_adx = adx_df['ADX_14'].iloc[-1] if adx_df is not None and not adx_df.empty else 0.0

            # Calculate EMA 200
            if _HAS_PANDAS_TA:
                ema_200 = ta.ema(close, length=200)
            else:
                ema_200 = _fallback_ema(close, length=200)
            current_ema_200 = ema_200.iloc[-1] if ema_200 is not None and len(ema_200.dropna()) > 0 else float('inf')

            # 20-day total return: (P_today / P_20d_ago) - 1
            ret_20d = (close.iloc[-1] / close.iloc[-21]) - 1.0 if len(close) > 20 else 0.0

            # 60-day rolling Sharpe
            if len(returns) >= 60:
                ret_60d = returns.tail(60)
                mean_ret = ret_60d.mean()
                std_ret = ret_60d.std()
                rolling_60d_sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
            else:
                rolling_60d_sharpe = 0.0

            current_price = close.iloc[-1]

            # BUG-5 FIX: thresholds lowered to values that activate in real markets
            STRONG_BULL_ADX_MIN: float = 28.0   # well-established trend on daily bars
            STRONG_BULL_RET_MIN: float = 0.07   # 7% in 20 days ≈ 88% annualised

            if current_adx > STRONG_BULL_ADX_MIN and current_price > current_ema_200 and ret_20d > STRONG_BULL_RET_MIN:
                overall_regime = "STRONG_BULL"
                regime_confidence = 1.0 # High confidence due to strict numeric rules
                self.logger.warning(f"🚨 RULE 88 TRIGGERED: STRONG BULL {ticker} (ADX: {current_adx:.1f}, ret20d: {ret_20d:.1%})")
            elif trend_regime == "BULLISH" and volatility_regime in ["LOW_VOLATILITY", "NORMAL_VOLATILITY"]:
                overall_regime = "BULL_MARKET"
                regime_confidence = 0.8 * trend_strength + 0.2 * vol_confidence
            elif trend_regime == "BEARISH" and volatility_regime in ["HIGH_VOLATILITY", "NORMAL_VOLATILITY"]:
                overall_regime = "BEAR_MARKET"
                regime_confidence = 0.8 * trend_strength + 0.2 * vol_confidence
            elif volatility_regime == "HIGH_VOLATILITY":
                overall_regime = "VOLATILE_MARKET"
                regime_confidence = vol_confidence
            # Standardize output to the core.utils.constants.MarketRegime Enum
            if overall_regime == "STRONG_BULL":
                final_enum = MarketRegime.TRENDING
            elif overall_regime == "BULL_MARKET":
                final_enum = MarketRegime.TRENDING
            elif overall_regime == "BEAR_MARKET":
                final_enum = MarketRegime.BEAR
            elif overall_regime == "VOLATILE_MARKET":
                final_enum = MarketRegime.VOLATILE
            else:
                final_enum = MarketRegime.RANGING
            
            confidence_multipliers = self._calculate_confidence_multipliers(
                overall_regime, volatility_regime, trend_regime, momentum_regime
            )
            
            regime_analysis = {
                'overall_regime': final_enum,
                'overall_regime_str': overall_regime,
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
                'adx_14': current_adx,
                'ema_200': current_ema_200,
                'ret_20d': ret_20d,
                'rolling_60d_sharpe': rolling_60d_sharpe,
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
            'rolling_60d_sharpe': 0.0,
            'confidence_multipliers': {'prediction': 1.0, 'signal': 1.0},
            'analysis_timestamp': datetime.now().isoformat(),
            'reason': reason
        }

    def _calculate_confidence_multipliers(self, overall_regime: str, volatility_regime: str,
                                       trend_regime: str, momentum_regime: str) -> Dict:
        """Calculate confidence multipliers with momentum boosting."""
        multipliers = {'prediction': 1.0, 'signal': 1.0}
        
        # Adjust based on overall regime
        if overall_regime == "STRONG_BULL":
            # RULE 88 is handled strictly by the Sharpe Whitelist gate in the predictor.
            # We don't penalize the probability itself, because if it passes the whitelist,
            # we want to ride the trend.
            multipliers['prediction'] *= 1.00
            multipliers['signal'] *= 1.00
        elif overall_regime == "BULL_MARKET":
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

        # H-5: Cap compounded multipliers. Worst-case uncapped product is ~1.60
        # (BULL_MARKET × LOW_VOL × BULLISH × STRONG_POSITIVE). Any consumer that
        # applies these as a direct probability modifier would exceed 1.0. Hard
        # cap at 1.35 (~35% max boost) preserves intent while keeping outputs sane.
        for key in multipliers:
            multipliers[key] = min(multipliers[key], 1.35)

        return multipliers
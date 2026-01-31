#!/usr/bin/env python3
"""
MARK5 Trading Signal Generator v5.0
Generate actionable trading signals from predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from core.system.container import container
import warnings
warnings.filterwarnings('ignore')

class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1

class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"

class TradingSignalGenerator:
    """Generate trading signals with confidence and risk assessment"""
    
    def __init__(self, min_confidence: float = 60.0, risk_tolerance: str = 'MEDIUM'):
        """
        Initialize signal generator
        
        Args:
            min_confidence: Minimum confidence for actionable signals (%)
            risk_tolerance: Risk tolerance level (LOW/MEDIUM/HIGH)
        """
        self.min_confidence = min_confidence
        self.risk_tolerance = risk_tolerance.upper()
        
        # Risk-adjusted thresholds
        self.thresholds = self._get_risk_thresholds()
    
    def _get_risk_thresholds(self) -> Dict:
        """Get signal thresholds based on risk tolerance"""
        if self.risk_tolerance == 'LOW':
            return {
                'strong_buy': 0.05,      # 5% predicted gain
                'buy': 0.03,             # 3% predicted gain
                'sell': -0.03,           # 3% predicted loss
                'strong_sell': -0.05,    # 5% predicted loss
                'min_confidence': 70.0   # Higher confidence required
            }
        elif self.risk_tolerance == 'HIGH':
            return {
                'strong_buy': 0.02,
                'buy': 0.01,
                'sell': -0.01,
                'strong_sell': -0.02,
                'min_confidence': 50.0   # Lower confidence OK
            }
        else:  # MEDIUM
            return {
                'strong_buy': 0.04,
                'buy': 0.02,
                'sell': -0.02,
                'strong_sell': -0.04,
                'min_confidence': 60.0
            }
    
    def generate_signal(self, prediction: Dict, market_data: pd.DataFrame = None) -> Dict:
        """
        Generate trading signal from prediction
        
        Args:
            prediction: Prediction dict from MARK5PredictionAPI
            market_data: Optional market data for context
        
        Returns:
            Trading signal with details
        """
        if not prediction.get('success'):
            return {
                'signal': SignalType.HOLD.value,
                'strength': SignalStrength.VERY_WEAK,
                'actionable': False,
                'reason': 'Prediction failed',
                'confidence': 0.0
            }
        
        # Extract prediction details
        pred_details = prediction['prediction']
        return_pct = pred_details['return_pct'] / 100  # Convert to decimal
        confidence = pred_details['confidence']
        
        # Check minimum confidence
        if confidence < self.thresholds['min_confidence']:
            return {
                'signal': SignalType.HOLD.value,
                'strength': SignalStrength.VERY_WEAK,
                'actionable': False,
                'reason': f'Low confidence ({confidence:.1f}% < {self.thresholds["min_confidence"]:.1f}%)',
                'confidence': confidence,
                'predicted_return': return_pct * 100
            }
        
        # Determine signal
        if return_pct >= self.thresholds['strong_buy']:
            signal = SignalType.STRONG_BUY.value
            strength = SignalStrength.VERY_STRONG if confidence > 80 else SignalStrength.STRONG
        elif return_pct >= self.thresholds['buy']:
            signal = SignalType.BUY.value
            strength = SignalStrength.STRONG if confidence > 70 else SignalStrength.MODERATE
        elif return_pct <= self.thresholds['strong_sell']:
            signal = SignalType.STRONG_SELL.value
            strength = SignalStrength.VERY_STRONG if confidence > 80 else SignalStrength.STRONG
        elif return_pct <= self.thresholds['sell']:
            signal = SignalType.SELL.value
            strength = SignalStrength.STRONG if confidence > 70 else SignalStrength.MODERATE
        else:
            signal = SignalType.HOLD.value
            strength = SignalStrength.MODERATE
        
        # Add market context if available
        context = {}
        if market_data is not None and len(market_data) > 0:
            context = self._analyze_market_context(market_data)
        
        # Calculate position size recommendation
        position_size = self._calculate_position_size(
            return_pct, confidence, strength.value
        )
        
        # Determine if actionable
        actionable = (
            signal != SignalType.HOLD.value and
            confidence >= self.thresholds['min_confidence'] and
            strength.value >= SignalStrength.MODERATE.value
        )
        
        return {
            'signal': signal,
            'strength': strength.name,
            'strength_value': strength.value,
            'actionable': actionable,
            'predicted_return': round(return_pct * 100, 2),
            'confidence': round(confidence, 1),
            'position_size_pct': round(position_size * 100, 1),
            'stop_loss': self._calculate_stop_loss(return_pct),
            'take_profit': self._calculate_take_profit(return_pct),
            'risk_reward_ratio': self._calculate_risk_reward(return_pct),
            'market_context': context,
            'timestamp': container.time.iso_format(),
            'risk_tolerance': self.risk_tolerance
        }
    
    def _analyze_market_context(self, data: pd.DataFrame) -> Dict:
        """Analyze current market context"""
        try:
            recent_data = data.tail(20)
            
            # Trend
            sma_20 = recent_data['close'].mean()
            current_price = recent_data['close'].iloc[-1]
            trend = 'UPTREND' if current_price > sma_20 else 'DOWNTREND'
            
            # Volatility
            volatility = recent_data['close'].pct_change().std() * np.sqrt(252) * 100
            vol_regime = 'HIGH' if volatility > 30 else 'MEDIUM' if volatility > 15 else 'LOW'
            
            # Volume trend
            avg_volume = recent_data['volume'].mean() if 'volume' in recent_data else None
            current_volume = recent_data['volume'].iloc[-1] if 'volume' in recent_data else None
            volume_trend = 'HIGH' if current_volume and current_volume > avg_volume * 1.5 else 'NORMAL'
            
            return {
                'trend': trend,
                'volatility': round(volatility, 1),
                'volatility_regime': vol_regime,
                'volume_trend': volume_trend
            }
        except:
            return {}
    
    def _calculate_position_size(self, return_pct: float, confidence: float, 
                                 strength: int) -> float:
        """
        Calculate recommended position size
        
        Returns:
            Position size as fraction of portfolio (0-1)
        """
        # Base position size on confidence and strength
        base_size = (confidence / 100) * (strength / 5)
        
        # Adjust for risk tolerance
        if self.risk_tolerance == 'LOW':
            base_size *= 0.5
        elif self.risk_tolerance == 'HIGH':
            base_size *= 1.5
        
        # Adjust for expected return magnitude
        if abs(return_pct) > 0.05:  # >5% expected return
            base_size *= 1.2
        
        # Cap at reasonable limits
        if self.risk_tolerance == 'LOW':
            max_size = 0.15
        elif self.risk_tolerance == 'HIGH':
            max_size = 0.40
        else:
            max_size = 0.25
        
        return min(base_size, max_size)
    
    def _calculate_stop_loss(self, return_pct: float, volatility: float = None) -> float:
        """
        Calculate recommended stop loss percentage
        
        Args:
            return_pct: Predicted return percentage (decimal)
            volatility: Optional volatility metric (e.g. ATR%)
        """
        # If we have volatility (e.g., ATR%), use 2x ATR
        if volatility and volatility > 0:
            # volatility is usually annual sigma, we need daily/intraday estimate
            # Simplified proxy: assume input is annual vol, convert to daily
            # Or assume input IS the dynamic stop distance (ATR)
            # Let's assume the caller passes a relevant volatility metric (like ATR%)
            # If it's annual vol, we divide by 16 (sqrt(252)).
            # For safety, let's assume it's annual vol if > 1.0, else it's a ratio.
            
            # Actually, let's stick to the plan: "Simplified proxy"
            # stop_pct = -(volatility / 16) * 1.5 # Approx daily vol * 1.5
            
            # If volatility is passed as a percentage (e.g. 20.0 for 20%), convert to decimal
            vol_decimal = volatility / 100 if volatility > 1 else volatility
            
            # Daily Vol Proxy = Annual / 16
            daily_vol = vol_decimal / 16
            stop_loss = -(daily_vol * 1.5)
            
            return round(stop_loss * 100, 2)

        # Fallback to Risk-Reward based (Original Logic)
        if return_pct > 0:
            # For long positions
            stop_loss = -abs(return_pct) / 2
        else:
            # For short positions
            stop_loss = abs(return_pct) / 2
        
        # Minimum stop loss based on risk tolerance
        min_stop = {
            'LOW': -0.02,    # 2%
            'MEDIUM': -0.03,  # 3%
            'HIGH': -0.05     # 5%
        }
        
        if return_pct > 0:
            stop_loss = max(stop_loss, min_stop[self.risk_tolerance])
        
        return round(stop_loss * 100, 2)
    
    def _calculate_take_profit(self, return_pct: float) -> float:
        """Calculate recommended take profit percentage"""
        # Use prediction as take profit, but cap it
        take_profit = return_pct * 1.5  # 150% of prediction
        
        # Cap based on risk tolerance
        max_tp = {
            'LOW': 0.10,     # 10%
            'MEDIUM': 0.15,   # 15%
            'HIGH': 0.25      # 25%
        }
        
        if return_pct > 0:
            take_profit = min(abs(take_profit), max_tp[self.risk_tolerance])
        else:
            take_profit = -min(abs(take_profit), max_tp[self.risk_tolerance])
        
        return round(take_profit * 100, 2)
    
    def _calculate_risk_reward(self, return_pct: float) -> float:
        """Calculate risk-reward ratio"""
        stop_loss = abs(self._calculate_stop_loss(return_pct) / 100)
        take_profit = abs(self._calculate_take_profit(return_pct) / 100)
        
        if stop_loss > 0:
            return round(take_profit / stop_loss, 2)
        return 0.0
    
    def batch_generate_signals(self, predictions: List[Dict], 
                              market_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate signals for multiple predictions
        
        Args:
            predictions: List of prediction dicts
            market_data: Dict of ticker -> DataFrame
        
        Returns:
            DataFrame with all signals
        """
        signals = []
        
        for pred in predictions:
            ticker = pred.get('ticker')
            data = market_data.get(ticker) if market_data else None
            
            signal = self.generate_signal(pred, data)
            signal['ticker'] = ticker
            signals.append(signal)
        
        df = pd.DataFrame(signals)
        
        # Sort by strength and confidence
        if len(df) > 0:
            df = df.sort_values(['strength_value', 'confidence'], 
                               ascending=[False, False])
        
        return df


if __name__ == '__main__':
    print("Testing Trading Signal Generator...")
    
    # Create signal generator
    generator = TradingSignalGenerator(
        min_confidence=60.0,
        risk_tolerance='MEDIUM'
    )
    
    print(f"\n✓ Signal generator initialized")
    print(f"  Risk tolerance: MEDIUM")
    print(f"  Min confidence: 60.0%")
    
    # Test with sample predictions
    test_predictions = [
        {
            'success': True,
            'ticker': 'RELIANCE.NS',
            'prediction': {
                'return_pct': 4.5,
                'confidence': 75.0,
                'direction': 'UP',
                'signal': 'BUY'
            }
        },
        {
            'success': True,
            'ticker': 'TCS.NS',
            'prediction': {
                'return_pct': 1.5,
                'confidence': 55.0,
                'direction': 'UP',
                'signal': 'BUY'
            }
        },
        {
            'success': True,
            'ticker': 'INFY.NS',
            'prediction': {
                'return_pct': -3.5,
                'confidence': 70.0,
                'direction': 'DOWN',
                'signal': 'SELL'
            }
        }
    ]
    
    print("\nGenerating signals...")
    for pred in test_predictions:
        signal = generator.generate_signal(pred)
        ticker = pred['ticker']
        
        print(f"\n{ticker}:")
        print(f"  Signal: {signal['signal']} ({signal['strength']})")
        print(f"  Actionable: {'YES' if signal['actionable'] else 'NO'}")
        print(f"  Confidence: {signal['confidence']:.1f}%")
        print(f"  Position Size: {signal['position_size_pct']:.1f}%")
        print(f"  Stop Loss: {signal['stop_loss']:.2f}%")
        print(f"  Take Profit: {signal['take_profit']:.2f}%")
    
    print("\n✓ Trading signal generator test complete")

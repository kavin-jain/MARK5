#!/usr/bin/env python3
"""
MARK5 TRADING SIGNAL GENERATOR v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, version bump
- [Previous] v7.0: Production-grade refactor
  • Added regime-aware signal generation, ATR-based stops
  • Added minimum R:R enforcement, signal expiry, conviction decay

TRADING ROLE: Convert predictions to actionable trading signals
SAFETY LEVEL: HIGH - Signals drive order placement

SIGNAL GENERATION FLOW:
1. Validate prediction confidence
2. Apply regime-based filters
3. Calculate ATR-based stops
4. Enforce minimum R:R ratio
5. Generate position size recommendation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import warnings
warnings.filterwarnings('ignore')

# Try to import container, fallback to UTC
try:
    from core.system.container import container
except ImportError:
    container = None

logger = logging.getLogger("MARK5.Signals")


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


class MarketRegime(Enum):
    """Market volatility regime — must include ALL regimes checked in thresholds/gates"""
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NORMAL = "NORMAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CRISIS = "CRISIS"
    # Regime-adaptive thresholds + RULE 88 depend on these:
    STRONG_BULL = "STRONG_BULL"
    BULL_MARKET = "BULL_MARKET"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS_MARKET = "SIDEWAYS_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    VOLATILE_MARKET = "VOLATILE_MARKET"
    CHOPPY = "CHOPPY"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"


@dataclass
class TradingSignal:
    """Structured trading signal with all required fields"""
    symbol: str
    signal: str
    strength: str
    strength_value: int
    actionable: bool
    predicted_return: float
    confidence: float
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_reward_ratio: float
    atr_stop_distance: Optional[float] = None
    regime: str = "NORMAL"
    expiry: Optional[datetime] = None
    reasons: List[str] = field(default_factory=list)
    market_context: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal': self.signal,
            'strength': self.strength,
            'strength_value': self.strength_value,
            'actionable': self.actionable,
            'predicted_return': self.predicted_return,
            'confidence': self.confidence,
            'position_size_pct': round(self.position_size_pct, 1),
            'stop_loss': self.stop_loss_pct,
            'take_profit': self.take_profit_pct,
            'risk_reward_ratio': self.risk_reward_ratio,
            'atr_stop_distance': self.atr_stop_distance,
            'regime': self.regime,
            'expiry': self.expiry.isoformat() if self.expiry else None,
            'reasons': self.reasons,
            'market_context': self.market_context
        }


@dataclass
class TimestampedData:
    """Wrapper for data with timestamp for staleness detection"""
    data: Any
    calculated_at: datetime
    source_timestamp: datetime
    component: str
    
    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        """Check if data is too old to use"""
        age = (datetime.now() - self.calculated_at).total_seconds()
        return age > max_age_seconds
    
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return (datetime.now() - self.calculated_at).total_seconds()


class TradingSignalGenerator:
    """
    Production-grade trading signal generator.
    
    TRADER INTELLIGENCE:
    - Filters signals in adverse regimes
    - Uses ATR for dynamic stop placement
    - Enforces minimum R:R ratio
    - Manages signal staleness
    - Resolves conflicting signals
    """
    
    # Regime-based confidence adjustments
    REGIME_CONFIDENCE_MULTIPLIERS = {
        MarketRegime.LOW_VOLATILITY: 1.0,     # Full confidence
        MarketRegime.NORMAL: 0.95,            # Slight discount
        MarketRegime.HIGH_VOLATILITY: 0.80,   # Reduce confidence
        MarketRegime.CRISIS: 0.60,            # Significant reduction
    }
    
    # Minimum required R:R by risk tolerance
    MIN_RR_RATIO = {
        'LOW': 2.5,
        'MEDIUM': 2.0,
        'HIGH': 1.5
    }
    
    # STATE CONSISTENCY: Maximum data staleness (seconds)
    MAX_DATA_STALENESS_SECONDS = 5.0
    
    def __init__(
        self,
        min_confidence: float = 60.0,
        risk_tolerance: str = 'MEDIUM',
        atr_stop_multiplier: float = 2.0,
        signal_expiry_minutes: int = 30,
        enforce_min_rr: bool = True
    ):
        """
        Initialize signal generator.
        
        Args:
            min_confidence: Minimum confidence for actionable signals (%)
            risk_tolerance: Risk tolerance level (LOW/MEDIUM/HIGH)
            atr_stop_multiplier: ATR multiplier for stop loss
            signal_expiry_minutes: Signal expiry time
            enforce_min_rr: Enforce minimum R:R ratio
        """
        self.min_confidence = min_confidence
        self.risk_tolerance = risk_tolerance.upper()
        self.atr_stop_multiplier = atr_stop_multiplier
        self.signal_expiry_minutes = signal_expiry_minutes
        self.enforce_min_rr = enforce_min_rr
        
        # Current market regime
        self.current_regime = MarketRegime.NORMAL
        
        # Risk-adjusted thresholds
        self.thresholds = self._get_risk_thresholds()
        
        # Minimum R:R
        self.min_rr = self.MIN_RR_RATIO.get(self.risk_tolerance, 2.0)
        
        # Statistics
        self._stats = {
            'signals_generated': 0,
            'actionable_signals': 0,
            'regime_filtered': 0,
            'rr_filtered': 0
        }
        
        logger.info(
            f"SignalGenerator v7.0 | MinConf: {min_confidence}% | "
            f"Risk: {risk_tolerance} | ATR×{atr_stop_multiplier} | "
            f"Min R:R: {self.min_rr}"
        )
    
    def _get_risk_thresholds(self) -> Dict:
        """Get signal thresholds based on risk tolerance"""
        if self.risk_tolerance == 'LOW':
            return {
                'strong_buy': 0.05,
                'buy': 0.03,
                'sell': -0.03,
                'strong_sell': -0.05,
                'min_confidence': 70.0
            }
        elif self.risk_tolerance == 'HIGH':
            return {
                'strong_buy': 0.02,
                'buy': 0.01,
                'sell': -0.01,
                'strong_sell': -0.02,
                'min_confidence': 50.0
            }
        else:  # MEDIUM
            return {
                'strong_buy': 0.04,
                'buy': 0.02,
                'sell': -0.02,
                'strong_sell': -0.04,
                'min_confidence': 60.0
            }

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def generate_signal(
        self,
        prediction: Dict,
        market_data: pd.DataFrame = None,
        atr: float = None,
        current_price: float = None
    ) -> Dict:
        """
        Generate trading signal from prediction.
        
        Args:
            prediction: Prediction dict from model
            market_data: Optional market data for context
            atr: ATR value for stop calculation
            current_price: Current price for ATR-based stops
        
        Returns:
            Trading signal dictionary
        """
        self._stats['signals_generated'] += 1
        symbol = prediction.get('ticker', prediction.get('symbol', 'UNKNOWN'))
        reasons = []
        
        # Handle failed predictions
        if not prediction.get('success'):
            return self._create_hold_signal(
                symbol, 0.0, 0.0, ['Prediction failed']
            ).to_dict()
        
        # STATE CONSISTENCY CHECK: Reject stale data
        pred_timestamp = prediction.get('timestamp') or prediction.get('calculated_at')
        if pred_timestamp:
            try:
                if isinstance(pred_timestamp, str):
                    pred_time = datetime.fromisoformat(pred_timestamp)
                else:
                    pred_time = pred_timestamp
                age_seconds = (datetime.now() - pred_time).total_seconds()
                if age_seconds > self.MAX_DATA_STALENESS_SECONDS:
                    logger.warning(
                        f"STALE DATA REJECTED: {symbol} prediction is {age_seconds:.1f}s old "
                        f"(max: {self.MAX_DATA_STALENESS_SECONDS}s)"
                    )
                    return self._create_hold_signal(
                        symbol, 0.0, 0.0, 
                        [f'Stale data: {age_seconds:.1f}s old (limit: {self.MAX_DATA_STALENESS_SECONDS}s)']
                    ).to_dict()
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse prediction timestamp for {symbol}: {e}")
        
        # Extract prediction details
        pred_details = prediction.get('prediction', prediction)
        return_pct = pred_details.get('return_pct', 0) / 100  # Convert to decimal
        confidence = pred_details.get('confidence', 0)
        
        # Apply regime confidence adjustment
        regime_mult = self.REGIME_CONFIDENCE_MULTIPLIERS.get(
            self.current_regime, 1.0
        )
        adj_confidence = confidence * regime_mult
        
        if regime_mult < 1.0:
            reasons.append(f"Regime adjustment: {self.current_regime.value}")
            self._stats['regime_filtered'] += 1
            
        # --- RULE 88: STRONG BULL GATE (Week 3 Rollout - Fully Implemented) ---
        if self.current_regime.name == 'STRONG_BULL':
            # Safely resolve journal without silent failures
            try:
                if container and hasattr(container, 'trade_journal') and container.trade_journal is not None:
                    journal = container.trade_journal
                elif hasattr(self, 'trade_journal') and self.trade_journal is not None:
                    journal = self.trade_journal
                else:
                    journal = None
                    
                assert journal is not None, "TradeJournal not registered in container or signal generator instance"
                sharpe = journal.get_rolling_sharpe(symbol)
            except Exception as e:
                logger.error(f"CRITICAL: Cannot resolve TradeJournal for RULE 88 — {e}")
                sharpe = 1.0  # Default to ALLOW (don't block all trades due to wiring issue)
                
            if sharpe < 1.0:
                logger.warning(f"🚨 RULE 88 BLOCKED: {symbol} in STRONG_BULL but Sharpe={sharpe:.2f} < 1.0")
                return self._create_hold_signal(
                    symbol, 0.0, 0.0,
                    [f'RULE 88: STRONG BULL but Sharpe {sharpe:.2f} < 1.0']
                ).to_dict()
        # -------------------------------------------------------------------
        
        # --- NSE EVENT CALENDAR FILTER ---
        # Block new entries on days where the market behaves out-of-distribution:
        # RBI MPC meetings, Budget Day, and near monthly F&O expiry
        try:
            today = datetime.now().date()
            
            # RBI MPC meeting dates 2026 (announced annually by RBI)
            RBI_MPC_DATES = {
                (2, 5), (2, 6), (2, 7),    # Feb 5-7
                (4, 8), (4, 9), (4, 10),   # Apr 8-10
                (6, 4), (6, 5), (6, 6),    # Jun 4-6
                (8, 5), (8, 6), (8, 7),    # Aug 5-7
                (10, 7), (10, 8), (10, 9), # Oct 7-9
                (12, 3), (12, 4), (12, 5), # Dec 3-5
            }
            
            # Budget Day
            BUDGET_DATES = {(2, 1), (7, 23)}  # Union Budget + possible interim
            
            today_tuple = (today.month, today.day)
            
            if today_tuple in RBI_MPC_DATES:
                logger.warning(f"🚫 EVENT FILTER: RBI MPC meeting day — no new entries")
                return self._create_hold_signal(
                    symbol, 0.0, 0.0,
                    ['EVENT FILTER: RBI MPC meeting day — out-of-distribution risk']
                ).to_dict()
            
            if today_tuple in BUDGET_DATES:
                logger.warning(f"🚫 EVENT FILTER: Budget Day — no new entries")
                return self._create_hold_signal(
                    symbol, 0.0, 0.0,
                    ['EVENT FILTER: Budget Day — extreme volatility expected']
                ).to_dict()
            
            # Monthly F&O expiry: last Thursday of the month
            # Block entries on expiry day and 1 day before
            import calendar
            cal = calendar.Calendar()
            month_days = list(cal.itermonthdays2(today.year, today.month))
            # Find last Thursday (weekday=3) in the month
            last_thursday = None
            for day, weekday in reversed(month_days):
                if day != 0 and weekday == 3:
                    last_thursday = day
                    break
            
            if last_thursday:
                from datetime import date
                expiry_date = date(today.year, today.month, last_thursday)
                days_to_expiry = (expiry_date - today).days
                
                if 0 <= days_to_expiry <= 1:
                    logger.warning(
                        f"🚫 EVENT FILTER: Monthly F&O expiry in {days_to_expiry}d — "
                        f"pinning/gamma effects, no new entries"
                    )
                    return self._create_hold_signal(
                        symbol, 0.0, 0.0,
                        [f'EVENT FILTER: Monthly F&O expiry in {days_to_expiry}d']
                    ).to_dict()
                    
        except Exception as e:
            logger.debug(f"Event calendar check failed (non-fatal): {e}")
        # --- END EVENT CALENDAR FILTER ---
        
        # --- REGIME-ADAPTIVE CONFIDENCE (Week 3 Rollout) ---
        # Instead of static self.min_confidence
        CONFIDENCE_THRESHOLDS = {
            "STRONG_BULL":    85.0,  # Whitelist only AND high conviction
            "BULL_MARKET":    78.0,  # Harder to beat B&H, be selective
            "TRENDING_UP":    75.0,  # Original RULE 20 threshold
            "TRENDING_DOWN":  70.0,  # Model's sweet spot
            "SIDEWAYS_MARKET":70.0,  # Model's sweet spot  
            "BEAR_MARKET":    68.0,  # Model excels here
            "VOLATILE_MARKET":82.0,  # High noise
            "CHOPPY":         88.0,  # Almost never trade
            "UNKNOWN":        80.0,  # Conservative default
        }
        
        regime_name = self.current_regime.name if hasattr(self.current_regime, 'name') else str(self.current_regime)
        effective_min_conf = CONFIDENCE_THRESHOLDS.get(regime_name, 75.0)

        # Fallback to absolute base if somehow threshold is lower than absolute config min
        effective_min_conf = max(
            effective_min_conf,
            self.thresholds.get('min_confidence', 0.0)
        )
        # ---------------------------------------------------
        
        if adj_confidence < effective_min_conf:
            return self._create_hold_signal(
                symbol, confidence, return_pct * 100,
                [f'Low confidence ({adj_confidence:.1f}% < {effective_min_conf:.1f}% for {regime_name})']
            ).to_dict()
        
        # Analyze market context
        context = {}
        computed_atr = atr
        if market_data is not None and len(market_data) > 0:
            context = self._analyze_market_context(market_data)
            if computed_atr is None:
                computed_atr = context.get('atr')
            if current_price is None:
                current_price = context.get('current_price')
        
        # Determine signal type
        signal_type, strength = self._determine_signal(return_pct, adj_confidence)
        
        # Calculate stops
        if computed_atr is not None and current_price is not None and computed_atr > 0:
            # Use ATR-based stop
            stop_loss, atr_distance = self._calculate_atr_stop(
                return_pct, computed_atr, current_price
            )
            reasons.append(f"ATR-based stop: {self.atr_stop_multiplier}×ATR")
        else:
            # Fallback to prediction-based stop
            stop_loss = self._calculate_fallback_stop(return_pct)
            atr_distance = None
            reasons.append("Fallback stop (no ATR)")
        
        # Calculate take profit
        take_profit = self._calculate_take_profit(return_pct)
        
        # Calculate R:R ratio
        rr_ratio = self._calculate_risk_reward(stop_loss, take_profit)
        
        # Enforce minimum R:R
        actionable = signal_type != SignalType.HOLD
        if self.enforce_min_rr and rr_ratio < self.min_rr and actionable:
            self._stats['rr_filtered'] += 1
            reasons.append(f"R:R {rr_ratio:.1f} < min {self.min_rr}")
            actionable = False
        
        # Calculate position size
        position_size = self._calculate_position_size(
            return_pct, adj_confidence, strength.value
        )
        
        # Final actionable check
        if actionable:
            actionable = (
                signal_type != SignalType.HOLD and
                strength.value >= SignalStrength.MODERATE.value
            )
        
        if actionable:
            self._stats['actionable_signals'] += 1
        
        # Build signal
        signal = TradingSignal(
            symbol=symbol,
            signal=signal_type.value,
            strength=strength.name,
            strength_value=strength.value,
            actionable=actionable,
            predicted_return=round(return_pct * 100, 2),
            confidence=round(adj_confidence, 1),
            position_size_pct=position_size * 100,
            stop_loss_pct=round(stop_loss, 2),
            take_profit_pct=round(take_profit, 2),
            risk_reward_ratio=round(rr_ratio, 2),
            atr_stop_distance=round(atr_distance, 2) if atr_distance else None,
            regime=self.current_regime.value,
            expiry=datetime.now() + timedelta(minutes=self.signal_expiry_minutes),
            reasons=reasons,
            market_context=context
        )
        
        return signal.to_dict()

    def _determine_signal(
        self,
        return_pct: float,
        confidence: float
    ) -> Tuple[SignalType, SignalStrength]:
        """Determine signal type and strength from prediction"""
        if return_pct >= self.thresholds['strong_buy']:
            signal = SignalType.STRONG_BUY
            strength = SignalStrength.VERY_STRONG if confidence > 80 else SignalStrength.STRONG
        elif return_pct >= self.thresholds['buy']:
            signal = SignalType.BUY
            strength = SignalStrength.STRONG if confidence > 70 else SignalStrength.MODERATE
        elif return_pct <= self.thresholds['strong_sell']:
            signal = SignalType.STRONG_SELL
            strength = SignalStrength.VERY_STRONG if confidence > 80 else SignalStrength.STRONG
        elif return_pct <= self.thresholds['sell']:
            signal = SignalType.SELL
            strength = SignalStrength.STRONG if confidence > 70 else SignalStrength.MODERATE
        else:
            signal = SignalType.HOLD
            strength = SignalStrength.MODERATE
        
        return signal, strength

    def _create_hold_signal(
        self,
        symbol: str,
        confidence: float,
        predicted_return: float,
        reasons: List[str]
    ) -> TradingSignal:
        """Create a HOLD signal with given reasons"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD.value,
            strength=SignalStrength.VERY_WEAK.name,
            strength_value=SignalStrength.VERY_WEAK.value,
            actionable=False,
            predicted_return=predicted_return,
            confidence=confidence,
            position_size_pct=0.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            risk_reward_ratio=0.0,
            reasons=reasons
        )

    # =========================================================================
    # STOP LOSS CALCULATION
    # =========================================================================

    def _calculate_atr_stop(
        self,
        return_pct: float,
        atr: float,
        current_price: float
    ) -> Tuple[float, float]:
        """
        Calculate ATR-based stop loss.
        
        Args:
            return_pct: Predicted return (decimal)
            atr: Average True Range (price units)
            current_price: Current price
            
        Returns:
            Tuple of (stop_loss_pct, atr_distance)
        """
        # ATR-based stop distance
        atr_distance = atr * self.atr_stop_multiplier
        stop_pct = (atr_distance / current_price) * 100
        
        # Apply direction
        if return_pct > 0:
            # Long position: stop is below entry
            stop_loss = -abs(stop_pct)
        else:
            # Short position: stop is above entry
            stop_loss = abs(stop_pct)
        
        # Apply minimum stops based on risk tolerance
        min_stops = {'LOW': 1.5, 'MEDIUM': 2.0, 'HIGH': 3.0}
        min_stop = min_stops.get(self.risk_tolerance, 2.0)
        
        if return_pct > 0:
            stop_loss = min(stop_loss, -min_stop)  # Ensure at least min %
        else:
            stop_loss = max(stop_loss, min_stop)
        
        return stop_loss, atr_distance

    def _calculate_fallback_stop(self, return_pct: float) -> float:
        """Calculate fallback stop loss when ATR not available"""
        # Base stop on risk tolerance
        base_stops = {'LOW': 2.0, 'MEDIUM': 3.0, 'HIGH': 5.0}
        base_stop = base_stops.get(self.risk_tolerance, 3.0)
        
        # Adjust based on predicted return magnitude
        if abs(return_pct) > 0.05:
            base_stop *= 1.5
        elif abs(return_pct) > 0.03:
            base_stop *= 1.2
        
        if return_pct > 0:
            return -base_stop
        else:
            return base_stop

    def _calculate_take_profit(self, return_pct: float) -> float:
        """Calculate take profit level"""
        # Use prediction as base, with multiplier
        take_profit = return_pct * 100 * 1.2  # 120% of prediction
        
        # Cap based on risk tolerance
        max_tp = {'LOW': 8.0, 'MEDIUM': 12.0, 'HIGH': 20.0}
        max_val = max_tp.get(self.risk_tolerance, 12.0)
        
        # Minimum PT floor to ensure sufficient edge over costs
        MIN_PT = 1.5
        
        if return_pct > 0:
            take_profit = max(MIN_PT, min(take_profit, max_val))
        else:
            take_profit = min(-MIN_PT, max(take_profit, -max_val))
        
        return take_profit

    def _calculate_risk_reward(
        self,
        stop_loss_pct: float,
        take_profit_pct: float
    ) -> float:
        """Calculate risk-reward ratio"""
        risk = abs(stop_loss_pct)
        reward = abs(take_profit_pct)
        
        if risk <= 0:
            return 0.0
        
        return reward / risk

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def _calculate_position_size(
        self,
        return_pct: float,
        confidence: float,
        strength: int
    ) -> float:
        """Calculate recommended position size as fraction"""
        # Base size on confidence and strength
        base_size = (confidence / 100) * (strength / 5)
        
        # Adjust for risk tolerance
        risk_multipliers = {'LOW': 0.5, 'MEDIUM': 1.0, 'HIGH': 1.5}
        base_size *= risk_multipliers.get(self.risk_tolerance, 1.0)
        
        # Adjust for regime
        regime_mult = {
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.7,
            MarketRegime.CRISIS: 0.4
        }
        base_size *= regime_mult.get(self.current_regime, 1.0)
        
        # Cap at RULE 11 compliant limits (max 5% of portfolio per position)
        # RULE 18: 30% cash reserve means max 70% deployed across all positions
        max_sizes = {'LOW': 0.03, 'MEDIUM': 0.05, 'HIGH': 0.05}
        max_size = max_sizes.get(self.risk_tolerance, 0.05)
        
        return min(base_size, max_size)

    # =========================================================================
    # MARKET CONTEXT ANALYSIS
    # =========================================================================

    def _analyze_market_context(self, data: pd.DataFrame) -> Dict:
        """Analyze current market context with ATR calculation"""
        try:
            recent_data = data.tail(20).copy()
            
            if len(recent_data) < 5:
                return {}
            
            # Current price
            current_price = float(recent_data['close'].iloc[-1])
            
            # Trend
            sma_20 = recent_data['close'].mean()
            trend = 'UPTREND' if current_price > sma_20 else 'DOWNTREND'
            
            # Calculate ATR
            atr = self._calculate_atr(recent_data)
            
            # Volatility (annualized)
            returns = recent_data['close'].pct_change().dropna()
            if len(returns) > 1:
                volatility = float(returns.std() * np.sqrt(252) * 100)
                vol_regime = 'HIGH' if volatility > 30 else 'MEDIUM' if volatility > 15 else 'LOW'
            else:
                volatility = 0.0
                vol_regime = 'UNKNOWN'
            
            # Volume trend
            if 'volume' in recent_data.columns:
                avg_volume = recent_data['volume'].mean()
                current_volume = recent_data['volume'].iloc[-1]
                volume_trend = 'HIGH' if current_volume > avg_volume * 1.5 else 'NORMAL'
            else:
                volume_trend = 'UNKNOWN'
            
            return {
                'current_price': current_price,
                'trend': trend,
                'atr': atr,
                'volatility': round(volatility, 1),
                'volatility_regime': vol_regime,
                'volume_trend': volume_trend
            }
        except Exception as e:
            logger.warning(f"Market context analysis failed: {e}")
            return {}

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(data) < 2:
            return 0.0
        
        try:
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # True Range components
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # ATR is EMA of TR
            if len(tr) >= period:
                atr = float(np.mean(tr[-period:]))
            else:
                atr = float(np.mean(tr))
            
            return atr
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return 0.0

    # =========================================================================
    # REGIME MANAGEMENT
    # =========================================================================

    def set_regime(self, regime: MarketRegime) -> None:
        """Set current market regime"""
        if regime != self.current_regime:
            old = self.current_regime
            self.current_regime = regime
            logger.info(f"Regime changed: {old.value} → {regime.value}")

    def detect_regime_from_volatility(self, volatility: float) -> MarketRegime:
        """Auto-detect regime from volatility"""
        if volatility > 40:
            return MarketRegime.CRISIS
        elif volatility > 25:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 10:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.NORMAL

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    def batch_generate_signals(
        self,
        predictions: List[Dict],
        market_data: Dict[str, pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate signals for multiple predictions.
        
        Args:
            predictions: List of prediction dicts
            market_data: Dict of ticker -> DataFrame
        
        Returns:
            DataFrame with all signals
        """
        signals = []
        
        for pred in predictions:
            ticker = pred.get('ticker', pred.get('symbol'))
            data = market_data.get(ticker) if market_data else None
            
            signal = self.generate_signal(pred, data)
            signal['ticker'] = ticker
            signals.append(signal)
        
        df = pd.DataFrame(signals)
        
        if len(df) > 0:
            df = df.sort_values(
                ['strength_value', 'confidence'],
                ascending=[False, False]
            )
        
        return df

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get signal generator statistics"""
        return {
            **self._stats,
            'current_regime': self.current_regime.value,
            'min_confidence': self.min_confidence,
            'risk_tolerance': self.risk_tolerance,
            'min_rr': self.min_rr
        }


if __name__ == '__main__':
    print("Testing Trading Signal Generator v7.0...")
    
    # Create signal generator
    generator = TradingSignalGenerator(
        min_confidence=60.0,
        risk_tolerance='MEDIUM',
        atr_stop_multiplier=2.0
    )
    
    print(f"\n✓ Signal generator initialized")
    print(f"  Risk tolerance: MEDIUM")
    print(f"  Min confidence: 60.0%")
    print(f"  ATR multiplier: 2.0x")
    
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
        print(f"  R:R Ratio: {signal['risk_reward_ratio']:.2f}")
    
    print("\n✓ Trading signal generator v7.0 test complete")

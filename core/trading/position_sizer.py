"""
MARK5 VOLATILITY AWARE POSITION SIZER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, version bump
- [Previous] v7.0: Production-grade refactor
  • Added regime-aware sizing, performance feedback
  • Added max concurrent positions, sector concentration limits
  • Added drawdown-based scaling, Kelly-optimal option

TRADING ROLE: Optimal position sizing for risk normalization
SAFETY LEVEL: CRITICAL - Controls capital allocation per trade

SIZING METHODOLOGY:
1. Base: ATR-based volatility sizing (constant dollar risk)
2. Adjusted: Conviction scaling (higher confidence = more size)
3. Constrained: Capital limits, concentration limits
4. Dynamic: Regime and performance adjustments
"""

import logging
import numpy as np
from decimal import Decimal
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple
from core.utils.constants import MarketRegime

logger = logging.getLogger("MARK5.PositionSizer")

# --------------------------------------------------------------------------- #
# Almgren-Chriss slippage model constants (rebuild report Section 5.1)
# Calibrated for NSE midcap stocks. η and γ are empirically derived.
# --------------------------------------------------------------------------- #

# Participation rate hard limit (rebuild report Section 7.5, quant_standards)
MAX_PARTICIPATION_RATE: float = 0.01   # Never consume >1% of 20d ADV per order

# Almgren-Chriss temporary impact coefficient (NSE midcap calibration)
AC_ETA: float = 0.0025   # η — sqrt-law temporary impact; η × sqrt(participation)

# Almgren-Chriss permanent impact coefficient (NSE midcap calibration)
AC_GAMMA: float = 0.001  # γ — linear permanent impact; γ × participation

# ADV liquidity tiers (₹ value of daily turnover)
ADV_ILLIQUID_THRESHOLD: float = 5e7    # < ₹5cr/day → illiquid
ADV_SEMILIQUID_THRESHOLD: float = 5e8  # ₹5cr–₹50cr/day → semi-liquid

# Half-spread cost per leg by liquidity tier (bps)
SPREAD_BPS_ILLIQUID: float = 0.0015    # 15bps — < ₹5cr ADV
SPREAD_BPS_SEMILIQUID: float = 0.0008  # 8bps  — ₹5cr–₹50cr ADV
SPREAD_BPS_LIQUID: float = 0.0003      # 3bps  — > ₹50cr ADV (Nifty 50 level)


# Standardized MarketRegime imported from constants


@dataclass
class SizingResult:
    """Detailed result from position sizing calculation - uses Decimal for monetary fields"""
    quantity: int
    reason: str
    price: Decimal  # HIGH-001 FIX: Changed from float
    atr: float  # ATR can remain float (statistical measure)
    conviction: float  # Percentage - can remain float
    risk_amount: Decimal  # HIGH-001 FIX: Changed from float
    stop_distance: float  # Percentage - can remain float
    volatility_qty: int
    max_capital_qty: int
    regime_multiplier: float
    performance_multiplier: float
    final_multiplier: float
    
    def to_dict(self) -> Dict:
        return {
            'final_qty': self.quantity,
            'reason': self.reason,
            'price': float(self.price),  # Convert for JSON serialization
            'atr': round(self.atr, 2),
            'conviction': round(self.conviction, 2),
            'risk_amount': float(self.risk_amount),  # Convert for JSON serialization
            'stop_distance': round(self.stop_distance, 2),
            'volatility_qty': self.volatility_qty,
            'max_cap_qty': self.max_capital_qty,
            'regime_mult': round(self.regime_multiplier, 2),
            'perf_mult': round(self.performance_multiplier, 2),
            'total_mult': round(self.final_multiplier, 2)
        }


class VolatilityAwarePositionSizer:
    """
    Production-grade position sizer with adaptive behavior.
    
    TRADER INTELLIGENCE:
    - Reduces size in high volatility regimes
    - Reduces size after consecutive losses
    - Increases size gradually after wins
    - Respects concentration limits
    - Tracks drawdown for extra safety
    
    SIZING FORMULA:
    Base Qty = (Capital × Risk% × Conviction) / (ATR × Multiplier)
    Final Qty = Base Qty × Regime_Mult × Performance_Mult × Drawdown_Mult
    
    Constrained by: max_position_size_pct, max_capital_per_trade
    """
    
    # Regime-based multipliers (rebuild report Section 7.2) -- Rule 23
    DEFAULT_REGIME_MULTIPLIERS = {
        MarketRegime.TRENDING: 1.0,  # Full position sizing
        MarketRegime.RANGING:  0.7,  # Size at 70%
        MarketRegime.VOLATILE: 0.5,  # Size at 50%
        MarketRegime.BEAR:     0.3,  # Long entries suspended / Size at 30%
    }
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        default_risk_per_trade: float = 0.01,    # 1% risk per trade
        max_position_size_pct: float = 0.05,     # RULE 11: Max 5% of equity per trade
        atr_stop_multiplier: float = 2.0,        # Stop loss at 2 * ATR
        min_conviction: float = 0.3,              # Min conviction to trade (Aligned with V2 Predictor)
        max_concurrent_positions: int = 10,       # Max open positions
        max_sector_concentration: float = 0.15,  # RULE 12: Max 15% in one sector
        enable_performance_feedback: bool = True,
        enable_regime_adjustment: bool = True,
        enable_drawdown_scaling: bool = True
    ):
        # Core parameters
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.risk_per_trade = default_risk_per_trade
        self.max_position_size_pct = max_position_size_pct
        self.atr_multiplier = atr_stop_multiplier
        self.min_conviction = min_conviction
        self.max_concurrent_positions = max_concurrent_positions
        self.max_sector_concentration = max_sector_concentration
        
        # Feature flags
        self.enable_performance_feedback = enable_performance_feedback
        self.enable_regime_adjustment = enable_regime_adjustment
        self.enable_drawdown_scaling = enable_drawdown_scaling
        
        # Regime settings
        self.regime_multipliers = dict(self.DEFAULT_REGIME_MULTIPLIERS)
        self.current_regime = MarketRegime.RANGING # Default to RANGING
        
        # Performance tracking
        self.trade_history: deque = deque(maxlen=50)  # Last 50 trades
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Position tracking
        self.open_positions: Dict[str, float] = {}  # symbol -> notional value
        self.sector_exposure: Dict[str, float] = {}  # sector -> total exposure
        
        # Drawdown tracking
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        
        # Statistics
        self._stats = {
            'sizes_calculated': 0,
            'zero_sizes': 0,
            'regime_adjustments': 0,
            'performance_adjustments': 0
        }
        
        logger.info(
            f"PositionSizer v7.0 | Capital: ₹{initial_capital:,.0f} | "
            f"Risk: {default_risk_per_trade:.1%} | Max Pos: {max_position_size_pct:.0%} | "
            f"ATR Mult: {atr_stop_multiplier}x"
        )
        
    # =========================================================================
    # MAIN SIZING METHOD
    # =========================================================================

    def calculate_size(
        self,
        symbol: str,
        price: float,
        atr: float,
        conviction: float,
        sector: str = None,
        adv_20d: float = 0.0,
    ) -> Tuple[int, Dict]:
        """
        Calculate optimal position size (number of shares).

        Args:
            symbol:    Ticker symbol
            price:     Current entry price (₹)
            atr:       Average True Range for the stock (₹)
            conviction: Model confidence score (0.0–1.0)
            sector:    Optional sector for concentration check
            adv_20d:   20-day average daily value traded (₹).
                       When provided, enforces MAX_PARTICIPATION_RATE (1% of ADV)
                       hard constraint. Pass 0.0 to skip (e.g., Nifty 50 stocks
                       where liquidity is not a concern).

        Returns:
            Tuple[int, Dict]: (Quantity, details dict with all constraints logged)
        """
        self._stats['sizes_calculated'] += 1

        # Input validation
        if price <= 0 or atr <= 0:
            logger.warning(f"{symbol}: Invalid price ({price}) or ATR ({atr})")
            return 0, {'reason': 'Invalid price/ATR', 'final_qty': 0}

        if conviction < self.min_conviction:
            logger.info(f"{symbol}: Conviction {conviction:.2f} below {self.min_conviction}")
            return 0, {'reason': 'Low conviction', 'final_qty': 0}

        # Check concurrent positions limit
        if len(self.open_positions) >= self.max_concurrent_positions:
            if symbol not in self.open_positions:
                logger.warning(f"{symbol}: Max concurrent positions reached")
                return 0, {'reason': 'Max positions reached', 'final_qty': 0}

        # Check sector concentration
        if sector and sector in self.sector_exposure:
            sector_pct = self.sector_exposure[sector] / self.capital
            if sector_pct >= self.max_sector_concentration:
                logger.warning(
                    f"{symbol}: Sector {sector} at {sector_pct:.0%} "
                    f"(limit {self.max_sector_concentration:.0%})"
                )
                return 0, {'reason': 'Sector concentration limit', 'final_qty': 0}

        # ── RULE 23: TREND/RANGE/BEAR MULTIPLY & GATE ──
        regime_mult = self._get_regime_multiplier()
        
        # Rule 23: BEAR Market Entry Gate
        # "Long entries suspended. Cash only unless signal confidence > 70%."
        if self.current_regime == MarketRegime.BEAR:
            if conviction < 0.70:
                logger.warning(
                    f"{symbol}: LONG entry suspended (Rule 23 BEAR gate: "
                    f"conviction {conviction:.2f} < 0.70)"
                )
                return 0, {'reason': 'Rule 23 BEAR Entry Gate', 'final_qty': 0}
            else:
                # High-conviction override allowed under Rule 23
                # We use a 0.7 multiplier (scaling) instead of the 0.3 base.
                regime_mult = 0.7 
                logger.info(f"{symbol}: BEAR Market high-conviction entry allowed (multiplier: 0.7)")

        perf_mult = self._get_performance_multiplier()
        dd_mult = self._get_drawdown_multiplier()
        final_multiplier = regime_mult * perf_mult * dd_mult

        # ATR-based risk sizing
        adj_risk_pct = self.risk_per_trade * conviction * final_multiplier
        risk_amount = self.capital * adj_risk_pct
        stop_distance = atr * self.atr_multiplier
        if stop_distance <= 0:
            self._stats['zero_sizes'] += 1
            return 0, {'reason': 'Zero volatility', 'final_qty': 0}

        volatility_qty = int(risk_amount / stop_distance)

        # Capital concentration constraint (RULE 11: 5% max per trade)
        max_capital_alloc = self.capital * self.max_position_size_pct
        max_qty_capital = int(max_capital_alloc / price)

        # Participation rate hard constraint (rebuild report Section 7.5)
        # MAX_PARTICIPATION_RATE = 1% of 20d ADV. Binding for midcap positions.
        participation_constrained = False
        max_qty_participation = int(1e9)  # effectively infinite when adv_20d not provided
        if adv_20d > 0:
            max_trade_value = adv_20d * MAX_PARTICIPATION_RATE
            max_qty_participation = int(max_trade_value / price)
            if max_qty_participation < volatility_qty:
                participation_constrained = True

        # Take the most restrictive constraint
        final_qty = min(volatility_qty, max_qty_capital, max_qty_participation)

        if final_qty < 1:
            self._stats['zero_sizes'] += 1
            return 0, {'reason': 'Calculated size too small', 'final_qty': 0}

        # Slippage estimate (Almgren-Chriss) for logging
        estimated_slippage = 0.0
        if adv_20d > 0:
            trade_value = price * final_qty
            estimated_slippage = self.almgren_chriss_slippage(trade_value, adv_20d)

        result = SizingResult(
            quantity=final_qty,
            reason='Success',
            price=Decimal(str(price)),
            atr=atr,
            conviction=conviction,
            risk_amount=Decimal(str(risk_amount)),
            stop_distance=stop_distance,
            volatility_qty=volatility_qty,
            max_capital_qty=max_qty_capital,
            regime_multiplier=regime_mult,
            performance_multiplier=perf_mult,
            final_multiplier=final_multiplier,
        )

        details = result.to_dict()
        details['participation_constrained'] = participation_constrained
        details['max_qty_participation'] = max_qty_participation if adv_20d > 0 else None
        details['estimated_slippage_bps'] = round(estimated_slippage * 10000, 1)

        logger.debug(
            f"{symbol}: qty={final_qty} | risk=₹{risk_amount:.0f} | "
            f"vol_qty={volatility_qty} | cap_qty={max_qty_capital} | "
            f"part_qty={max_qty_participation if adv_20d > 0 else 'n/a'} | "
            f"slip={estimated_slippage*10000:.1f}bps | mult={final_multiplier:.2f}"
        )
        return final_qty, details

    # =========================================================================
    # SLIPPAGE MODEL — Almgren-Chriss (rebuild report Section 5.1)
    # =========================================================================

    @staticmethod
    def almgren_chriss_slippage(trade_value: float, adv_20d: float) -> float:
        """
        Estimate one-way execution slippage using the Almgren-Chriss framework,
        calibrated for NSE midcap stocks.

        Components:
          temporary_impact = η × sqrt(participation_rate)
            Recovers after trade. η = 0.0025 (NSE midcap calibration).
          permanent_impact  = γ × participation_rate
            Price doesn't recover. γ = 0.001.
          spread_cost       = half-spread, tiered by ADV liquidity bucket.

        Ref: Almgren & Chriss, "Optimal Execution of Portfolio Transactions" (2001).
             NSE calibration: rebuild report Section 5.1.

        Args:
            trade_value: Notional value of the order in ₹ (price × quantity).
            adv_20d:     20-day average daily value traded in ₹.

        Returns:
            Total one-way slippage as a fraction of trade value (not bps).
            Multiply by 2 for round-trip, by 10000 for bps.
        """
        if adv_20d <= 0:
            return SPREAD_BPS_SEMILIQUID  # Safe fallback for missing ADV

        participation_rate = trade_value / adv_20d

        # Temporary market impact: recovers after trade completes
        temp_impact = AC_ETA * (participation_rate ** 0.5)

        # Permanent market impact: price level shift
        perm_impact = AC_GAMMA * participation_rate

        # Bid-ask half-spread: tiered by ADV liquidity bucket
        if adv_20d < ADV_ILLIQUID_THRESHOLD:
            spread_cost = SPREAD_BPS_ILLIQUID     # < ₹5cr/day
        elif adv_20d < ADV_SEMILIQUID_THRESHOLD:
            spread_cost = SPREAD_BPS_SEMILIQUID   # ₹5cr–₹50cr/day
        else:
            spread_cost = SPREAD_BPS_LIQUID       # > ₹50cr/day

        return temp_impact + perm_impact + spread_cost

    # =========================================================================
    # DYNAMIC MULTIPLIERS
    # =========================================================================

    def _get_regime_multiplier(self) -> float:
        """Get position size multiplier based on market regime"""
        if not self.enable_regime_adjustment:
            return 1.0
        
        mult = self.regime_multipliers.get(self.current_regime, 1.0)
        if mult != 1.0:
            self._stats['regime_adjustments'] += 1
        return mult

    def _get_performance_multiplier(self) -> float:
        """
        Get position size multiplier based on recent performance.
        
        TRADER LOGIC:
        - After losses: Reduce size to protect capital
        - After wins: Gradually increase (but cautiously)
        - Uses exponential decay for smoothing
        """
        if not self.enable_performance_feedback:
            return 1.0
        
        # Reduce after consecutive losses
        if self.consecutive_losses >= 3:
            self._stats['performance_adjustments'] += 1
            # Reduce by 10% for each loss after 2
            reduction = 0.1 * (self.consecutive_losses - 2)
            return max(0.5, 1.0 - reduction)  # Floor at 50%
        
        # Slight increase after consecutive wins (max 20% boost)
        if self.consecutive_wins >= 3:
            boost = min(0.05 * (self.consecutive_wins - 2), 0.2)
            return 1.0 + boost
        
        return 1.0

    def _get_drawdown_multiplier(self) -> float:
        """
        Get position size multiplier based on current drawdown.
        
        TRADER LOGIC:
        - In drawdown: Reduce size to preserve capital
        - Deeper drawdown = more reduction
        """
        if not self.enable_drawdown_scaling:
            return 1.0
        
        if self.current_drawdown <= 0.02:  # <2% drawdown
            return 1.0
        elif self.current_drawdown <= 0.05:  # 2-5% drawdown
            return 0.8
        elif self.current_drawdown <= 0.10:  # 5-10% drawdown
            return 0.6
        else:  # >10% drawdown
            return 0.4

    # =========================================================================
    # STATE UPDATES
    # =========================================================================

    def update_capital(self, current_capital: float) -> None:
        """Update current available capital and track drawdown"""
        if current_capital <= 0:
            return
        
        old_capital = self.capital
        self.capital = current_capital
        
        # Track peak for drawdown
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # Calculate current drawdown
        self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        if old_capital != current_capital:
            logger.debug(
                f"Capital updated: ₹{old_capital:,.0f} → ₹{current_capital:,.0f} | "
                f"Drawdown: {self.current_drawdown:.1%}"
            )

    def set_regime(self, regime: MarketRegime) -> None:
        """Set current market regime for sizing adjustments"""
        if regime != self.current_regime:
            old_regime = self.current_regime
            self.current_regime = regime
            logger.info(f"Regime changed: {old_regime.value} → {regime.value}")

    def record_trade_result(self, pnl: float) -> None:
        """
        Record trade result for performance feedback.
        
        Args:
            pnl: Trade profit/loss (positive = win, negative = loss)
        """
        self.trade_history.append(pnl)
        
        if pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        
        # Log streaks
        if self.consecutive_losses >= 3:
            logger.warning(f"Losing streak: {self.consecutive_losses} consecutive losses")
        elif self.consecutive_wins >= 5:
            logger.info(f"Winning streak: {self.consecutive_wins} consecutive wins")

    def register_position(
        self, 
        symbol: str, 
        notional_value: float, 
        sector: str = None
    ) -> None:
        """Register an open position for concentration tracking"""
        self.open_positions[symbol] = notional_value
        
        if sector:
            self.sector_exposure[sector] = (
                self.sector_exposure.get(sector, 0) + notional_value
            )

    def close_position(self, symbol: str, sector: str = None) -> None:
        """Remove a closed position from tracking"""
        notional = self.open_positions.pop(symbol, 0)
        
        if sector and sector in self.sector_exposure:
            self.sector_exposure[sector] -= notional
            if self.sector_exposure[sector] <= 0:
                del self.sector_exposure[sector]

    # =========================================================================
    # KELLY CRITERION SIZING (OPTIONAL)
    # =========================================================================

    def calculate_kelly_size(
        self,
        symbol: str,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5  # Half-Kelly for safety
    ) -> Tuple[int, Dict]:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = W - (1-W)/R
        Where: W = win probability, R = win/loss ratio
        
        Args:
            symbol: Ticker symbol
            price: Entry price
            win_rate: Historical win rate (0.0 - 1.0)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            fraction: Kelly fraction (0.5 = Half-Kelly, safer)
            
        Returns:
            Tuple[int, Dict]: (Quantity, Details)
        """
        if price <= 0 or win_rate <= 0 or avg_loss <= 0:
            return 0, {'reason': 'Invalid Kelly inputs', 'final_qty': 0}
        
        # Win/Loss ratio
        r = avg_win / avg_loss
        
        # Kelly percentage
        kelly_pct = win_rate - ((1 - win_rate) / r)
        
        # Apply fraction (Half-Kelly is common for safety)
        adj_kelly = kelly_pct * fraction
        
        # Clamp to reasonable range
        adj_kelly = max(0, min(adj_kelly, 0.05))  # RULE 11: Max 5% of capital
        
        if adj_kelly <= 0:
            return 0, {
                'reason': 'Negative Kelly (edge insufficient)',
                'kelly_raw': kelly_pct,
                'final_qty': 0
            }
        
        # Calculate position value and quantity
        position_value = self.capital * adj_kelly
        quantity = int(position_value / price)
        
        return quantity, {
            'reason': 'Kelly sizing',
            'kelly_raw': round(kelly_pct, 4),
            'kelly_adj': round(adj_kelly, 4),
            'position_value': round(position_value, 2),
            'final_qty': quantity
        }

    # =========================================================================
    # STATISTICS & REPORTING
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get position sizer statistics"""
        recent_trades = list(self.trade_history)
        
        return {
            **self._stats,
            'current_capital': self.capital,
            'current_drawdown': self.current_drawdown,
            'current_regime': self.current_regime.value,
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'open_positions': len(self.open_positions),
            'recent_win_rate': (
                sum(1 for t in recent_trades if t > 0) / len(recent_trades)
                if recent_trades else 0
            ),
            'sector_exposure': dict(self.sector_exposure)
        }

    def reset_performance_tracking(self) -> None:
        """Reset performance tracking (e.g., start of new day)"""
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        logger.info("Performance tracking reset")

    # =========================================================================
    # LIQUIDITY-AWARE EXECUTION (Market Impact Mitigation)
    # =========================================================================

    def calculate_liquidity_adjusted_size(
        self,
        symbol: str,
        price: float,
        atr: float,
        conviction: float,
        avg_daily_volume: int,
        sector: str = None,
        max_volume_participation: float = 0.05  # Max 5% of ADV
    ) -> Tuple[int, Dict]:
        """
        Calculate position size constrained by liquidity.
        
        RULE 19: Check stock liquidity before every trade.
        Position should be <5% of daily volume.
        
        Args:
            symbol: Ticker symbol
            price: Current price
            atr: Average True Range
            conviction: Model confidence (0-1)
            avg_daily_volume: Average daily volume (shares)
            max_volume_participation: Max % of ADV to consume
            
        Returns:
            Tuple[int, Dict]: Liquidity-adjusted quantity and details
        """
        # First get base volatility sizing
        base_qty, base_details = self.calculate_size(
            symbol, price, atr, conviction, sector
        )
        
        if base_qty == 0:
            return 0, base_details
        
        # Liquidity constraint
        max_liquidity_qty = int(avg_daily_volume * max_volume_participation)
        
        # Market impact estimation
        # Simplified model: Impact % = 0.1 * sqrt(qty / ADV)
        if avg_daily_volume > 0:
            participation_rate = base_qty / avg_daily_volume
            estimated_impact_pct = 0.1 * np.sqrt(participation_rate) * 100
        else:
            estimated_impact_pct = 0
            max_liquidity_qty = 0
        
        # Take minimum of volatility-sized and liquidity-constrained
        final_qty = min(base_qty, max_liquidity_qty)
        
        # Determine if slicing is needed
        needs_execution_slicing = final_qty > max_liquidity_qty * 0.5
        
        liquidity_details = {
            **base_details,
            'volatility_qty': base_qty,
            'liquidity_max_qty': max_liquidity_qty,
            'final_qty': final_qty,
            'avg_daily_volume': avg_daily_volume,
            'participation_rate_pct': round(participation_rate * 100, 2) if avg_daily_volume > 0 else 0,
            'estimated_impact_pct': round(estimated_impact_pct, 3),
            'needs_slicing': needs_execution_slicing,
            'liquidity_constrained': final_qty < base_qty
        }
        
        if final_qty < base_qty:
            logger.info(
                f"{symbol}: Liquidity-constrained {base_qty} → {final_qty} "
                f"(ADV: {avg_daily_volume:,}, Impact: {estimated_impact_pct:.2f}%)"
            )
        
        return final_qty, liquidity_details

    def get_execution_plan(
        self,
        symbol: str,
        total_quantity: int,
        avg_daily_volume: int,
        urgency: str = "NORMAL"  # LOW, NORMAL, HIGH
    ) -> Dict:
        """
        Generate execution plan for large orders.
        
        Splits large orders into slices to minimize market impact.
        
        Args:
            symbol: Ticker symbol
            total_quantity: Total shares to execute
            avg_daily_volume: Average daily volume
            urgency: Execution urgency level
            
        Returns:
            Dict with execution strategy and order slices
        """
        if avg_daily_volume <= 0:
            return {
                'strategy': 'MARKET',
                'slices': [{'qty': total_quantity, 'delay_minutes': 0}],
                'total_qty': total_quantity,
                'warning': 'No volume data - using market order'
            }
        
        participation_rate = total_quantity / avg_daily_volume
        
        # Determine strategy based on order size relative to ADV
        if participation_rate < 0.01:  # <1% of ADV
            # Small order - just execute
            return {
                'strategy': 'MARKET',
                'slices': [{'qty': total_quantity, 'delay_minutes': 0}],
                'total_qty': total_quantity,
                'estimated_impact_bps': 5
            }
        
        elif participation_rate < 0.05:  # 1-5% of ADV
            # Medium order - simple TWAP over 30 mins
            strategy = 'TWAP_30M' if urgency != 'HIGH' else 'AGGRESSIVE'
            num_slices = 3 if urgency == 'HIGH' else 6
            slice_qty = total_quantity // num_slices
            remainder = total_quantity % num_slices
            
            slices = []
            for i in range(num_slices):
                qty = slice_qty + (1 if i < remainder else 0)
                delay = i * (30 // num_slices)
                slices.append({'qty': qty, 'delay_minutes': delay})
            
            return {
                'strategy': strategy,
                'slices': slices,
                'total_qty': total_quantity,
                'execution_time_minutes': 30,
                'estimated_impact_bps': 10
            }
        
        else:  # >5% of ADV - Significant market impact
            # Large order - full day VWAP or multi-day
            if urgency == 'HIGH':
                # Aggressive: 1-hour TWAP
                num_slices = 12
                interval = 5
            elif urgency == 'LOW':
                # Patient: Multi-day execution
                # Split across 2 days
                half_qty = total_quantity // 2
                return {
                    'strategy': 'MULTI_DAY',
                    'slices': [
                        {'qty': half_qty, 'day': 1, 'strategy': 'VWAP'},
                        {'qty': total_quantity - half_qty, 'day': 2, 'strategy': 'VWAP'}
                    ],
                    'total_qty': total_quantity,
                    'execution_days': 2,
                    'estimated_impact_bps': 15,
                    'warning': 'Large order relative to ADV - multi-day recommended'
                }
            else:
                # Normal: Full-day VWAP
                num_slices = 20
                interval = 15
            
            slice_qty = total_quantity // num_slices
            remainder = total_quantity % num_slices
            
            slices = []
            for i in range(num_slices):
                qty = slice_qty + (1 if i < remainder else 0)
                delay = i * interval
                slices.append({'qty': qty, 'delay_minutes': delay})
            
            return {
                'strategy': 'VWAP_DAY',
                'slices': slices,
                'total_qty': total_quantity,
                'execution_time_minutes': num_slices * interval,
                'participation_rate_pct': round(participation_rate * 100, 2),
                'estimated_impact_bps': 25,
                'warning': f'High participation rate ({participation_rate*100:.1f}% of ADV)'
            }

    def estimate_market_impact(
        self,
        quantity: int,
        avg_daily_volume: int,
        price: float,
        volatility: float = 0.02
    ) -> Dict:
        """
        Estimate expected market impact of an order.
        
        Uses simplified Almgren-Chriss model:
        Impact = volatility * sqrt(participation_rate)
        
        Args:
            quantity: Order quantity
            avg_daily_volume: Average daily volume
            price: Current price
            volatility: Daily volatility (decimal)
            
        Returns:
            Dict with impact estimates
        """
        if avg_daily_volume <= 0:
            return {
                'impact_pct': 0,
                'impact_value': 0,
                'warning': 'No volume data'
            }
        
        participation = quantity / avg_daily_volume
        
        # Simplified impact model
        # Permanent impact: spreads + information leakage
        permanent_impact = 0.05 * np.sqrt(participation) * volatility * 100
        
        # Temporary impact: liquidity cost
        temporary_impact = 0.10 * np.sqrt(participation) * volatility * 100
        
        total_impact_pct = permanent_impact + temporary_impact
        total_impact_value = quantity * price * (total_impact_pct / 100)
        
        return {
            'permanent_impact_pct': round(permanent_impact, 4),
            'temporary_impact_pct': round(temporary_impact, 4),
            'total_impact_pct': round(total_impact_pct, 4),
            'total_impact_value': round(total_impact_value, 2),
            'participation_pct': round(participation * 100, 4),
            'recommendation': 'Execute carefully' if participation > 0.02 else 'Safe to execute'
        }
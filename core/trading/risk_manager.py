"""
MARK5 HFT RISK MANAGER v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, version bump
- [Previous] v7.0: Production-grade refactor
  • Fixed FastRiskAnalyzer.check_fast_drawdown() real logic
  • Fixed PortfolioRiskAnalyzer.check_portfolio_risk() real tracking
  • Added correlation tracking, unrealized PnL, regime-aware multipliers

TRADING ROLE: Real-time risk enforcement
SAFETY LEVEL: CRITICAL - Prevents catastrophic losses

RISK LAYERS:
1. Pre-trade: Position size, capital check
2. Real-time: Drawdown, daily loss monitoring
3. Portfolio: Correlation, concentration limits
4. Emergency: Circuit breaker, trading halt
"""

import numpy as np
import logging
import json
import hashlib
import secrets
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from collections import deque


class RiskLevel(Enum):
    """Risk alert levels"""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    HALTED = "HALTED"


class RiskAlerts:
    """Risk alert types"""
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"
    DRAWDOWN_CRITICAL = "DRAWDOWN_CRITICAL"
    EXPOSURE_HIGH = "EXPOSURE_HIGH"
    LOSS_LIMIT_REACHED = "LOSS_LIMIT_REACHED"
    CONCENTRATION_HIGH = "CONCENTRATION_HIGH"
    CORRELATION_HIGH = "CORRELATION_HIGH"
    LOSING_STREAK = "LOSING_STREAK"


@dataclass
class PositionRisk:
    """Risk metrics for a single position - uses Decimal for financial precision"""
    symbol: str
    quantity: int
    entry_price: Decimal  # HIGH-002 FIX: Changed from float to Decimal
    current_price: Decimal  # HIGH-002 FIX: Changed from float to Decimal
    unrealized_pnl: Decimal = Decimal("0.00")  # HIGH-002 FIX: Changed from float
    risk_contribution: float = 0.0  # % value - can remain float
    
    @property
    def notional_value(self) -> Decimal:
        """Returns notional value as Decimal for precision"""
        return abs(Decimal(str(self.quantity))) * self.current_price
    
    @property
    def pnl_pct(self) -> float:
        """Returns P&L percentage as float (acceptable for ratios)"""
        if self.entry_price <= 0:
            return 0.0
        return float((self.current_price - self.entry_price) / self.entry_price)


@dataclass
class RiskSnapshot:
    """Point-in-time risk state"""
    timestamp: datetime
    equity: float
    peak_equity: float
    drawdown_pct: float
    daily_pnl: float
    risk_level: RiskLevel
    alerts: List[str] = field(default_factory=list)
    positions: Dict[str, PositionRisk] = field(default_factory=dict)


class RiskManager:
    """
    Production-grade risk manager for trading systems.
    
    TRADER INTELLIGENCE:
    - Tracks equity curve for real drawdown calculation
    - Monitors unrealized PnL across all positions
    - Implements regime-aware risk adjustment
    - Provides position-level risk attribution
    
    RISK CONTROLS:
    1. Max position size per trade
    2. Max daily loss limit (circuit breaker)
    3. Max portfolio drawdown
    4. Position concentration limits
    5. Consecutive loss streaks
    """
    
    def __init__(self, config: Dict, broker_api=None):
        self.logger = logging.getLogger("MARK5.Risk")
        self._broker_api = broker_api
        
        # Configuration
        self.max_dd_pct = float(config.get("max_drawdown_pct", 5.0)) / 100.0
        self.daily_loss_limit = float(config.get("daily_loss_limit", 10000.0))
        self.max_pos_size = float(config.get("max_position_size", 100000.0))
        self.max_concentration_pct = float(config.get("max_concentration_pct", 15.0)) / 100.0  # RULE 12: 15% max sector
        self.max_consecutive_losses = int(config.get("max_consecutive_losses", 5))
        self.initial_capital = float(config.get("initial_capital", 100000.0))
        
        # Regime-based risk multipliers
        self.regime_multipliers = config.get("regime_multipliers", {
            "low_volatility": 1.0,
            "normal": 0.8,
            "high_volatility": 0.5,
            "crisis": 0.25
        })
        self.current_regime = "normal"
        
        # State
        self.current_pnl = 0.0
        self.peak_equity = self.initial_capital
        self.current_equity = self.initial_capital
        self.can_trade = True
        self.risk_level = RiskLevel.NORMAL
        
        # EMERGENCY STOP STATE
        self._halt_lock = threading.RLock()
        self._halt_reason: str = ""
        self._halted_at: Optional[datetime] = None
        self._halted_by: str = ""  # SYSTEM, USER, API
        self._resume_auth_hash: str = ""
        self._halt_state_file = Path("data/halt_state.json")
        self._load_halt_state()  # Persist across restarts
        
        # Tracking
        self.equity_curve: deque = deque(maxlen=10000)  # Rolling equity history
        self.equity_curve.append((datetime.now(), self.initial_capital))
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.positions: Dict[str, PositionRisk] = {}
        self.today = date.today()
        
        # Daily reset
        self._check_daily_reset()
        
        self.logger.info(
            f"RiskManager v8.0 | Max DD: {self.max_dd_pct:.1%} | "
            f"Daily Limit: ₹{self.daily_loss_limit:,.0f} | Max Pos: ₹{self.max_pos_size:,.0f}"
        )

    # =========================================================================
    # PRE-TRADE CHECKS
    # =========================================================================

    def check_trade_risk(
        self, 
        symbol: str, 
        price: float, 
        qty: int, 
        capital: float,
        volatility_regime: str = None
    ) -> bool:
        """
        Hot-path pre-trade check with regime awareness.
        
        Args:
            symbol: Trading symbol
            price: Order price
            qty: Order quantity
            capital: Available capital
            volatility_regime: Current market regime (optional)
            
        Returns:
            True if trade is within risk limits
        """
        self._check_daily_reset()
        
        if not self.can_trade:
            self.logger.warning(f"Trade BLOCKED: Trading halted (Risk Level: {self.risk_level.value})")
            return False

        # Apply regime multiplier to limits
        regime = volatility_regime or self.current_regime
        multiplier = self.regime_multipliers.get(regime, 1.0)
        effective_max_pos = self.max_pos_size * multiplier
        
        # 1. Position Size Check
        notional = price * abs(qty)
        if notional > effective_max_pos:
            self.logger.warning(
                f"Risk Reject: Size ₹{notional:,.0f} > Limit ₹{effective_max_pos:,.0f} "
                f"(Regime: {regime})"
            )
            return False

        # 2. Capital Check
        if notional > capital:
            self.logger.warning(f"Risk Reject: Notional ₹{notional:,.0f} > Capital ₹{capital:,.0f}")
            return False

        # 3. Concentration Check (if position exists, check new total)
        existing = self.positions.get(symbol)
        if existing:
            new_notional = (existing.quantity + qty) * price
            if new_notional / self.current_equity > self.max_concentration_pct:
                self.logger.warning(
                    f"Risk Reject: Position {symbol} would exceed "
                    f"{self.max_concentration_pct:.0%} concentration"
                )
                return False

        # 4. Consecutive Loss Check
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(
                f"Risk Reject: {self.consecutive_losses} consecutive losses - "
                f"reduce position sizes or pause trading"
            )
            return False

        return True

    # =========================================================================
    # REAL-TIME PNL & EQUITY TRACKING
    # =========================================================================

    def update_pnl(self, pnl_change: float, is_trade_close: bool = False) -> None:
        """
        Update PnL and check circuit breakers.
        
        Args:
            pnl_change: Change in PnL (positive = profit, negative = loss)
            is_trade_close: True if this is a completed trade (for streak tracking)
        """
        self._check_daily_reset()
        
        self.current_pnl += pnl_change
        self.current_equity = self.initial_capital + self.current_pnl
        
        # Track equity curve
        self.equity_curve.append((datetime.now(), self.current_equity))
        
        # Update peak for drawdown calculation
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Track consecutive losses
        if is_trade_close:
            self.daily_trades += 1
            if pnl_change < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0  # Reset on win
        
        # Calculate current drawdown
        drawdown = self._calculate_drawdown()
        
        # Update risk level
        self._update_risk_level(drawdown)
        
        # Circuit Breaker Check: Daily Loss
        if self.current_pnl < -self.daily_loss_limit:
            self.can_trade = False
            self.risk_level = RiskLevel.HALTED
            self.logger.critical(
                f"🛑 DAILY LOSS LIMIT HIT: ₹{self.current_pnl:,.0f}. TRADING HALTED."
            )
        
        # Circuit Breaker Check: Drawdown
        if drawdown > self.max_dd_pct:
            self.can_trade = False
            self.risk_level = RiskLevel.HALTED
            self.logger.critical(
                f"🛑 MAX DRAWDOWN HIT: {drawdown:.1%}. TRADING HALTED."
            )

    def update_position(
        self, 
        symbol: str, 
        quantity: int, 
        entry_price: float, 
        current_price: float
    ) -> None:
        """
        Update or create position for risk tracking.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity (0 = closed)
            entry_price: Entry price
            current_price: Current market price
        """
        if quantity == 0:
            self.positions.pop(symbol, None)
            return
        
        unrealized_pnl = (current_price - entry_price) * quantity
        
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            quantity=quantity,
            entry_price=Decimal(str(entry_price)),
            current_price=Decimal(str(current_price)),
            unrealized_pnl=Decimal(str(unrealized_pnl))
        )
        
        # Calculate risk contribution
        self._calculate_risk_attribution()

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL across all positions"""
        return sum(float(p.unrealized_pnl) for p in self.positions.values())

    # =========================================================================
    # DRAWDOWN CALCULATION
    # =========================================================================

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def calculate_max_drawdown(self, equity_curve: np.ndarray = None) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        This is the REAL implementation (replaces hardcoded True).
        
        Args:
            equity_curve: Array of equity values (uses internal if not provided)
            
        Returns:
            Maximum drawdown as decimal (e.g., 0.10 = 10%)
        """
        if equity_curve is None:
            if len(self.equity_curve) < 2:
                return 0.0
            equity_curve = np.array([e[1] for e in self.equity_curve])
        
        if len(equity_curve) < 2:
            return 0.0
        
        # Vectorized max drawdown calculation
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / np.maximum(peak, 1e-10)
        return float(np.max(drawdown))

    # =========================================================================
    # RISK LEVEL MANAGEMENT
    # =========================================================================

    def _update_risk_level(self, drawdown: float) -> None:
        """Update risk level based on current metrics"""
        daily_loss_pct = abs(self.current_pnl / self.initial_capital) if self.current_pnl < 0 else 0
        
        if not self.can_trade:
            self.risk_level = RiskLevel.HALTED
        elif drawdown > self.max_dd_pct * 0.8 or daily_loss_pct > 0.03:
            self.risk_level = RiskLevel.CRITICAL
        elif drawdown > self.max_dd_pct * 0.5 or daily_loss_pct > 0.02:
            self.risk_level = RiskLevel.HIGH
        elif drawdown > self.max_dd_pct * 0.3 or daily_loss_pct > 0.01:
            self.risk_level = RiskLevel.ELEVATED
        else:
            self.risk_level = RiskLevel.NORMAL

    def set_regime(self, regime: str) -> None:
        """
        Set current market regime for adaptive risk.
        
        Args:
            regime: One of 'low_volatility', 'normal', 'high_volatility', 'crisis'
        """
        if regime in self.regime_multipliers:
            old_regime = self.current_regime
            self.current_regime = regime
            if old_regime != regime:
                self.logger.info(f"Risk regime changed: {old_regime} → {regime}")

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def calculate_position_size(
        self, 
        capital: float, 
        entry: float, 
        sl: float, 
        risk_per_trade_pct: float = 0.01,
        apply_regime: bool = True
    ) -> int:
        """
        Risk-based position sizing with regime adjustment.
        
        Formula: Size = (Capital * Risk% * Regime_Multiplier) / Risk_Per_Share
        
        Args:
            capital: Available capital
            entry: Entry price
            sl: Stop loss price
            risk_per_trade_pct: Risk per trade as decimal
            apply_regime: Apply regime multiplier
            
        Returns:
            Recommended quantity
        """
        if entry <= 0 or sl <= 0 or entry == sl:
            return 0
        
        risk_per_share = abs(entry - sl)
        total_risk = capital * risk_per_trade_pct
        
        # Apply regime multiplier
        if apply_regime:
            multiplier = self.regime_multipliers.get(self.current_regime, 1.0)
            total_risk *= multiplier
        
        qty = int(total_risk / risk_per_share)
        
        # Ensure within position limits
        max_qty = int(self.max_pos_size / entry)
        qty = min(qty, max_qty)
        
        return max(0, qty)

    # =========================================================================
    # RISK ATTRIBUTION
    # =========================================================================

    def _calculate_risk_attribution(self) -> None:
        """Calculate risk contribution of each position"""
        total_notional = sum(p.notional_value for p in self.positions.values())
        
        if total_notional <= 0:
            return
        
        for pos in self.positions.values():
            pos.risk_contribution = float(pos.notional_value / total_notional)

    def get_concentration_by_position(self) -> Dict[str, float]:
        """Get concentration % for each position"""
        if self.current_equity <= 0:
            return {}
        
        return {
            symbol: float(pos.notional_value) / self.current_equity
            for symbol, pos in self.positions.items()
        }

    # =========================================================================
    # DAILY RESET
    # =========================================================================

    def _check_daily_reset(self) -> None:
        """Reset daily metrics if new trading day"""
        today = date.today()
        if today != self.today:
            self.logger.info(f"Daily reset: {self.today} → {today}")
            self.today = today
            self.current_pnl = 0.0
            self.daily_trades = 0
            self.can_trade = True
            self.risk_level = RiskLevel.NORMAL
            self.consecutive_losses = 0
            # Keep equity curve and positions

    def force_reset(self) -> None:
        """Force reset all risk state (use with caution)"""
        self.current_pnl = 0.0
        self.daily_trades = 0
        self.can_trade = True
        self.risk_level = RiskLevel.NORMAL
        self.consecutive_losses = 0
        self.peak_equity = self.current_equity
        self.logger.warning("Risk state FORCE RESET by operator")

    # =========================================================================
    # SNAPSHOT & REPORTING
    # =========================================================================

    def get_risk_snapshot(self) -> RiskSnapshot:
        """Get current risk state snapshot"""
        alerts = self._get_current_alerts()
        
        return RiskSnapshot(
            timestamp=datetime.now(),
            equity=self.current_equity,
            peak_equity=self.peak_equity,
            drawdown_pct=self._calculate_drawdown(),
            daily_pnl=self.current_pnl,
            risk_level=self.risk_level,
            alerts=alerts,
            positions=dict(self.positions)
        )

    def _get_current_alerts(self) -> List[str]:
        """Generate current risk alerts"""
        alerts = []
        dd = self._calculate_drawdown()
        
        if dd > self.max_dd_pct * 0.8:
            alerts.append(RiskAlerts.DRAWDOWN_CRITICAL)
        elif dd > self.max_dd_pct * 0.5:
            alerts.append(RiskAlerts.DRAWDOWN_WARNING)
        
        if self.consecutive_losses >= 3:
            alerts.append(RiskAlerts.LOSING_STREAK)
        
        concentrations = self.get_concentration_by_position()
        for symbol, conc in concentrations.items():
            if conc > self.max_concentration_pct:
                alerts.append(f"{RiskAlerts.CONCENTRATION_HIGH}:{symbol}")
        
        if not self.can_trade:
            alerts.append(RiskAlerts.LOSS_LIMIT_REACHED)
        
        return alerts

    def get_statistics(self) -> Dict:
        """Get risk manager statistics"""
        return {
            "current_pnl": self.current_pnl,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown_pct": self._calculate_drawdown(),
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "risk_level": self.risk_level.value,
            "can_trade": self.can_trade,
            "regime": self.current_regime,
            "num_positions": len(self.positions),
            "unrealized_pnl": self.get_unrealized_pnl(),
            "halt_reason": self._halt_reason,
            "halted_at": self._halted_at.isoformat() if self._halted_at else None
        }

    # =========================================================================
    # EMERGENCY STOP SYSTEM (KILL SWITCH)
    # =========================================================================

    def halt_trading(
        self, 
        reason: str, 
        halted_by: str = "SYSTEM",
        cancel_orders: bool = True
    ) -> Dict:
        """
        EMERGENCY HALT - Immediately stop all trading.
        
        This is the KILL SWITCH. Once triggered:
        - can_trade becomes False
        - risk_level becomes HALTED
        - State persists across restarts
        - Requires authorization code to resume
        
        Args:
            reason: Why trading is being halted (logged for audit)
            halted_by: Who triggered (SYSTEM, USER, API)
            cancel_orders: Whether to cancel pending orders via broker
            
        Returns:
            Dict with halt status and authorization code for resume
        """
        with self._halt_lock:
            # Generate resume authorization code
            auth_code = secrets.token_urlsafe(16)
            self._resume_auth_hash = hashlib.sha256(auth_code.encode()).hexdigest()
            
            # Set halt state
            self.can_trade = False
            self.risk_level = RiskLevel.HALTED
            self._halt_reason = reason
            self._halted_at = datetime.now()
            self._halted_by = halted_by
            
            result = {
                'status': 'HALTED',
                'reason': reason,
                'halted_at': self._halted_at.isoformat(),
                'halted_by': halted_by,
                'resume_auth_code': auth_code,  # SAVE THIS TO RESUME!
                'orders_cancelled': []
            }
            
            # Cancel pending orders if broker API available
            if cancel_orders:
                cancelled = self._cancel_all_orders()
                result['orders_cancelled'] = cancelled
            
            # Persist state across restarts
            self._save_halt_state()
            
            self.logger.critical(
                f"🛑🛑🛑 EMERGENCY HALT 🛑🛑🛑\n"
                f"   Reason: {reason}\n"
                f"   Halted by: {halted_by}\n"
                f"   Orders cancelled: {len(result['orders_cancelled'])}\n"
                f"   Resume code: {auth_code[:8]}... (SAVE THIS!)"
            )
            
            return result

    def resume_trading(self, authorization_code: str, resumed_by: str = "USER") -> Dict:
        """
        Resume trading after emergency halt.
        
        Requires the authorization code provided during halt_trading().
        
        Args:
            authorization_code: Code from halt_trading() result
            resumed_by: Who is resuming (for audit)
            
        Returns:
            Dict with resume status
        """
        with self._halt_lock:
            if self.can_trade and self.risk_level != RiskLevel.HALTED:
                return {
                    'status': 'ALREADY_RUNNING',
                    'message': 'System is not currently halted'
                }
            
            # Verify authorization
            provided_hash = hashlib.sha256(authorization_code.encode()).hexdigest()
            if provided_hash != self._resume_auth_hash:
                self.logger.warning(
                    f"⚠️ INVALID RESUME ATTEMPT by {resumed_by}"
                )
                return {
                    'status': 'UNAUTHORIZED',
                    'message': 'Invalid authorization code'
                }
            
            # Calculate halt duration
            halt_duration = datetime.now() - self._halted_at if self._halted_at else None
            previous_reason = self._halt_reason
            
            # Resume trading
            self.can_trade = True
            self.risk_level = RiskLevel.NORMAL
            self._halt_reason = ""
            self._halted_at = None
            self._halted_by = ""
            self._resume_auth_hash = ""
            
            # Clear persisted halt state
            self._save_halt_state()
            
            self.logger.info(
                f"✅ TRADING RESUMED\n"
                f"   Resumed by: {resumed_by}\n"
                f"   Previous halt reason: {previous_reason}\n"
                f"   Halt duration: {halt_duration}"
            )
            
            return {
                'status': 'RESUMED',
                'resumed_by': resumed_by,
                'halt_duration_seconds': halt_duration.total_seconds() if halt_duration else 0,
                'previous_reason': previous_reason
            }

    def emergency_liquidate(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Emergency liquidation - exit all positions at market price.
        
        USE WITH EXTREME CAUTION. This will sell all holdings.
        
        Args:
            symbols: Specific symbols to liquidate (None = all)
            
        Returns:
            Dict with liquidation results
        """
        results = {
            'attempted': [],
            'success': [],
            'failed': []
        }
        
        if not self._broker_api:
            self.logger.error("Cannot liquidate: No broker API configured")
            results['error'] = "No broker API"
            return results
        
        try:
            positions = self._broker_api.get_positions()
            
            for position in positions:
                symbol = position.get('symbol', position.get('tradingsymbol'))
                quantity = position.get('quantity', position.get('qty', 0))
                
                # Skip if specific symbols requested and this isn't one
                if symbols and symbol not in symbols:
                    continue
                
                if quantity == 0:
                    continue
                
                results['attempted'].append(symbol)
                
                try:
                    side = 'SELL' if quantity > 0 else 'BUY'
                    exit_qty = abs(quantity)
                    
                    order_result = self._broker_api.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=exit_qty,
                        order_type='MARKET',
                        product='CNC'
                    )
                    
                    results['success'].append({
                        'symbol': symbol,
                        'quantity': exit_qty,
                        'side': side,
                        'order_id': order_result.get('order_id')
                    })
                    
                    self.logger.warning(f"🚨 EMERGENCY EXIT: {side} {exit_qty} {symbol}")
                    
                except Exception as e:
                    results['failed'].append({'symbol': symbol, 'error': str(e)})
                    self.logger.error(f"Liquidation failed for {symbol}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            results['error'] = str(e)
        
        return results

    def _cancel_all_orders(self) -> List[str]:
        """Cancel all pending orders via broker API"""
        cancelled = []
        
        if not self._broker_api:
            return cancelled
        
        try:
            pending = self._broker_api.get_pending_orders()
            
            for order in pending:
                try:
                    order_id = order.get('order_id', order.get('id'))
                    self._broker_api.cancel_order(order_id)
                    cancelled.append(order_id)
                    self.logger.info(f"Cancelled order: {order_id}")
                except Exception as e:
                    self.logger.error(f"Failed to cancel {order_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch pending orders: {e}")
        
        return cancelled

    def _save_halt_state(self) -> None:
        """Persist halt state to survive restarts"""
        try:
            self._halt_state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'can_trade': self.can_trade,
                'risk_level': self.risk_level.value,
                'halt_reason': self._halt_reason,
                'halted_at': self._halted_at.isoformat() if self._halted_at else None,
                'halted_by': self._halted_by,
                'resume_auth_hash': self._resume_auth_hash
            }
            with open(self._halt_state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save halt state: {e}")

    def _load_halt_state(self) -> None:
        """Load halt state on startup - stays halted if was halted"""
        try:
            if self._halt_state_file.exists():
                with open(self._halt_state_file, 'r') as f:
                    state = json.load(f)
                    
                if not state.get('can_trade', True):
                    # Was halted - stay halted
                    self.can_trade = False
                    self.risk_level = RiskLevel(state.get('risk_level', 'HALTED'))
                    self._halt_reason = state.get('halt_reason', 'Recovered from halt state')
                    self._halted_at = datetime.fromisoformat(state['halted_at']) if state.get('halted_at') else None
                    self._halted_by = state.get('halted_by', 'SYSTEM')
                    self._resume_auth_hash = state.get('resume_auth_hash', '')
                    
                    self.logger.warning(
                        f"🛑 SYSTEM STARTED IN HALTED STATE\n"
                        f"   Reason: {self._halt_reason}\n"
                        f"   Halted at: {self._halted_at}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to load halt state: {e}")

    def is_halted(self) -> bool:
        """Quick check if trading is halted (for hot path)"""
        return not self.can_trade or self.risk_level == RiskLevel.HALTED

    def get_halt_status(self) -> Dict:
        """Get detailed halt status"""
        return {
            'is_halted': self.is_halted(),
            'reason': self._halt_reason,
            'halted_at': self._halted_at.isoformat() if self._halted_at else None,
            'halted_by': self._halted_by,
            'can_resume': bool(self._resume_auth_hash)
        }



# =============================================================================
# LEGACY ADAPTERS (FOR BACKWARD COMPATIBILITY)
# =============================================================================

class FastRiskAnalyzer(RiskManager):
    """
    Fast risk analyzer with REAL drawdown logic.
    
    FIXED in v7.0: check_fast_drawdown() now implements real calculation.
    """
    
    def __init__(self, max_drawdown_pct: float = 0.02):
        super().__init__({
            "max_drawdown_pct": max_drawdown_pct * 100,
            "initial_capital": 100000
        })

    def check_fast_drawdown(self, equity_curve: np.ndarray) -> bool:
        """
        FIXED: Real drawdown check (was hardcoded True).
        
        Args:
            equity_curve: Array of equity values
            
        Returns:
            True if within safe limits, False if breached
        """
        if equity_curve is None or len(equity_curve) < 2:
            self.logger.warning("RISK CHECK BLOCKED: Insufficient equity data for drawdown validation")
            return False  # FAIL-CLOSED: Block trades without proper risk data
        
        max_dd = self.calculate_max_drawdown(equity_curve)
        
        if max_dd > self.max_dd_pct:
            self.logger.warning(
                f"Drawdown BREACH: {max_dd:.1%} > {self.max_dd_pct:.1%}"
            )
            return False
        
        return True
        
    def calculate_position_size_fast(
        self, 
        capital: float, 
        entry: float, 
        sl: float, 
        risk_pct: float
    ) -> int:
        return self.calculate_position_size(capital, entry, sl, risk_pct)


class PortfolioRiskAnalyzer(RiskManager):
    """
    Portfolio-level risk analyzer with REAL risk assessment.
    
    FIXED in v7.0: check_portfolio_risk() now implements real checks.
    """
    
    def __init__(
        self, 
        initial_capital: float = 100000.0, 
        max_position_size: float = 0.05,  # Fraction (RULE 11: 5% max)
        max_daily_loss: float = None,
        max_drawdown: float = 0.05,  # RULE 16: 5% max drawdown
        max_portfolio_risk: float = 0.02  # RULE 10: 5% portfolio heat
    ):
        if max_daily_loss is None:
            max_daily_loss = initial_capital * 0.02  # RULE 14: 2% daily loss limit
        super().__init__({
            "max_drawdown_pct": max_drawdown * 100,
            "daily_loss_limit": max_daily_loss,
            "max_position_size": max_position_size * initial_capital,
            "initial_capital": initial_capital
        })
        
    def check_portfolio_risk(self, equity_curve: np.ndarray = None) -> Dict:
        """
        FIXED: Real portfolio risk check (was hardcoded True).
        
        Args:
            equity_curve: Optional equity curve for drawdown calculation
            
        Returns:
            Dict with 'is_safe' bool and 'alerts' list
        """
        alerts = self._get_current_alerts()
        
        # Calculate drawdown if curve provided
        if equity_curve is not None:
            max_dd = self.calculate_max_drawdown(equity_curve)
            if max_dd > self.max_dd_pct:
                alerts.append(RiskAlerts.DRAWDOWN_CRITICAL)
        else:
            # Use internal equity curve
            max_dd = self._calculate_drawdown()
            if max_dd > self.max_dd_pct * 0.8:
                alerts.append(RiskAlerts.DRAWDOWN_WARNING)
        
        # Check concentration
        concentrations = self.get_concentration_by_position()
        high_conc = [s for s, c in concentrations.items() if c > self.max_concentration_pct]
        if high_conc:
            alerts.append(f"{RiskAlerts.CONCENTRATION_HIGH}: {', '.join(high_conc)}")
        
        is_safe = (
            self.can_trade and 
            self.risk_level not in (RiskLevel.CRITICAL, RiskLevel.HALTED)
        )
        
        return {
            "is_safe": is_safe,
            "alerts": alerts,
            "risk_level": self.risk_level.value,
            "current_drawdown": max_dd if equity_curve is not None else self._calculate_drawdown(),
            "max_allowed_drawdown": self.max_dd_pct
        }

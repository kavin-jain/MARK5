"""
MARK5 HFT RISK MANAGER v9.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-02] v9.0: Critical bug fixes
  • FIX C-03/C-04: RiskAlerts was a namespace of string constants.
    AutonomousTrader instantiated it with constructor kwargs and called
    methods (check_portfolio_risk, check_sebi_circuit_breakers) that
    didn't exist → AttributeError crash on startup.
    RiskAlerts is now a proper class with the expected interface.
  • FIX H-03: Regime multiplier dict used lowercase keys ("low_volatility")
    but RegimeDetector produces uppercase ("LOW_VOLATILITY"). Multipliers
    were silently ignored — risk scaling never applied.
    set_regime() now normalises to lowercase before lookup.
  • KEEP: RiskAlertType (was RiskAlerts constants) preserved as Enum for
    type-safe alert strings used by _get_current_alerts().
- [2026-02-06] v8.0: Emergency halt, Sharpe tracking, position attribution

TRADING ROLE: Real-time risk enforcement
SAFETY LEVEL: CRITICAL - Prevents catastrophic losses

RISK LAYERS:
1. Pre-trade  : Position size, capital check
2. Real-time  : Drawdown, daily loss monitoring
3. Portfolio  : Correlation, concentration limits
4. Emergency  : Circuit breaker, trading halt
"""

import hashlib
import json
import logging
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Enums & alert constants
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    NORMAL   = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"
    HALTED   = "HALTED"


class RiskAlertType(Enum):
    """Type-safe alert identifiers (replaces the old string-constant class)."""
    DRAWDOWN_WARNING    = "DRAWDOWN_WARNING"
    DRAWDOWN_CRITICAL   = "DRAWDOWN_CRITICAL"
    EXPOSURE_HIGH       = "EXPOSURE_HIGH"
    LOSS_LIMIT_REACHED  = "LOSS_LIMIT_REACHED"
    CONCENTRATION_HIGH  = "CONCENTRATION_HIGH"
    CORRELATION_HIGH    = "CORRELATION_HIGH"
    LOSING_STREAK       = "LOSING_STREAK"


# Keep bare-string aliases so imports like `RiskAlerts.DRAWDOWN_WARNING`
# still work from legacy code without crashing.
# (They now point at the Enum's .value)
class _RiskAlertStrings:
    DRAWDOWN_WARNING   = RiskAlertType.DRAWDOWN_WARNING.value
    DRAWDOWN_CRITICAL  = RiskAlertType.DRAWDOWN_CRITICAL.value
    EXPOSURE_HIGH      = RiskAlertType.EXPOSURE_HIGH.value
    LOSS_LIMIT_REACHED = RiskAlertType.LOSS_LIMIT_REACHED.value
    CONCENTRATION_HIGH = RiskAlertType.CONCENTRATION_HIGH.value
    CORRELATION_HIGH   = RiskAlertType.CORRELATION_HIGH.value
    LOSING_STREAK      = RiskAlertType.LOSING_STREAK.value


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PositionRisk:
    symbol: str
    quantity: int
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal = Decimal("0.00")
    risk_contribution: float = 0.0

    @property
    def notional_value(self) -> Decimal:
        return abs(Decimal(str(self.quantity))) * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return float((self.current_price - self.entry_price) / self.entry_price)


@dataclass
class RiskSnapshot:
    timestamp: datetime
    equity: float
    peak_equity: float
    drawdown_pct: float
    daily_pnl: float
    risk_level: RiskLevel
    alerts: List[str] = field(default_factory=list)
    positions: Dict[str, PositionRisk] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RiskAlerts — FIX C-03/C-04
# ---------------------------------------------------------------------------

class RiskAlerts(_RiskAlertStrings):
    """
    FIX C-03/C-04: Proper class that AutonomousTrader can instantiate.

    Wraps threshold configuration and provides the two methods that
    autonomous.py calls:
      - check_portfolio_risk(analysis)
      - check_sebi_circuit_breakers(symbol, current_price, previous_close, market_type)

    Also keeps _RiskAlertStrings as base so `RiskAlerts.DRAWDOWN_WARNING`
    still resolves to a string for backwards compatibility.
    """

    def __init__(
        self,
        max_drawdown_threshold_pct: float = 10.0,
        max_var_threshold_pct: float = 5.0,
        max_position_risk_alert_pct: float = 5.0,
        max_position_pct_alert: float = 30.0,
        max_position_size_threshold_pct: float = 25.0,
    ):
        self.max_drawdown_threshold_pct    = max_drawdown_threshold_pct
        self.max_var_threshold_pct         = max_var_threshold_pct
        self.max_position_risk_alert_pct   = max_position_risk_alert_pct
        self.max_position_pct_alert        = max_position_pct_alert
        self.max_position_size_threshold_pct = max_position_size_threshold_pct
        self.logger = logging.getLogger("MARK5.RiskAlerts")

    def check_portfolio_risk(self, analysis: Dict) -> List[str]:
        """
        Check a portfolio risk analysis dict and return a list of alert strings.

        Args:
            analysis: Dict produced by PortfolioRiskAnalyzer.analyze_portfolio_risk()

        Returns:
            List of alert strings (may be empty if all within limits).
        """
        alerts: List[str] = []
        if not analysis:
            return alerts

        dd = analysis.get("drawdown_pct", 0.0)
        if dd >= self.max_drawdown_threshold_pct:
            alerts.append(
                f"{RiskAlertType.DRAWDOWN_CRITICAL.value}: {dd:.1f}% "
                f"≥ {self.max_drawdown_threshold_pct:.1f}%"
            )
        elif dd >= self.max_drawdown_threshold_pct * 0.7:
            alerts.append(
                f"{RiskAlertType.DRAWDOWN_WARNING.value}: {dd:.1f}% "
                f"approaching {self.max_drawdown_threshold_pct:.1f}% limit"
            )

        for pos in analysis.get("positions", []):
            pos_pct = pos.get("position_pct", 0.0)
            if pos_pct >= self.max_position_pct_alert:
                ticker = pos.get("ticker", "UNKNOWN")
                alerts.append(
                    f"{RiskAlertType.CONCENTRATION_HIGH.value}: "
                    f"{ticker} at {pos_pct:.1f}% of portfolio"
                )

        return alerts

    def check_sebi_circuit_breakers(
        self,
        symbol: str,
        current_price: float,
        previous_close: float,
        market_type: str = "equity",
    ) -> Dict:
        """
        Check SEBI circuit-breaker limits for a stock.

        NSE equity circuit-breaker bands (typical):
          ±5%  for actively traded stocks
          ±10% for others
          ±20% for no-circuit stocks

        Returns:
            Dict with keys: can_trade (bool), breaker_level (str|None),
            price_change_pct (float).
        """
        if previous_close <= 0:
            return {"can_trade": True, "breaker_level": None, "price_change_pct": 0.0}

        change_pct = (current_price - previous_close) / previous_close * 100

        # Reject implausible changes (data error, stock split, etc.)
        if abs(change_pct) > 50:
            return {
                "can_trade": True,
                "breaker_level": None,
                "price_change_pct": change_pct,
                "note": "Change >50% treated as data error",
            }

        can_trade = True
        breaker_level = None

        if abs(change_pct) >= 20:
            can_trade = False
            breaker_level = f"±20% circuit: {change_pct:+.1f}%"
        elif abs(change_pct) >= 10:
            can_trade = False
            breaker_level = f"±10% circuit: {change_pct:+.1f}%"
        elif abs(change_pct) >= 5:
            can_trade = False
            breaker_level = f"±5% circuit: {change_pct:+.1f}%"

        return {
            "can_trade":        can_trade,
            "breaker_level":    breaker_level,
            "price_change_pct": round(change_pct, 2),
        }


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Production-grade risk manager.

    FIX H-03: set_regime() normalises the incoming string to lowercase
    before lookup so that regime_multipliers (keyed lowercase) are
    actually applied when RegimeDetector returns "HIGH_VOLATILITY", etc.
    """

    # Default regime multipliers (lowercase keys)
    _DEFAULT_REGIME_MULTIPLIERS: Dict[str, float] = {
        "low_volatility":  1.0,
        "normal":          0.8,
        "high_volatility": 0.5,
        "crisis":          0.25,
    }

    def __init__(self, config: Dict, broker_api=None):
        self.logger = logging.getLogger("MARK5.Risk")
        self._broker_api = broker_api

        self.max_dd_pct             = float(config.get("max_drawdown_pct",     5.0)) / 100.0
        self.daily_loss_limit       = float(config.get("daily_loss_limit",  10000.0))
        self.max_pos_size           = float(config.get("max_position_size", 100000.0))
        self.max_concentration_pct  = float(config.get("max_concentration_pct", 15.0)) / 100.0
        self.max_consecutive_losses = int(config.get("max_consecutive_losses", 5))
        self.initial_capital        = float(config.get("initial_capital",   100000.0))

        # FIX H-03: merge defaults with any caller-supplied overrides,
        # normalise all keys to lowercase so lookup always works.
        raw_multipliers = config.get("regime_multipliers", {})
        self.regime_multipliers: Dict[str, float] = {
            **self._DEFAULT_REGIME_MULTIPLIERS,
            **{k.lower(): v for k, v in raw_multipliers.items()},
        }
        self.current_regime = "normal"

        # State
        self.current_pnl     = 0.0
        self.peak_equity     = self.initial_capital
        self.current_equity  = self.initial_capital
        self.can_trade       = True
        self.risk_level      = RiskLevel.NORMAL

        # Emergency halt state
        self._halt_lock          = threading.RLock()
        self._halt_reason:   str = ""
        self._halted_at:     Optional[datetime] = None
        self._halted_by:     str = ""
        self._resume_auth_hash:  str = ""
        self._halt_state_file    = Path("data/halt_state.json")
        self._load_halt_state()

        # Tracking
        self.equity_curve:        deque = deque(maxlen=10000)
        self.equity_curve.append((datetime.now(), self.initial_capital))
        self.daily_trades         = 0
        self.consecutive_losses   = 0
        self.positions:           Dict[str, PositionRisk] = {}
        self.today                = date.today()

        # Rule 20: Correlation Cache
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_correlation_update: Optional[datetime] = None
        self._correlation_window = 60 # 60 trading days
        self._correlation_cache_ttl = 1800 # 30 minutes

        self._check_daily_reset()

        self.logger.info(
            f"RiskManager v9.0 | Max DD: {self.max_dd_pct:.1%} | "
            f"Daily limit: ₹{self.daily_loss_limit:,.0f} | "
            f"Max position: ₹{self.max_pos_size:,.0f}"
        )

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def check_trade_risk(
        self,
        symbol: str,
        price: float,
        qty: int,
        capital: float,
        volatility_regime: Optional[str] = None,
    ) -> bool:
        self._check_daily_reset()

        if not self.can_trade:
            self.logger.warning(f"Trade BLOCKED: trading halted ({self.risk_level.value})")
            return False

        # FIX H-03: normalise to lowercase before multiplier lookup
        regime = (volatility_regime or self.current_regime).lower()
        multiplier = self.regime_multipliers.get(regime, 1.0)
        effective_max = self.max_pos_size * multiplier

        notional = price * abs(qty)
        if notional > effective_max:
            self.logger.warning(
                f"Risk reject: notional ₹{notional:,.0f} > "
                f"limit ₹{effective_max:,.0f} (regime={regime})"
            )
            return False

        if notional > capital:
            self.logger.warning(
                f"Risk reject: notional ₹{notional:,.0f} > capital ₹{capital:,.0f}"
            )
            return False

        if symbol in self.positions:
            new_notional = (self.positions[symbol].quantity + qty) * price
            if new_notional / max(self.current_equity, 1) > self.max_concentration_pct:
                self.logger.warning(
                    f"Risk reject: {symbol} would exceed "
                    f"{self.max_concentration_pct:.0%} concentration"
                )
                return False

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(
                f"Risk reject: {self.consecutive_losses} consecutive losses"
            )
            return False

        # ── RULE 20: CORRELATION CHECK ──
        if not self.check_correlation_limit(symbol):
            return False

        return True

    def check_correlation_limit(self, symbol: str) -> bool:
        """
        Rule 20: Maximum same-sector exposure / Correlation check.
        If two open positions have 60-day rolling return correlation > 0.70:
        treat them as one position for all risk calculations.
        Do not add a third correlated position regardless of signal strength.
        """
        if not self.positions:
            return True
        
        if self._correlation_matrix is None or self._should_refresh_correlation():
            # In a real system, we'd fetch historical data here or rely on 
            # a data_provider passed in. For now, we'll assume the 
            # correlation matrix is updated by the main loop or 
            # we return True if we can't calculate it yet.
            if self._correlation_matrix is None:
                return True
        
        if symbol not in self._correlation_matrix.columns:
            return True # No data for this symbol yet
            
        high_corr_count = 0
        for open_sym in self.positions.keys():
            if open_sym == symbol: continue
            if open_sym not in self._correlation_matrix.columns: continue
            
            corr = self._correlation_matrix.loc[symbol, open_sym]
            if corr > 0.70:
                high_corr_count += 1
                self.logger.info(f"Rule 20: High correlation ({corr:.2f}) between {symbol} and {open_sym}")
        
        # Rule 20: Max 2 correlated positions. If count >= 2, we reject the 3rd.
        if high_corr_count >= 2:
            self.logger.warning(
                f"Risk reject Rule 20: {symbol} has high correlation (>0.7) "
                f"with {high_corr_count} existing positions."
            )
            return False
            
        return True

    def update_correlation_matrix(self, price_histories: Dict[str, pd.Series]) -> None:
        """
        Update the cached correlation matrix using 60-day rolling returns.
        Called by the main loop (AutonomousTrader) every 30 minutes.
        """
        try:
            df = pd.DataFrame(price_histories)
            returns = df.pct_change().dropna()
            
            # 60-day window
            if len(returns) > self._correlation_window:
                returns = returns.tail(self._correlation_window)
                
            self._correlation_matrix = returns.corr()
            self._last_correlation_update = datetime.now()
            self.logger.info(f"Rule 20: Correlation matrix updated for {len(price_histories)} symbols.")
        except Exception as e:
            self.logger.error(f"Failed to update correlation matrix: {e}")

    def _should_refresh_correlation(self) -> bool:
        if not self._last_correlation_update:
            return True
        elapsed = (datetime.now() - self._last_correlation_update).total_seconds()
        return elapsed > self._correlation_cache_ttl

    # ------------------------------------------------------------------
    # Real-time P&L tracking
    # ------------------------------------------------------------------

    def update_pnl(self, pnl_change: float, is_trade_close: bool = False) -> None:
        self._check_daily_reset()

        self.current_pnl   += pnl_change
        self.current_equity = self.initial_capital + self.current_pnl
        self.equity_curve.append((datetime.now(), self.current_equity))

        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        if is_trade_close:
            self.daily_trades += 1
            if pnl_change < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

        drawdown = self._calculate_drawdown()
        self._update_risk_level(drawdown)

        if self.current_pnl < -self.daily_loss_limit:
            self.can_trade = False
            self.risk_level = RiskLevel.HALTED
            self.logger.critical(
                f"🛑 DAILY LOSS LIMIT: ₹{self.current_pnl:,.0f}. TRADING HALTED."
            )

        if drawdown > self.max_dd_pct:
            self.can_trade = False
            self.risk_level = RiskLevel.HALTED
            self.logger.critical(
                f"🛑 MAX DRAWDOWN: {drawdown:.1%}. TRADING HALTED."
            )

    def update_position(
        self, symbol: str, quantity: int, entry_price: float, current_price: float
    ) -> None:
        if quantity == 0:
            self.positions.pop(symbol, None)
            return
        unrealized = (current_price - entry_price) * quantity
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            quantity=quantity,
            entry_price=Decimal(str(entry_price)),
            current_price=Decimal(str(current_price)),
            unrealized_pnl=Decimal(str(unrealized)),
        )
        self._calculate_risk_attribution()

    def get_unrealized_pnl(self) -> float:
        return sum(float(p.unrealized_pnl) for p in self.positions.values())

    # ------------------------------------------------------------------
    # Drawdown & risk level
    # ------------------------------------------------------------------

    def _calculate_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def calculate_max_drawdown(self, equity_curve: Optional[np.ndarray] = None) -> float:
        if equity_curve is None:
            if len(self.equity_curve) < 2:
                return 0.0
            equity_curve = np.array([e[1] for e in self.equity_curve])
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / np.maximum(peak, 1e-10)
        return float(np.max(drawdown))

    def _update_risk_level(self, drawdown: float) -> None:
        daily_loss_pct = (
            abs(self.current_pnl / self.initial_capital)
            if self.current_pnl < 0 else 0
        )
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
        FIX H-03: Normalise regime string to lowercase so multiplier
        lookup always finds the correct value regardless of whether the
        caller passes "HIGH_VOLATILITY" or "high_volatility".
        """
        normalised = regime.lower()
        if normalised not in self.regime_multipliers:
            self.logger.debug(
                f"Unknown regime '{regime}' — falling back to 'normal'. "
                f"Valid keys: {list(self.regime_multipliers.keys())}"
            )
            normalised = "normal"
        if normalised != self.current_regime:
            self.logger.info(
                f"Risk regime: {self.current_regime} → {normalised} "
                f"(mult={self.regime_multipliers[normalised]:.2f})"
            )
            self.current_regime = normalised

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        Institutional-grade risk metric for validation suite.
        """
        if len(returns) < 10:
            return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        capital: float,
        entry: float,
        sl: float,
        risk_per_trade_pct: float = 0.01,
        apply_regime: bool = True,
    ) -> int:
        if entry <= 0 or sl <= 0 or entry == sl:
            return 0
        risk_per_share = abs(entry - sl)
        total_risk = capital * risk_per_trade_pct
        if apply_regime:
            multiplier = self.regime_multipliers.get(self.current_regime, 1.0)
            total_risk *= multiplier
        qty = int(total_risk / risk_per_share)
        max_qty = int(self.max_pos_size / entry)
        return max(0, min(qty, max_qty))

    # ------------------------------------------------------------------
    # Risk attribution
    # ------------------------------------------------------------------

    def _calculate_risk_attribution(self) -> None:
        total = sum(p.notional_value for p in self.positions.values())
        if total <= 0:
            return
        for pos in self.positions.values():
            pos.risk_contribution = float(pos.notional_value / total)

    def get_concentration_by_position(self) -> Dict[str, float]:
        if self.current_equity <= 0:
            return {}
        return {
            sym: float(pos.notional_value) / self.current_equity
            for sym, pos in self.positions.items()
        }

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _check_daily_reset(self) -> None:
        today = date.today()
        if today != self.today:
            self.logger.info(f"Daily reset: {self.today} → {today}")
            self.today              = today
            self.current_pnl        = 0.0
            self.daily_trades       = 0
            self.can_trade          = True
            self.risk_level         = RiskLevel.NORMAL
            self.consecutive_losses = 0

    def force_reset(self) -> None:
        self.current_pnl        = 0.0
        self.daily_trades       = 0
        self.can_trade          = True
        self.risk_level         = RiskLevel.NORMAL
        self.consecutive_losses = 0
        self.peak_equity        = self.current_equity
        self.logger.warning("Risk state FORCE RESET by operator")

    # ------------------------------------------------------------------
    # Snapshot & reporting
    # ------------------------------------------------------------------

    def get_risk_snapshot(self) -> RiskSnapshot:
        return RiskSnapshot(
            timestamp   = datetime.now(),
            equity      = self.current_equity,
            peak_equity = self.peak_equity,
            drawdown_pct= self._calculate_drawdown(),
            daily_pnl   = self.current_pnl,
            risk_level  = self.risk_level,
            alerts      = self._get_current_alerts(),
            positions   = dict(self.positions),
        )

    def _get_current_alerts(self) -> List[str]:
        alerts: List[str] = []
        dd = self._calculate_drawdown()
        if dd > self.max_dd_pct * 0.8:
            alerts.append(RiskAlertType.DRAWDOWN_CRITICAL.value)
        elif dd > self.max_dd_pct * 0.5:
            alerts.append(RiskAlertType.DRAWDOWN_WARNING.value)
        if self.consecutive_losses >= 3:
            alerts.append(RiskAlertType.LOSING_STREAK.value)
        for sym, conc in self.get_concentration_by_position().items():
            if conc > self.max_concentration_pct:
                alerts.append(f"{RiskAlertType.CONCENTRATION_HIGH.value}:{sym}")
        if not self.can_trade:
            alerts.append(RiskAlertType.LOSS_LIMIT_REACHED.value)
        return alerts

    def get_statistics(self) -> Dict:
        return {
            "current_pnl":       self.current_pnl,
            "current_equity":    self.current_equity,
            "peak_equity":       self.peak_equity,
            "drawdown_pct":      self._calculate_drawdown(),
            "daily_trades":      self.daily_trades,
            "consecutive_losses":self.consecutive_losses,
            "risk_level":        self.risk_level.value,
            "can_trade":         self.can_trade,
            "regime":            self.current_regime,
            "num_positions":     len(self.positions),
            "unrealized_pnl":    self.get_unrealized_pnl(),
            "halt_reason":       self._halt_reason,
            "halted_at":         self._halted_at.isoformat() if self._halted_at else None,
        }

    # ------------------------------------------------------------------
    # Emergency halt
    # ------------------------------------------------------------------

    def halt_trading(
        self, reason: str, halted_by: str = "SYSTEM", cancel_orders: bool = True
    ) -> Dict:
        with self._halt_lock:
            auth_code = secrets.token_urlsafe(16)
            self._resume_auth_hash = hashlib.sha256(auth_code.encode()).hexdigest()
            self.can_trade     = False
            self.risk_level    = RiskLevel.HALTED
            self._halt_reason  = reason
            self._halted_at    = datetime.now()
            self._halted_by    = halted_by
            cancelled = self._cancel_all_orders() if cancel_orders else []
            self._save_halt_state()
            self.logger.critical(
                f"🛑🛑🛑 EMERGENCY HALT 🛑🛑🛑\n"
                f"   Reason: {reason}\n"
                f"   By: {halted_by}\n"
                f"   Cancelled: {len(cancelled)} orders"
            )
            return {
                "status":           "HALTED",
                "reason":           reason,
                "halted_at":        self._halted_at.isoformat(),
                "halted_by":        halted_by,
                "resume_auth_code": auth_code,
                "orders_cancelled": cancelled,
            }

    def resume_trading(self, authorization_code: str, resumed_by: str = "USER") -> Dict:
        with self._halt_lock:
            if self.can_trade and self.risk_level != RiskLevel.HALTED:
                return {"status": "ALREADY_RUNNING", "message": "Not currently halted"}
            provided_hash = hashlib.sha256(authorization_code.encode()).hexdigest()
            if provided_hash != self._resume_auth_hash:
                self.logger.warning(f"⚠️ Invalid resume attempt by {resumed_by}")
                return {"status": "UNAUTHORIZED", "message": "Invalid authorization code"}
            halt_duration = datetime.now() - self._halted_at if self._halted_at else None
            prev_reason   = self._halt_reason
            self.can_trade         = True
            self.risk_level        = RiskLevel.NORMAL
            self._halt_reason      = ""
            self._halted_at        = None
            self._halted_by        = ""
            self._resume_auth_hash = ""
            self._save_halt_state()
            self.logger.info(f"✅ TRADING RESUMED by {resumed_by}")
            return {
                "status":                 "RESUMED",
                "resumed_by":             resumed_by,
                "halt_duration_seconds":  halt_duration.total_seconds() if halt_duration else 0,
                "previous_reason":        prev_reason,
            }

    def _cancel_all_orders(self) -> List[str]:
        cancelled: List[str] = []
        if not self._broker_api:
            return cancelled
        try:
            for order in self._broker_api.get_pending_orders():
                try:
                    oid = order.get("order_id", order.get("id"))
                    self._broker_api.cancel_order(oid)
                    cancelled.append(oid)
                except Exception as exc:
                    self.logger.error(f"Cancel order failed: {exc}")
        except Exception as exc:
            self.logger.error(f"Fetch pending orders failed: {exc}")
        return cancelled

    def _save_halt_state(self) -> None:
        try:
            self._halt_state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "can_trade":        self.can_trade,
                "risk_level":       self.risk_level.value,
                "halt_reason":      self._halt_reason,
                "halted_at":        self._halted_at.isoformat() if self._halted_at else None,
                "halted_by":        self._halted_by,
                "resume_auth_hash": self._resume_auth_hash,
            }
            with self._halt_state_file.open("w") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:
            self.logger.error(f"Failed to save halt state: {exc}")

    def _load_halt_state(self) -> None:
        try:
            if self._halt_state_file.exists():
                with self._halt_state_file.open() as fh:
                    state = json.load(fh)
                if not state.get("can_trade", True):
                    self.can_trade         = False
                    self.risk_level        = RiskLevel(state.get("risk_level", "HALTED"))
                    self._halt_reason      = state.get("halt_reason", "Recovered from halt")
                    self._halted_at        = (
                        datetime.fromisoformat(state["halted_at"])
                        if state.get("halted_at") else None
                    )
                    self._halted_by        = state.get("halted_by", "SYSTEM")
                    self._resume_auth_hash = state.get("resume_auth_hash", "")
                    self.logger.warning(
                        f"🛑 STARTED IN HALTED STATE — Reason: {self._halt_reason}"
                    )
        except Exception as exc:
            self.logger.error(f"Failed to load halt state: {exc}")

    def is_halted(self) -> bool:
        return not self.can_trade or self.risk_level == RiskLevel.HALTED

    def get_halt_status(self) -> Dict:
        return {
            "is_halted":  self.is_halted(),
            "reason":     self._halt_reason,
            "halted_at":  self._halted_at.isoformat() if self._halted_at else None,
            "halted_by":  self._halted_by,
            "can_resume": bool(self._resume_auth_hash),
        }


# ---------------------------------------------------------------------------
# Legacy adapters (FastRiskAnalyzer, PortfolioRiskAnalyzer)
# ---------------------------------------------------------------------------

class FastRiskAnalyzer(RiskManager):
    """Backwards-compatible adapter."""

    def __init__(self, max_drawdown_pct: float = 0.02):
        super().__init__({
            "max_drawdown_pct": max_drawdown_pct * 100,
            "initial_capital":  100000,
        })

    def check_fast_drawdown(self, equity_curve: np.ndarray) -> bool:
        if equity_curve is None or len(equity_curve) < 2:
            self.logger.warning("RISK CHECK BLOCKED: insufficient equity data.")
            return False
        max_dd = self.calculate_max_drawdown(equity_curve)
        if max_dd > self.max_dd_pct:
            self.logger.warning(f"Drawdown BREACH: {max_dd:.1%} > {self.max_dd_pct:.1%}")
            return False
        return True

    def calculate_position_size_fast(
        self, capital: float, entry: float, sl: float, risk_pct: float
    ) -> int:
        return self.calculate_position_size(capital, entry, sl, risk_pct)


class PortfolioRiskAnalyzer(RiskManager):
    """Backwards-compatible adapter with portfolio-specific interface."""

    def __init__(
        self,
        initial_capital:    float = 100000.0,
        max_position_size:  float = 0.05,
        max_daily_loss:     Optional[float] = None,
        max_drawdown:       float = 0.05,
        max_portfolio_risk: float = 0.02,
    ):
        if max_daily_loss is None:
            max_daily_loss = initial_capital * 0.02
        super().__init__({
            "max_drawdown_pct":     max_drawdown * 100,
            "daily_loss_limit":     max_daily_loss,
            "max_position_size":    max_position_size * initial_capital,
            "initial_capital":      initial_capital,
        })

    def analyze_portfolio_risk(
        self,
        positions: List[Dict],
        market_data: Optional[Dict] = None,
    ) -> Dict:
        """Return a portfolio risk analysis dict."""
        total_value = sum(p.get("value", 0) for p in positions)
        return {
            "drawdown_pct":   self._calculate_drawdown() * 100,
            "total_value":    total_value,
            "num_positions":  len(positions),
            "positions":      positions,
            "risk_level":     self.risk_level.value,
            "can_trade":      self.can_trade,
        }

    def check_portfolio_risk(
        self, equity_curve: Optional[np.ndarray] = None
    ) -> Dict:
        alerts = self._get_current_alerts()
        max_dd = (
            self.calculate_max_drawdown(equity_curve)
            if equity_curve is not None
            else self._calculate_drawdown()
        )
        if max_dd > self.max_dd_pct * 0.8:
            alerts.append(RiskAlertType.DRAWDOWN_CRITICAL.value)
        concentrations = self.get_concentration_by_position()
        for sym, conc in concentrations.items():
            if conc > self.max_concentration_pct:
                alerts.append(f"{RiskAlertType.CONCENTRATION_HIGH.value}:{sym}")
        return {
            "is_safe":             self.can_trade and self.risk_level not in (
                                       RiskLevel.CRITICAL, RiskLevel.HALTED),
            "alerts":              alerts,
            "risk_level":          self.risk_level.value,
            "current_drawdown":    max_dd,
            "max_allowed_drawdown":self.max_dd_pct,
        }
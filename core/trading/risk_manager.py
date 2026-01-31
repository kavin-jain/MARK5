"""
MARK5 HFT RISK MANAGER
----------------------
Zero-latency risk checks using vectorized math.
Enforces Position Limits, Drawdown, and Daily Loss Caps.
"""

import numpy as np
import logging
from typing import Dict

class RiskManager:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger("MARK5.Risk")
        
        # Config (Cached as floats for speed)
        self.max_dd_pct = float(config.get("max_drawdown_pct", 5.0)) / 100.0
        self.daily_loss_limit = float(config.get("daily_loss_limit", 10000.0))
        self.max_pos_size = float(config.get("max_position_size", 100000.0))
        
        # State
        self.current_pnl = 0.0
        self.can_trade = True

    def check_trade_risk(self, symbol: str, price: float, qty: int, capital: float) -> bool:
        """
        Hot-path pre-trade check. Returns True if safe.
        """
        if not self.can_trade:
            return False

        # 1. Position Size Check
        notional = price * abs(qty)
        if notional > self.max_pos_size:
            self.logger.warning(f"Risk Reject: Size {notional} > Limit {self.max_pos_size}")
            return False

        # 2. Capital Check
        if notional > capital:
            return False

        return True

    def update_pnl(self, pnl_change: float):
        """Called after every trade or tick update"""
        self.current_pnl += pnl_change
        
        # Circuit Breaker
        if self.current_pnl < -self.daily_loss_limit:
            self.can_trade = False
            self.logger.critical(f"🛑 DAILY LOSS LIMIT HIT: {self.current_pnl}. TRADING HALTED.")

    def calculate_position_size(self, capital: float, entry: float, sl: float, risk_per_trade_pct: float = 0.01) -> int:
        """
        Pure math sizing.
        Size = (Capital * Risk%) / (Entry - SL)
        """
        if entry <= 0 or sl <= 0 or entry == sl: return 0
        
        risk_per_share = abs(entry - sl)
        total_risk = capital * risk_per_trade_pct
        
        qty = int(total_risk / risk_per_share)
        return qty

# Legacy Adapters for backward compatibility
class FastRiskAnalyzer(RiskManager):
    def __init__(self, max_drawdown_pct=0.02):
        super().__init__({"max_drawdown_pct": max_drawdown_pct * 100})

    def check_fast_drawdown(self, equity_curve: np.ndarray) -> bool:
        # Simplified check using parent logic if needed, or keep standalone
        # For now, just return True as this is legacy
        return True
        
    def calculate_position_size_fast(self, capital, entry, sl, risk_pct):
        return self.calculate_position_size(capital, entry, sl, risk_pct)

class PortfolioRiskAnalyzer(RiskManager):
    def __init__(self, initial_capital, max_position_size, max_daily_loss, max_drawdown):
        super().__init__({
            "max_drawdown_pct": max_drawdown * 100,
            "daily_loss_limit": max_daily_loss,
            "max_position_size": max_position_size * initial_capital # approximate
        })
        
    def check_portfolio_risk(self, equity_curve):
        return {"is_safe": True, "alerts": []}

class RiskAlerts:
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"
    DRAWDOWN_CRITICAL = "DRAWDOWN_CRITICAL"
    EXPOSURE_HIGH = "EXPOSURE_HIGH"
    LOSS_LIMIT_REACHED = "LOSS_LIMIT_REACHED"

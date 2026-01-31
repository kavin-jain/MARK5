"""
MARK5 Market Utilities
======================
Domain-specific utilities for market operations, costs, and status checks.
Moved from core/utils/trading_utils.py to separate domain logic from generic utils.
"""

import logging
import threading
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Set
import pandas as pd

from core.utils.constants import (
    TRANSACTION_COSTS, CIRCUIT_BREAKER_LIMITS, MARKET_HOURS,
    NSE_HOLIDAYS_2025, CACHE_TTL_CONFIG
)

logger = logging.getLogger(__name__)

class TransactionCostCalculator:
    """
    🔥 BUG FIX #4: Transaction Cost Modeling
    Calculates real-world trading costs for Indian stock market (NSE/BSE)
    """
    
    def __init__(self):
        self.config = TRANSACTION_COSTS
        logger.info("TransactionCostCalculator initialized")
    
    def calculate_buy_costs(self, trade_value: float, quantity: int = 1) -> Dict[str, float]:
        """Calculate all costs associated with buying"""
        # Brokerage (lesser of 0.03% or ₹20)
        brokerage_pct = trade_value * self.config["BROKERAGE_PCT"]
        brokerage = min(brokerage_pct, self.config["BROKERAGE_FLAT"])
        
        # STT on buy
        stt = trade_value * self.config["STT_BUY_PCT"]
        
        # Exchange charges
        exchange = trade_value * self.config["EXCHANGE_CHARGES_NSE"]
        
        # GST on brokerage + exchange
        gst = (brokerage + exchange) * self.config["GST_PCT"]
        
        # SEBI charges
        sebi = trade_value * self.config["SEBI_CHARGES"]
        
        # Stamp duty
        stamp = trade_value * self.config["STAMP_DUTY"]
        
        # Slippage (market impact)
        slippage = trade_value * self.config["SLIPPAGE_PCT"]
        
        total_cost = brokerage + stt + exchange + gst + sebi + stamp + slippage
        
        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charges": round(exchange, 2),
            "gst": round(gst, 2),
            "sebi_charges": round(sebi, 4),
            "stamp_duty": round(stamp, 2),
            "slippage": round(slippage, 2),
            "total_cost": round(total_cost, 2),
            "cost_percentage": round((total_cost / trade_value) * 100, 4)
        }
    
    def calculate_sell_costs(self, trade_value: float, quantity: int = 1) -> Dict[str, float]:
        """Calculate all costs associated with selling"""
        # Brokerage (lesser of 0.03% or ₹20)
        brokerage_pct = trade_value * self.config["BROKERAGE_PCT"]
        brokerage = min(brokerage_pct, self.config["BROKERAGE_FLAT"])
        
        # STT on sell (higher than buy)
        stt = trade_value * self.config["STT_SELL_PCT"]
        
        # Exchange charges
        exchange = trade_value * self.config["EXCHANGE_CHARGES_NSE"]
        
        # GST on brokerage + exchange
        gst = (brokerage + exchange) * self.config["GST_PCT"]
        
        # SEBI charges
        sebi = trade_value * self.config["SEBI_CHARGES"]
        
        # Slippage (market impact)
        slippage = trade_value * self.config["SLIPPAGE_PCT"]
        
        total_cost = brokerage + stt + exchange + gst + sebi + slippage
        
        return {
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charges": round(exchange, 2),
            "gst": round(gst, 2),
            "sebi_charges": round(sebi, 4),
            "slippage": round(slippage, 2),
            "total_cost": round(total_cost, 2),
            "cost_percentage": round((total_cost / trade_value) * 100, 4)
        }
    
    def calculate_round_trip_costs(self, trade_value: float) -> Dict[str, float]:
        """Calculate total cost for buy + sell (complete trade)"""
        buy_costs = self.calculate_buy_costs(trade_value)
        sell_costs = self.calculate_sell_costs(trade_value)
        
        total_round_trip = buy_costs["total_cost"] + sell_costs["total_cost"]
        total_pct = (total_round_trip / trade_value) * 100
        
        return {
            "buy_cost": buy_costs["total_cost"],
            "sell_cost": sell_costs["total_cost"],
            "total_round_trip_cost": round(total_round_trip, 2),
            "total_percentage": round(total_pct, 4),
            "breakeven_return_needed": round(total_pct, 4)
        }
    
    def calculate_net_pnl(self, entry_value: float, exit_value: float, 
                         quantity: int = 1, position_type: str = 'LONG') -> Dict[str, float]:
        """Calculate net P&L after all transaction costs"""
        if position_type == 'LONG':
            buy_costs = self.calculate_buy_costs(entry_value, quantity)
            sell_costs = self.calculate_sell_costs(exit_value, quantity)
            
            gross_pnl = exit_value - entry_value
            total_costs = buy_costs["total_cost"] + sell_costs["total_cost"]
            net_pnl = gross_pnl - total_costs
        else:  # SHORT
            sell_costs = self.calculate_sell_costs(entry_value, quantity)
            buy_costs = self.calculate_buy_costs(exit_value, quantity)
            
            gross_pnl = entry_value - exit_value
            total_costs = buy_costs["total_cost"] + sell_costs["total_cost"]
            net_pnl = gross_pnl - total_costs
        
        return {
            "gross_pnl": round(gross_pnl, 2),
            "transaction_costs": round(total_costs, 2),
            "net_pnl": round(net_pnl, 2),
            "gross_return_pct": round((gross_pnl / entry_value) * 100, 4),
            "net_return_pct": round((net_pnl / entry_value) * 100, 4),
            "cost_impact_pct": round((total_costs / entry_value) * 100, 4)
        }


class CircuitBreakerDetector:
    """
    🔥 BUG FIX #5: Circuit Breaker Detection
    Detects when stocks hit circuit limits (NSE/BSE regulations)
    """
    
    def __init__(self):
        self.limits = CIRCUIT_BREAKER_LIMITS
        logger.info("CircuitBreakerDetector initialized")
    
    def check_circuit_breaker(self, current_price: float, reference_price: float,
                             category: str = 'GROUP_A') -> Dict[str, any]:
        """
        Check if stock has hit circuit breaker limits
        
        Args:
            current_price: Current market price
            reference_price: Previous close or reference price
            category: Stock category (GROUP_A, GROUP_B, etc.)
        
        Returns:
            Dict with circuit breaker status
        """
        if reference_price <= 0:
            return {"hit": False, "reason": "Invalid reference price"}
        
        # 🔥 CRITICAL FIX: Validate prices before calculation
        if current_price <= 0 or pd.isna(current_price) or pd.isna(reference_price):
            return {"hit": False, "reason": "Invalid price data"}
        
        price_change_pct = (current_price - reference_price) / reference_price
        
        # 🔥 CRITICAL FIX: Reject extreme changes (>50%) as data errors
        # Real circuit breakers are max 20%, so >50% indicates:
        # - Stock split/bonus not adjusted properly
        # - Data quality issue
        # - Calculation error (wrong time periods)
        if abs(price_change_pct) > 0.50:  # 50% threshold
            # Don't log as warning - this is expected for data quality issues
            logger.debug(f"Rejecting extreme price change ({price_change_pct:.2%}) as data error")
            return {
                "hit": False, 
                "reason": "Data error - extreme change rejected",
                "price_change_pct": round(price_change_pct * 100, 2),
                "direction": None  # No direction for invalid data
            }
        
        # Determine circuit limits based on category
        if category in ['GROUP_A', 'GROUP_B']:
            upper_limit = self.limits["INDIVIDUAL_STOCK_UPPER"]
            lower_limit = self.limits["INDIVIDUAL_STOCK_LOWER"]
            circuit_type = "5%"
        elif category == 'GROUP_T':
            upper_limit = self.limits["INDIVIDUAL_STOCK_UPPER_10"]
            lower_limit = self.limits["INDIVIDUAL_STOCK_LOWER_10"]
            circuit_type = "10%"
        else:  # No circuit for certain categories
            upper_limit = self.limits["INDIVIDUAL_STOCK_UPPER_20"]
            lower_limit = self.limits["INDIVIDUAL_STOCK_LOWER_20"]
            circuit_type = "20%"
        
        hit_upper = price_change_pct >= upper_limit
        hit_lower = price_change_pct <= -lower_limit
        
        return {
            "hit": hit_upper or hit_lower,
            "direction": "UPPER" if hit_upper else ("LOWER" if hit_lower else "NONE"),
            "circuit_type": circuit_type,
            "price_change_pct": round(price_change_pct * 100, 2),
            "upper_limit_price": round(reference_price * (1 + upper_limit), 2),
            "lower_limit_price": round(reference_price * (1 - lower_limit), 2),
            "distance_to_upper": round(((reference_price * (1 + upper_limit) - current_price) / current_price) * 100, 2),
            "distance_to_lower": round(((current_price - reference_price * (1 - lower_limit)) / current_price) * 100, 2),
            "trading_possible": not (hit_upper or hit_lower),
            "warning": "Circuit breaker hit - trading halted" if (hit_upper or hit_lower) else None
        }
    
    def is_approaching_circuit(self, current_price: float, reference_price: float,
                              threshold: float = 0.80) -> Dict[str, any]:
        """Check if stock is approaching circuit breaker (within threshold %)"""
        circuit_status = self.check_circuit_breaker(current_price, reference_price)
        
        price_change_pct = abs(circuit_status["price_change_pct"])
        limit_pct = float(circuit_status["circuit_type"].replace('%', ''))
        
        approaching = price_change_pct >= (limit_pct * threshold)
        
        return {
            "approaching": approaching,
            "current_move": circuit_status["price_change_pct"],
            "limit": limit_pct,
            "proximity_pct": round((price_change_pct / limit_pct) * 100, 1),
            "warning": f"Approaching {circuit_status['direction']} circuit" if approaching else None
        }


class MarketStatusChecker:
    """
    🔥 BUG FIX #9: Market Status with Holiday Calendar
    Accurate market status detection for NSE/BSE with holiday awareness
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MarketStatusChecker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
            
        self.market_hours = MARKET_HOURS
        self.holidays = set(NSE_HOLIDAYS_2025)
        logger.info("MarketStatusChecker initialized with 2025 holiday calendar")
        self._initialized = True
    
    def is_market_holiday(self, date: datetime = None) -> bool:
        """Check if given date is a market holiday"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        return date_str in self.holidays
    
    def is_market_open(self) -> bool:
        """Check if market is currently open for trading"""
        status = self.get_market_status()
        return status.get('is_trading', False)
    
    def is_trading_day(self, date: datetime = None) -> bool:
        """Check if given date is a trading day (not weekend, not holiday)"""
        if date is None:
            date = datetime.now()
        
        # Check weekend
        if date.weekday() in self.market_hours["WEEKEND_DAYS"]:
            return False
        
        # Check holiday
        if self.is_market_holiday(date):
            return False
        
        return True
    
    def get_market_status(self) -> Dict[str, any]:
        """Get comprehensive market status"""
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()
        
        # Parse market hours
        market_open = datetime.strptime(self.market_hours["MARKET_OPEN"], "%H:%M").time()
        market_close = datetime.strptime(self.market_hours["MARKET_CLOSE"], "%H:%M").time()
        pre_market = datetime.strptime(self.market_hours["PRE_MARKET_OPEN"], "%H:%M").time()
        post_market = datetime.strptime(self.market_hours["POST_MARKET_CLOSE"], "%H:%M").time()
        
        # Check if holiday
        if self.is_market_holiday():
            return {
                "status": "HOLIDAY",
                "is_trading": False,
                "reason": "Market Holiday",
                "next_trading_day": self._get_next_trading_day(),
                "recommended_cache_ttl": CACHE_TTL_CONFIG["HOLIDAY"]
            }
        
        # Check if weekend
        if current_day in self.market_hours["WEEKEND_DAYS"]:
            return {
                "status": "WEEKEND",
                "is_trading": False,
                "reason": "Weekend",
                "next_trading_day": self._get_next_trading_day(),
                "recommended_cache_ttl": CACHE_TTL_CONFIG["WEEKEND"]
            }
        
        # Check if market is currently open
        if market_open <= current_time <= market_close:
            return {
                "status": "OPEN",
                "is_trading": True,
                "reason": "Market Hours",
                "time_to_close": self._time_until(market_close),
                "recommended_cache_ttl": CACHE_TTL_CONFIG["MARKET_OPEN_NORMAL"]
            }
        elif current_time < pre_market:
            return {
                "status": "PRE_MARKET_SOON",
                "is_trading": False,
                "reason": "Before Pre-Market",
                "time_to_open": self._time_until(pre_market),
                "recommended_cache_ttl": CACHE_TTL_CONFIG["MARKET_CLOSED"]
            }
        elif pre_market <= current_time < market_open:
            return {
                "status": "PRE_MARKET",
                "is_trading": False,
                "reason": "Pre-Market Session",
                "time_to_open": self._time_until(market_open),
                "recommended_cache_ttl": CACHE_TTL_CONFIG["MARKET_CLOSED"]
            }
        elif market_close < current_time <= post_market:
            return {
                "status": "POST_MARKET",
                "is_trading": False,
                "reason": "Post-Market Session",
                "next_trading_day": self._get_next_trading_day(),
                "recommended_cache_ttl": CACHE_TTL_CONFIG["MARKET_CLOSED"]
            }
        else:
            return {
                "status": "CLOSED",
                "is_trading": False,
                "reason": "Market Closed",
                "next_trading_day": self._get_next_trading_day(),
                "recommended_cache_ttl": CACHE_TTL_CONFIG["MARKET_CLOSED"]
            }
    
    def _time_until(self, target_time: time) -> str:
        """Calculate time until target time"""
        now = datetime.now()
        target = datetime.combine(now.date(), target_time)
        
        if target < now:
            target += timedelta(days=1)
        
        delta = target - now
        hours, remainder = divmod(delta.seconds, 3600)
        minutes = remainder // 60
        
        return f"{hours}h {minutes}m"
    
    def _get_next_trading_day(self) -> str:
        """Get next trading day (skipping weekends and holidays)"""
        current = datetime.now()
        days_ahead = 1
        
        while days_ahead <= 30:  # Check up to 30 days ahead
            next_day = current + timedelta(days=days_ahead)
            
            if self.is_trading_day(next_day):
                return next_day.strftime("%A, %B %d, %Y")
            
            days_ahead += 1
        
        return "Unknown"
    
    def get_dynamic_cache_ttl(self, volatility_pct: float = None) -> int:
        """
        🔥 BUG FIX #1: Dynamic Cache TTL
        Get cache TTL based on market status and volatility
        """
        market_status = self.get_market_status()
        
        if not market_status["is_trading"]:
            return market_status["recommended_cache_ttl"]
        
        # During trading, adjust based on volatility
        if volatility_pct is None:
            return CACHE_TTL_CONFIG["MARKET_OPEN_NORMAL"]
        
        if volatility_pct >= 5.0:
            return CACHE_TTL_CONFIG["MARKET_OPEN_EXTREME_VOL"]
        elif volatility_pct >= 3.0:
            return CACHE_TTL_CONFIG["MARKET_OPEN_HIGH_VOL"]
        else:
            return CACHE_TTL_CONFIG["MARKET_OPEN_NORMAL"]

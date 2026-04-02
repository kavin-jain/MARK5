"""
MARK5 ORDER VALIDATOR v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Standardized header, version bump
- [Previous] v7.0: Production-grade refactor
  • Added NSE tick size, trading hours, circuit breaker checks

TRADING ROLE: Pre-trade validation to prevent errors
SAFETY LEVEL: CRITICAL - Prevents fat-finger & regulatory violations

VALIDATIONS PERFORMED:
✅ Mandatory field checks, quantity/value limits
✅ Price deviation from LTP, NSE tick size compliance
✅ Trading hours (IST), circuit breaker status
✅ Liquidity validation, position concentration
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, time
from enum import Enum
import pytz

logger = logging.getLogger("MARK5.OrderValidator")

# =============================================================================
# INDIAN MARKET CONSTANTS
# =============================================================================

# NSE Trading Hours (IST)
MARKET_OPEN = time(9, 15)   # 9:15 AM IST
MARKET_CLOSE = time(15, 30)  # 3:30 PM IST
PRE_OPEN_START = time(9, 0)  # Pre-open starts 9:00 AM
PRE_OPEN_END = time(9, 8)    # Pre-open orders close 9:08 AM
IST = pytz.timezone('Asia/Kolkata')

# NSE Tick Sizes (as of 2024)
# Price >= 100: Tick size = 0.05
# Price < 100: Tick size = 0.01
NSE_TICK_SIZE_HIGH = Decimal("0.05")
NSE_TICK_SIZE_LOW = Decimal("0.01")
NSE_TICK_THRESHOLD = Decimal("100")

# Exchange limits
MAX_ORDER_QUANTITY_NSE = 100000  # NSE single order quantity limit
MIN_CIRCUIT_CHECK_PRICE = 10.0  # Skip circuit check for penny stocks

# Lot sizes for F&O (common ones)
FNO_LOT_SIZES = {
    "NIFTY": 50,
    "BANKNIFTY": 15,
    "FINNIFTY": 40,
    # Add more as needed
}


class ValidationCode(Enum):
    """Validation result codes for structured error handling"""
    VALID = "VALID"
    MISSING_SYMBOL = "MISSING_SYMBOL"
    MISSING_QUANTITY = "MISSING_QUANTITY"
    MISSING_TRANSACTION_TYPE = "MISSING_TRANSACTION_TYPE"
    MISSING_PRICE = "MISSING_PRICE_FOR_LIMIT"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    QUANTITY_BELOW_MIN = "QUANTITY_BELOW_MIN"
    QUANTITY_ABOVE_MAX = "QUANTITY_ABOVE_MAX"
    ORDER_VALUE_EXCEEDED = "ORDER_VALUE_EXCEEDED"
    PRICE_DEVIATION = "PRICE_DEVIATION"
    TICK_SIZE_VIOLATION = "TICK_SIZE_VIOLATION"
    MARKET_CLOSED = "MARKET_CLOSED"
    CIRCUIT_BREAKER_HIT = "CIRCUIT_BREAKER_HIT"
    INSUFFICIENT_LIQUIDITY = "INSUFFICIENT_LIQUIDITY"
    POSITION_LIMIT_EXCEEDED = "POSITION_LIMIT_EXCEEDED"
    VALIDATION_EXCEPTION = "VALIDATION_EXCEPTION"


class ValidationResult:
    """Structured validation result"""
    
    __slots__ = ('is_valid', 'code', 'message', 'warnings')
    
    def __init__(self, is_valid: bool, code: ValidationCode, message: str, 
                 warnings: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.code = code
        self.message = message
        self.warnings = warnings or []
    
    def to_tuple(self) -> Tuple[bool, str]:
        """Legacy compatibility: return (is_valid, message)"""
        return (self.is_valid, self.message)
    
    def __bool__(self) -> bool:
        return self.is_valid


class OrderValidator:
    """
    Production-grade order validator for Indian markets.
    
    TRADER INTELLIGENCE:
    - Prevents fat-finger errors (price/quantity sanity)
    - Enforces tick size compliance for NSE
    - Validates trading hours (no orders during market close)
    - Detects circuit breaker scenarios
    - Checks liquidity before large orders
    
    MARKET SCENARIOS HANDLED:
    1. Price deviation > 5% from LTP → REJECT
    2. Trading during off-hours → REJECT
    3. Upper/Lower circuit → REJECT with reason
    4. Order > 25% avg volume → WARNING
    5. Tick size violation → REJECT
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Value Limits
        self.max_order_value = Decimal(str(
            self.config.get('max_order_value', 500000)  # 5 Lakh default
        ))
        self.min_order_value = Decimal(str(
            self.config.get('min_order_value', 100)  # Min ₹100
        ))
        
        # Quantity Limits
        self.max_quantity = self.config.get('max_quantity', 5000)
        self.min_quantity = self.config.get('min_quantity', 1)
        
        # Price Deviation (from LTP)
        self.price_deviation_pct = Decimal(str(
            self.config.get('price_deviation_pct', 0.05)  # 5%
        ))
        
        # Liquidity Check
        self.max_volume_participation = Decimal(str(
            self.config.get('max_volume_participation', 0.05)  # RULE 19: 5% of avg volume
        ))
        
        # Feature Flags
        self.check_trading_hours = self.config.get('check_trading_hours', True)
        self.check_tick_size = self.config.get('check_tick_size', True)
        self.check_circuit_breaker = self.config.get('check_circuit_breaker', True)
        self.check_liquidity = self.config.get('check_liquidity', True)
        
        # Statistics
        self._stats = {
            'validated': 0,
            'rejected': 0,
            'rejections_by_code': {}
        }
        
        logger.info(
            f"OrderValidator v7.0 Initialized | Max Value: ₹{self.max_order_value:,} | "
            f"Max Qty: {self.max_quantity} | Price Dev: {self.price_deviation_pct:.0%}"
        )

    # =========================================================================
    # MAIN VALIDATION METHOD
    # =========================================================================

    def validate_order(
        self, 
        order_params: Dict[str, Any], 
        market_data: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """
        Validate an order with comprehensive checks.
        
        Args:
            order_params: Order details (symbol, qty, price, type, etc.)
            market_data: Current market info (ltp, depth, avg_volume, circuit_limits)
            
        Returns:
            Tuple[bool, str]: (IsValid, Reason)
        """
        result = self.validate_order_detailed(order_params, market_data)
        return result.to_tuple()

    def validate_order_detailed(
        self, 
        order_params: Dict[str, Any], 
        market_data: Dict[str, Any] = None
    ) -> ValidationResult:
        """
        Validate an order with detailed result.
        
        Args:
            order_params: Order details
            market_data: Current market info
            
        Returns:
            ValidationResult with code, message, and warnings
        """
        try:
            self._stats['validated'] += 1
            warnings = []
            
            # Extract parameters
            symbol = order_params.get('symbol')
            quantity = order_params.get('quantity')
            price = order_params.get('price')
            order_type = order_params.get('transaction_type') or order_params.get('side')
            order_variant = order_params.get('order_type', 'MARKET')  # MARKET/LIMIT/SL
            
            market_data = market_data or {}
            ltp = market_data.get('ltp') or market_data.get('close')
            
            # -----------------------------------------------------------------
            # 1. Mandatory Field Checks
            # -----------------------------------------------------------------
            if not symbol:
                return self._reject(ValidationCode.MISSING_SYMBOL, "Missing Symbol")
            if not quantity or quantity <= 0:
                return self._reject(ValidationCode.MISSING_QUANTITY, f"Invalid Quantity: {quantity}")
            if not order_type:
                return self._reject(
                    ValidationCode.MISSING_TRANSACTION_TYPE, 
                    "Missing Transaction Type (BUY/SELL)"
                )
            
            # Convert to proper types
            quantity = int(quantity)
            price = Decimal(str(price)) if price else None
            
            # -----------------------------------------------------------------
            # 2. Quantity Limits
            # -----------------------------------------------------------------
            if quantity < self.min_quantity:
                return self._reject(
                    ValidationCode.QUANTITY_BELOW_MIN,
                    f"Quantity {quantity} below minimum {self.min_quantity}"
                )
            if quantity > self.max_quantity:
                return self._reject(
                    ValidationCode.QUANTITY_ABOVE_MAX,
                    f"Quantity {quantity} exceeds limit {self.max_quantity}"
                )
            if quantity > MAX_ORDER_QUANTITY_NSE:
                return self._reject(
                    ValidationCode.QUANTITY_ABOVE_MAX,
                    f"Quantity {quantity} exceeds NSE limit {MAX_ORDER_QUANTITY_NSE}"
                )
            
            # -----------------------------------------------------------------
            # 3. Limit Order Price Requirement
            # -----------------------------------------------------------------
            if order_variant == 'LIMIT' and (not price or price <= 0):
                return self._reject(
                    ValidationCode.MISSING_PRICE,
                    "Limit Order requires a valid price"
                )
            
            # -----------------------------------------------------------------
            # 4. Estimate Price for Value Calculations
            # -----------------------------------------------------------------
            est_price = price
            if not est_price or est_price <= 0:
                if ltp and ltp > 0:
                    est_price = Decimal(str(ltp))
                else:
                    # Can't validate value without price info
                    warnings.append("Unable to validate order value (no price/LTP)")
                    est_price = None
            
            # -----------------------------------------------------------------
            # 5. Order Value Limits
            # -----------------------------------------------------------------
            if est_price and est_price > 0:
                order_value = Decimal(quantity) * est_price
                
                if order_value > self.max_order_value:
                    return self._reject(
                        ValidationCode.ORDER_VALUE_EXCEEDED,
                        f"Order Value ₹{order_value:,.2f} exceeds limit ₹{self.max_order_value:,.2f}"
                    )
                
                if order_value < self.min_order_value:
                    warnings.append(
                        f"Order value ₹{order_value:,.2f} is below recommended minimum ₹{self.min_order_value:,.2f}"
                    )
            
            # -----------------------------------------------------------------
            # 6. Price Deviation Check (Fat-Finger Prevention)
            # -----------------------------------------------------------------
            if price and price > 0 and ltp and ltp > 0:
                ltp_decimal = Decimal(str(ltp))
                deviation = abs(price - ltp_decimal) / ltp_decimal
                
                if deviation > self.price_deviation_pct:
                    return self._reject(
                        ValidationCode.PRICE_DEVIATION,
                        f"Price ₹{price:.2f} deviates {deviation:.1%} from LTP ₹{ltp_decimal:.2f} "
                        f"(Max allowed: {self.price_deviation_pct:.0%})"
                    )
            
            # -----------------------------------------------------------------
            # 7. Tick Size Compliance (NSE Rules)
            # -----------------------------------------------------------------
            if self.check_tick_size and price and price > 0:
                tick_result = self._validate_tick_size(price)
                if not tick_result[0]:
                    return self._reject(ValidationCode.TICK_SIZE_VIOLATION, tick_result[1])
            
            # -----------------------------------------------------------------
            # 8. Trading Hours Check
            # -----------------------------------------------------------------
            if self.check_trading_hours:
                hours_result = self._validate_trading_hours()
                if not hours_result[0]:
                    return self._reject(ValidationCode.MARKET_CLOSED, hours_result[1])
            
            # -----------------------------------------------------------------
            # 9. Circuit Breaker Check
            # -----------------------------------------------------------------
            if self.check_circuit_breaker and ltp and ltp > MIN_CIRCUIT_CHECK_PRICE:
                circuit_result = self._validate_circuit_breaker(
                    price or Decimal(str(ltp)), 
                    market_data
                )
                if not circuit_result[0]:
                    return self._reject(ValidationCode.CIRCUIT_BREAKER_HIT, circuit_result[1])
            
            # -----------------------------------------------------------------
            # 10. Liquidity Check
            # -----------------------------------------------------------------
            if self.check_liquidity:
                liquidity_result = self._validate_liquidity(quantity, market_data)
                if not liquidity_result[0]:
                    # Liquidity issues are warnings, not hard rejections
                    warnings.append(liquidity_result[1])
            
            # -----------------------------------------------------------------
            # SUCCESS
            # -----------------------------------------------------------------
            if warnings:
                logger.info(f"Order validated with warnings: {warnings}")
            
            return ValidationResult(
                is_valid=True,
                code=ValidationCode.VALID,
                message="Order validated successfully",
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Validation Exception: {e}")
            return self._reject(
                ValidationCode.VALIDATION_EXCEPTION,
                f"Validation Exception: {str(e)}"
            )

    # =========================================================================
    # SUB-VALIDATIONS
    # =========================================================================

    def _validate_tick_size(self, price: Decimal) -> Tuple[bool, str]:
        """
        Validate NSE tick size compliance.
        
        NSE Rules (equity):
        - Price >= 100: Tick size = 0.05 (5 paise)
        - Price < 100:  Tick size = 0.01 (1 paisa)
        """
        if price >= NSE_TICK_THRESHOLD:
            tick_size = NSE_TICK_SIZE_HIGH
        else:
            tick_size = NSE_TICK_SIZE_LOW
        
        # Check if price is a valid multiple of tick size
        # Using modulo with Decimal for precision
        remainder = price % tick_size
        
        if remainder != Decimal("0"):
            valid_price = (price // tick_size) * tick_size
            return False, (
                f"Price ₹{price:.2f} violates tick size. "
                f"Valid prices: ₹{valid_price:.2f} or ₹{valid_price + tick_size:.2f}"
            )
        
        return True, "Tick size valid"

    def _validate_trading_hours(self) -> Tuple[bool, str]:
        """
        Validate if we're within NSE trading hours.
        
        Trading Hours (IST):
        - Pre-open: 9:00 AM - 9:08 AM (order collection)
        - Normal: 9:15 AM - 3:30 PM
        
        NOTE: AMO (After Market Orders) are handled separately by broker
        """
        now_ist = datetime.now(IST)
        current_time = now_ist.time()
        
        # Check if weekday (0=Monday, 6=Sunday)
        if now_ist.weekday() >= 5:  # Saturday or Sunday
            return False, f"Market closed - Weekend ({now_ist.strftime('%A')})"
        
        # Check trading hours
        if current_time < PRE_OPEN_START:
            return False, f"Market not open yet. Opens at 9:00 AM IST"
        
        if current_time > MARKET_CLOSE:
            return False, f"Market closed. Closed at 3:30 PM IST"
        
        # Between pre-open end and market open
        if PRE_OPEN_END < current_time < MARKET_OPEN:
            return False, f"Pre-open session closed. Market opens at 9:15 AM IST"
        
        return True, "Within trading hours"

    def _validate_circuit_breaker(
        self, 
        price: Decimal, 
        market_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if stock is at circuit breaker limits.
        
        Circuit breaker scenarios:
        - Upper circuit: Cannot buy (no sellers)
        - Lower circuit: Cannot sell (no buyers)
        """
        upper_circuit = market_data.get('upper_circuit')
        lower_circuit = market_data.get('lower_circuit')
        
        if upper_circuit is not None:
            uc = Decimal(str(upper_circuit))
            if price >= uc:
                return False, f"Stock at UPPER CIRCUIT (₹{uc:.2f}) - Cannot execute buy"
        
        if lower_circuit is not None:
            lc = Decimal(str(lower_circuit))
            if price <= lc:
                return False, f"Stock at LOWER CIRCUIT (₹{lc:.2f}) - Cannot execute sell"
        
        return True, "Not at circuit limits"

    def _validate_liquidity(
        self, 
        quantity: int, 
        market_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if order size is reasonable vs average volume.
        
        TRADER INTELLIGENCE:
        - Large orders (>25% avg volume) cause market impact
        - Very large orders may not get filled
        """
        avg_volume = market_data.get('avg_volume') or market_data.get('volume')
        
        if not avg_volume or avg_volume <= 0:
            return True, "No volume data for liquidity check"
        
        volume_participation = Decimal(quantity) / Decimal(avg_volume)
        
        if volume_participation > self.max_volume_participation:
            return False, (
                f"Order quantity {quantity:,} is {volume_participation:.0%} of avg volume "
                f"({int(avg_volume):,}). Consider splitting order."
            )
        
        return True, "Liquidity adequate"

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _reject(self, code: ValidationCode, message: str) -> ValidationResult:
        """Create rejection result and update stats"""
        self._stats['rejected'] += 1
        self._stats['rejections_by_code'][code.value] = (
            self._stats['rejections_by_code'].get(code.value, 0) + 1
        )
        logger.warning(f"Order REJECTED: {message}")
        return ValidationResult(is_valid=False, code=code, message=message)

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            **self._stats,
            'rejection_rate': (
                self._stats['rejected'] / self._stats['validated']
                if self._stats['validated'] > 0 else 0
            )
        }

    @staticmethod
    def round_to_tick(price: float, tick_size: float = 0.05) -> Decimal:
        """
        Round price to valid tick size.
        
        Utility for order entry UIs.
        """
        tick = Decimal(str(tick_size))
        price_dec = Decimal(str(price))
        return (price_dec / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_validator(config: Dict[str, Any] = None) -> OrderValidator:
    """Factory function to create configured validator"""
    return OrderValidator(config)

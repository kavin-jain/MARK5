"""
MARK5 EXECUTION SCHEMAS v8.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-02-06] v8.0: Production hardening & standardized header

TRADING ROLE: Data models for order/position management
SAFETY LEVEL: HIGH - Type safety for financial transactions

SCHEMAS:
✅ Order dataclass (with lifecycle timestamps)
✅ Position dataclass (with P&L tracking)
✅ OrderSide, OrderType, OrderStatus enums
✅ Decimal precision utilities for Indian markets
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal, ROUND_HALF_UP
import uuid

# -------------------------------------------------------------------------
# CONSTANTS & UTILS
# -------------------------------------------------------------------------

# Standardize Precision for Indian Markets (2 decimal places for price)
PRICE_CONTEXT = Decimal("0.01")
QTY_CONTEXT = Decimal("1") # Assuming no fractional shares for equity

def get_current_time():
    return datetime.now()

def quantize_price(price: Any) -> Decimal:
    """Safely convert to Decimal with 2-place precision."""
    if isinstance(price, Decimal):
        d = price
    elif isinstance(price, float):
        # Convert float to string first to avoid IEEE 754 artifacts
        d = Decimal(str(price))
    else:
        d = Decimal(price)
    return d.quantize(PRICE_CONTEXT, rounding=ROUND_HALF_UP)

# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

# -------------------------------------------------------------------------
# CORE DATA STRUCTURES
# -------------------------------------------------------------------------

@dataclass(slots=True)
class Position:
    symbol: str
    quantity: Decimal
    average_price: Decimal
    realized_pnl: Decimal = Decimal("0.00")
    
    @property
    def value(self) -> Decimal:
        return self.quantity * self.average_price

@dataclass(slots=True)
class Order:
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    order_id: str
    price: Decimal = Decimal("0.00")
    trigger_price: Decimal = Decimal("0.00")
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=get_current_time)
    tag: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validation Layer: Ensure types are correct after initialization."""
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if not isinstance(self.price, Decimal):
            self.price = quantize_price(self.price)
        if not isinstance(self.trigger_price, Decimal):
            self.trigger_price = quantize_price(self.trigger_price)
            
        # Logical Guardrails
        if not self.quantity.is_finite():
            raise ValueError(f"CRITICAL: Order quantity must be finite. Got {self.quantity}")
        if self.quantity <= 0:
            raise ValueError(f"CRITICAL: Order quantity must be positive. Got {self.quantity}")
        
        if not self.price.is_finite():
            raise ValueError(f"CRITICAL: Price must be finite. Got {self.price}")
        if self.price < 0:
            raise ValueError("CRITICAL: Price cannot be negative.")

    def to_dict(self) -> Dict[str, Any]:
        """Optimized serialization for MsgPack."""
        return {
            "order_id": self.order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity), # Serialize Decimal as string
            "order_type": self.order_type.value,
            "price": str(self.price),
            "trigger_price": str(self.trigger_price),
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "tag": self.tag,
            "error_message": self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Robust Factory Method for Deserialization."""
        return cls(
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            quantity=Decimal(data['quantity']),
            order_type=OrderType(data['order_type']),
            order_id=data['order_id'],
            price=Decimal(data['price']),
            trigger_price=Decimal(data.get('trigger_price', "0.00")),
            exchange_order_id=data.get('exchange_order_id'),
            status=OrderStatus(data['status']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            tag=data.get('tag'),
            error_message=data.get('error_message')
        )

    @staticmethod
    def create_id() -> str:
        return str(uuid.uuid4())

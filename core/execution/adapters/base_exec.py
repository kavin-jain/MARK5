from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from decimal import Decimal, ROUND_HALF_UP
from core.execution.schemas import Order, Position

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
NSE_TICK_SIZE = Decimal("0.05")

class BaseExecutor(ABC):
    """
    Standard Interface for all Broker Adapters.
    Now enforces Schema return types (Order/Position) instead of raw Dicts.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.config = config
        self.name = name

    @staticmethod
    def quantize_tick(price: Decimal) -> Decimal:
        """
        CRITICAL: Rounds price to nearest 0.05 (NSE Standard).
        Prevents 'InputException' from broker APIs.
        """
        if not isinstance(price, Decimal):
            price = Decimal(str(price))
        # Logic: (Price / 0.05).quantize(1) * 0.05
        return (price / NSE_TICK_SIZE).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * NSE_TICK_SIZE

    @abstractmethod
    def place_order(self, order: Order) -> bool:
        """Send order to exchange. Return True if accepted."""
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel open order."""
        pass
        
    @abstractmethod
    def modify_order(self, order_id: str, price: Decimal = None, qty: int = None) -> bool:
        """Modify open order."""
        pass
        
    @abstractmethod
    def fetch_orders(self) -> List[Order]:
        """
        Reconciliation: Get daily order book converted to Internal Schema.
        """
        pass
        
    @abstractmethod
    def fetch_positions(self) -> List[Position]:
        """
        Reconciliation: Get net positions converted to Internal Schema.
        """
        pass

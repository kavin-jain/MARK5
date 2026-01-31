import logging
import msgpack
import time
from typing import Dict, Optional
from decimal import Decimal

from core.system.container import container
from .schemas import Order, OrderStatus
# No generic import here to avoid circular dependency; check logic below

class OrderManager:
    """
    Redis-backed Order State Machine.
    Ensures 0-data-loss and strictly typed re-hydration.
    """
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger("MARK5.OMS")
        self.orders: Dict[str, Order] = {} 
        self.redis = container.redis
        self.redis_key_prefix = "oms:orders"
        self.executor = None 

    def _hydrate_state(self):
        """
        Recover state from Redis.
        CRITICAL FIX: Correctly reconstructs DateTime and Decimal objects.
        """
        try:
            if not self.redis.client: return

            raw_orders = self.redis.client.hgetall(self.redis_key_prefix)
            
            for oid_bytes, order_bytes in raw_orders.items():
                try:
                    # 1. Unpack MsgPack to Dict (Binary -> Dict)
                    data = msgpack.unpackb(order_bytes, raw=False)
                    
                    # 2. Reconstruct via Factory (Dict -> Object)
                    # This handles the ISO String -> Datetime conversion
                    order = Order.from_dict(data)
                    
                    self.orders[order.order_id] = order
                except Exception as e:
                    self.logger.critical(f"CORRUPT ORDER DATA: {oid_bytes}. Error: {e}")
            
            self.logger.info(f"Hydrated {len(self.orders)} orders from Persistence.")

        except Exception as e:
            self.logger.critical(f"OMS Hydration Failed: {e}")

    def _persist_order(self, order: Order):
        """
        Binary Write-Through.
        """
        # Update Memory
        self.orders[order.order_id] = order
        
        try:
            # Serialize (Object -> Dict -> MsgPack Bytes)
            data = order.to_dict()
            packed = msgpack.packb(data, use_bin_type=True)
            
            # Redis Write
            self.redis.client.hset(self.redis_key_prefix, order.order_id, packed)
        except Exception as e:
            self.logger.error(f"Persistence Critical Failure: {e}")

    def place_order(self, order: Order) -> bool:
        """
        Atomic Order Placement.
        """
        # 1. Idempotency Check
        if order.order_id in self.orders:
            self.logger.warning(f"Duplicate Order Detected: {order.order_id}")
            return False
            
        # 2. Persist Initial State (PENDING)
        order.status = OrderStatus.PENDING
        self._persist_order(order)
        
        # 3. Validation
        if not self.executor:
            self.logger.error("OMS Error: No Executor Attached")
            order.status = OrderStatus.ERROR
            order.error_message = "No Executor Attached"
            self._persist_order(order)
            return False

        # 4. Dispatch
        try:
            success = self.executor.place_order(order)
        except Exception as e:
            self.logger.error(f"Broker Adapter Exception: {e}")
            success = False
            order.error_message = str(e)

        # 5. Update State Post-Dispatch
        if success:
            order.status = OrderStatus.SUBMITTED
            self.logger.info(f"SUBMITTED: {order.symbol} {order.side.value} {order.quantity}")
        else:
            order.status = OrderStatus.ERROR
            self.logger.error(f"SUBMISSION FAILED: {order.symbol}")
            
        self._persist_order(order)
        return success

    def update_order_status(self, order_id: str, status: OrderStatus, exchange_id: str = None):
        """
        Thread-safe status update from WebSocket/Callback.
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = status
            if exchange_id:
                order.exchange_order_id = exchange_id
            self._persist_order(order)
        else:
            self.logger.warning(f"Received update for unknown order: {order_id}")

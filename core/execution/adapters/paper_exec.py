import logging
import uuid
import random
import time
import msgpack
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime

from .base_exec import BaseExecutor
from core.execution.schemas import Order, OrderStatus, OrderSide, OrderType, Position
from core.system.container import container

class PaperExecutor(BaseExecutor):
    def __init__(self, config: Dict):
        super().__init__("Paper", config)
        self.logger = logging.getLogger("MARK5.Exec.Paper")
        # Use getattr for Pydantic model access. Fallback to defaults.
        self.latency_ms = getattr(config, 'latency_budget_ms', 50) 
        # Slippage BPS not in ExecutionConfig, default to 5
        self.slippage_bps = Decimal(str(getattr(config, 'slippage_bps', 5))) 
        self.redis = container.redis
        
        # In-Memory Virtual Order Book
        self.active_orders: Dict[str, Order] = {}
        self.fills: List[Dict] = []
        
        # Virtual Position Tracker (for reconciliation testing)
        self._virtual_positions: Dict[str, Position] = {}

    def place_order(self, order: Order) -> bool:
        # 1. Simulate Network Latency
        # In production, we don't block. In paper, we sleep to test timeouts.
        time.sleep(self.latency_ms / 1000.0)
        
        order.exchange_order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        order.status = OrderStatus.OPEN
        
        self.active_orders[order.exchange_order_id] = order
        self.logger.info(f"PAPER OPEN: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
        
        # 2. Try Immediate Match (Market Orders)
        if order.order_type in [OrderType.MARKET, OrderType.SL_M]:
            self._fill_market_order(order)
            
        return True

    def _get_live_price(self, symbol: str) -> Decimal:
        """Fetch latest tick from Redis, ensure Decimal return."""
        if not self.redis.client: return Decimal("0.00")
        
        # Use a distinct key for MD (Market Data)
        tick_bytes = self.redis.client.get(f"md:LTP:{symbol}")
        if tick_bytes:
            try:
                # Assuming raw bytes of the float string or msgpack
                # Robust approach: Try decode to float then Decimal
                val = float(tick_bytes) # Redis usually stores strings/bytes
                return Decimal(str(val))
            except Exception:
                return Decimal("0.00")
        return Decimal("0.00")

    def calculate_slippage(self, quantity: Decimal, price: Decimal) -> Decimal:
        """
        Calculate impact cost based on order size and volatility noise.
        Slippage = Price * (BPS / 10000) * Noise
        """
        # Base Slippage
        slippage_amt = price * (self.slippage_bps / Decimal("10000"))
        
        # Add random noise (0.5x to 1.5x) to simulate market depth variance
        noise_factor = Decimal(str(random.uniform(0.5, 1.5)))
        return slippage_amt * noise_factor

    def _fill_market_order(self, order: Order):
        market_price = self._get_live_price(order.symbol)
        if market_price <= 0:
            self.logger.warning(f"PAPER: No Liquidity (Price=0) for {order.symbol}")
            return

        # Calculate Slippage
        actual_slippage = self.calculate_slippage(order.quantity, market_price)

        if order.side == OrderSide.BUY:
            raw_fill_price = market_price + actual_slippage
        else:
            raw_fill_price = market_price - actual_slippage

        # 4. Quantize to Tick Size
        final_fill_price = self.quantize_tick(raw_fill_price)

        self._execute_fill(order, final_fill_price)

    def _execute_fill(self, order: Order, price: Decimal):
        order.status = OrderStatus.COMPLETE
        order.price = price # Update order object with actual fill price
        
        # Update Virtual Positions (for fetch_positions support)
        self._update_virtual_position(order)

        # Remove from active book
        if order.exchange_order_id in self.active_orders:
            del self.active_orders[order.exchange_order_id]
            
        self.logger.info(f"PAPER FILL: {order.symbol} @ {price} (Slippage incl.)")
        
        # Notify Execution Engine (In a real reactor, this is an event)
        # In this synchronous mock, the Engine calls us, so we just update state.

    def _update_virtual_position(self, order: Order):
        """Mock Position Tracking to test Reconciliation logic"""
        # (Simplified logic: assumes simple net qty tracking)
        sym = order.symbol
        if sym not in self._virtual_positions:
            self._virtual_positions[sym] = Position(sym, Decimal("0"), Decimal("0"))
        
        pos = self._virtual_positions[sym]
        qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        pos.quantity += qty

    def cancel_order(self, order_id: str) -> bool:
        # Search by internal ID or Exchange ID
        target_oid = None
        for eid, o in self.active_orders.items():
            if o.order_id == order_id or eid == order_id:
                target_oid = eid
                break
        
        if target_oid:
            self.active_orders[target_oid].status = OrderStatus.CANCELLED
            del self.active_orders[target_oid]
            return True
        return False

    def modify_order(self, order_id: str, price: Decimal = None, qty: int = None) -> bool:
        # Not implemented for paper yet
        return False

    def fetch_orders(self) -> List[Order]:
        return list(self.active_orders.values())

    def fetch_positions(self) -> List[Position]:
        return list(self._virtual_positions.values())

    def on_tick(self, tick: Dict):
        """
        Engine calls this on every market tick.
        Checks Limit Orders against LTP.
        """
        symbol = tick.get('symbol')
        try:
            ltp = Decimal(str(tick.get('last_price', 0)))
        except:
            return

        if ltp <= 0: return

        # Optimize: Don't iterate all orders. In prod, use a OrderBook per symbol.
        # For <1000 orders, list iteration is "okay" but not "perfect".
        # Strict Limit Logic: 
        # BUY LIMIT fills if LTP <= Price
        # SELL LIMIT fills if LTP >= Price
        
        # Create a copy to allow deletion during iteration
        for eid, order in list(self.active_orders.items()):
            if order.symbol != symbol: continue
            if order.order_type != OrderType.LIMIT: continue

            should_fill = False
            if order.side == OrderSide.BUY and ltp <= order.price:
                should_fill = True
            elif order.side == OrderSide.SELL and ltp >= order.price:
                should_fill = True
            
            if should_fill:
                # Limit orders fill at their LIMIT price (Passive) 
                # or LTP? In matching engine, if aggressive, it's LTP. 
                # If resting, it's Limit Price.
                # We assume Resting for simplicity in Paper.
                self._execute_fill(order, order.price)

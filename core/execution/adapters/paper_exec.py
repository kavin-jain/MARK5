"""
MARK5 PAPER EXECUTOR v9.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-02] v9.0: Bug fix
  • FIX L-05: Removed time.sleep() from place_order(). System rules forbid
    sleep() in any trading code path. The latency simulation provided zero
    value — it just blocked the calling thread.
- [2026-02-06] v8.0: Configurable slippage, in-memory order book

TRADING ROLE: Simulated execution for paper trading
SAFETY LEVEL: LOW - Testing environment only

FEATURES:
✅ Configurable slippage (BPS)
✅ In-memory order book and fill tracking
✅ Limit order matching on tick updates
"""

import logging
import random
import uuid
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Dict, List

from .base_exec import BaseExecutor
from core.execution.schemas import (
    Order, OrderStatus, OrderSide, OrderType, Position,
)
from core.system.container import container


class PaperExecutor(BaseExecutor):
    def __init__(self, config: Dict):
        super().__init__("Paper", config)
        self.logger = logging.getLogger("MARK5.Exec.Paper")

        # Slippage in basis points (not from config to keep it simple)
        self.slippage_bps = Decimal(str(getattr(config, "slippage_bps", 5)))

        self.redis = container.redis

        # In-memory state
        self.active_orders: Dict[str, Order] = {}
        self._virtual_positions: Dict[str, Position] = {}

    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> bool:
        # FIX L-05: Removed time.sleep(self.latency_ms / 1000.0)
        # sleep() in a trading code path violates MARK5 coding rules.
        order.exchange_order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        order.status = OrderStatus.OPEN
        self.active_orders[order.exchange_order_id] = order

        self.logger.info(
            f"PAPER OPEN: {order.symbol} {order.side.value} "
            f"qty={order.quantity} @ {order.price}"
        )

        # Immediate fill for market orders
        if order.order_type in (OrderType.MARKET, OrderType.SL_M):
            self._fill_market_order(order)

        return True

    def _get_live_price(self, symbol: str) -> Decimal:
        """Fetch latest LTP from Redis. Returns Decimal(0) if unavailable."""
        if not (self.redis and self.redis.client):
            return Decimal("0.00")
        try:
            raw = self.redis.client.get(f"md:LTP:{symbol}")
            if raw:
                return Decimal(str(float(raw)))
        except Exception:
            pass
        return Decimal("0.00")

    def calculate_slippage(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Slippage = price × BPS/10000 × noise(0.5–1.5)."""
        base_slippage = price * (self.slippage_bps / Decimal("10000"))
        noise = Decimal(str(random.uniform(0.5, 1.5)))
        return base_slippage * noise

    def _fill_market_order(self, order: Order) -> None:
        market_price = self._get_live_price(order.symbol)
        if market_price <= 0:
            self.logger.warning(
                f"PAPER: No price data for {order.symbol} — fill deferred."
            )
            return

        slippage = self.calculate_slippage(order.quantity, market_price)
        if order.side == OrderSide.BUY:
            fill_price = market_price + slippage
        else:
            fill_price = market_price - slippage

        final_price = self.quantize_tick(fill_price)
        self._execute_fill(order, final_price)

    def _execute_fill(self, order: Order, price: Decimal) -> None:
        order.price = price
        order.status = OrderStatus.COMPLETE
        self._update_virtual_position(order)
        if order.exchange_order_id in self.active_orders:
            del self.active_orders[order.exchange_order_id]
        self.logger.info(f"PAPER FILL: {order.symbol} @ {price}")

    def _update_virtual_position(self, order: Order) -> None:
        sym = order.symbol
        if sym not in self._virtual_positions:
            self._virtual_positions[sym] = Position(
                sym, Decimal("0"), Decimal("0")
            )
        pos = self._virtual_positions[sym]
        delta = order.quantity if order.side == OrderSide.BUY else -order.quantity
        pos.quantity += delta

    def on_tick(self, tick: Dict) -> None:
        """Called on every market tick to check limit order fills."""
        symbol = tick.get("symbol")
        try:
            ltp = Decimal(str(tick.get("last_price", 0)))
        except (TypeError, ValueError, InvalidOperation):
            return
        if ltp <= 0:
            return

        for eid, order in list(self.active_orders.items()):
            if order.symbol != symbol or order.order_type != OrderType.LIMIT:
                continue
            should_fill = (
                (order.side == OrderSide.BUY  and ltp <= order.price) or
                (order.side == OrderSide.SELL and ltp >= order.price)
            )
            if should_fill:
                self._execute_fill(order, order.price)

    def cancel_order(self, order_id: str) -> bool:
        for eid, order in list(self.active_orders.items()):
            if order.order_id == order_id or eid == order_id:
                order.status = OrderStatus.CANCELLED
                del self.active_orders[eid]
                return True
        return False

    def modify_order(self, order_id: str, price: Decimal = None, qty: int = None) -> bool:
        return False  # Not implemented for paper mode

    def fetch_orders(self) -> List[Order]:
        return list(self.active_orders.values())

    def fetch_positions(self) -> List[Position]:
        return list(self._virtual_positions.values())
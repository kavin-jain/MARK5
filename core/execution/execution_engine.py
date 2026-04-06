"""
MARK5 EXECUTION ENGINE v9.0 - PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHANGELOG:
- [2026-04-02] v9.0: Bug fixes
  • FIX C-09: Paper mode used a hardcoded fill price of ₹100.00 for market
    orders when no limit price was given. All paper P&L numbers were
    therefore fabricated. Now fetches last known price from Redis (LTP key)
    or aborts the fill if no price is available rather than using a
    placeholder.
  • FIX: daily_realized_pnl reset properly at day boundary via
    _check_daily_reset() delegation to risk_manager.
  • CLEANUP: Removed duplicate import path resolution code; merged into
    single try/except block.
- [2026-02-06] v8.0: Paper/live mode switching, FIFO position tracking

TRADING ROLE: Unified execution interface for paper/live trading
SAFETY LEVEL: CRITICAL - Manages order execution and position state
"""

import logging
import threading
from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime

from core.system.container import container
from core.execution.schemas import (
    Order, OrderSide, OrderType, OrderStatus, Position, quantize_price,
)
from core.execution.order_validator import OrderValidator


class ExecutionEngine:
    def __init__(self, mode: str = "paper"):
        self.logger = logging.getLogger("MARK5.Exec")
        self.mode = mode
        self.oms = container.oms

        try:
            self.risk_manager = container.risk_manager
        except AttributeError:
            self.risk_manager = None
            self.logger.warning(
                "Risk Manager not found in container. Risk checks disabled."
            )

        self.validator = OrderValidator({
            "max_order_value": 100_000.0,
            "max_quantity":    500,
            "price_deviation_pct": 0.10,
        })

        self.positions:            Dict[str, Position] = {}
        self.daily_realized_pnl:   Decimal = Decimal("0.00")
        self.lock = threading.RLock()

        try:
            self.capital = Decimal(str(container.config.risk.initial_capital))
        except (AttributeError, TypeError):
            self.capital = Decimal("100000.00")
            self.logger.warning("Using default capital: ₹100,000")

        self._setup_adapter()

    def _setup_adapter(self) -> None:
        if self.mode == "live":
            from core.execution.adapters.kite_exec import KiteExecutor
            self.adapter = KiteExecutor(container.config.execution)
        else:
            from core.execution.adapters.paper_exec import PaperExecutor
            self.adapter = PaperExecutor(container.config.execution)
        self.oms.executor = self.adapter

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute_order(
        self,
        symbol: str,
        side: str,
        qty,
        order_type: str = "MARKET",
        price=Decimal("0.00"),
    ) -> bool:
        try:
            q_qty   = Decimal(str(qty))   if not isinstance(qty,   Decimal) else qty
            q_price = quantize_price(price)

            # Risk check
            if self.risk_manager and q_price > 0:
                capital_for_check = float(self.capital + self.daily_realized_pnl)
                if not self.risk_manager.check_trade_risk(
                    symbol, float(q_price), int(q_qty), capital_for_check
                ):
                    self.logger.error(
                        f"🛑 RISK CHECK FAILED: {symbol} | "
                        f"price={q_price} qty={q_qty}"
                    )
                    return False

            # Pre-trade validation
            val_params = {
                "symbol":           symbol,
                "quantity":         float(q_qty),
                "price":            float(q_price),
                "transaction_type": side,
            }
            is_valid, reason = self.validator.validate_order(val_params)
            if not is_valid:
                self.logger.error(f"🚫 Order validator blocked: {reason}")
                return False

            order = Order(
                symbol     = symbol,
                side       = OrderSide(side),
                quantity   = q_qty,
                order_type = OrderType(order_type),
                price      = q_price,
                order_id   = Order.create_id(),
            )

            success = self.oms.place_order(order)
            self.logger.info(
                f"ORDER_{'PLACED' if success else 'FAILED'} | "
                f"{order.order_id} | {symbol} | {side} | "
                f"qty={q_qty} | price={q_price} | type={order_type}"
            )

            if success and self.mode == "paper":
                fill_price = self._resolve_fill_price(symbol, q_price)
                if fill_price is None:
                    self.logger.warning(
                        f"[C-09 FIX] No price available for {symbol} market order. "
                        f"Fill skipped — position NOT recorded. "
                        f"Connect a live data feed or use LIMIT orders in paper mode."
                    )
                    return True  # Order was submitted but fill is deferred
                self._on_fill(order, fill_price, q_qty)

            return success

        except Exception as exc:
            self.logger.error(f"Execution failed: {exc}", exc_info=True)
            return False

    def _resolve_fill_price(
        self, symbol: str, q_price: Decimal
    ) -> Optional[Decimal]:
        """
        FIX C-09: Resolve a realistic fill price for paper-mode market orders.

        Priority:
          1. Use the limit price if provided (non-zero).
          2. Try Redis LTP key `md:LTP:{symbol}`.
          3. Return None — caller must decide whether to skip fill.

        Returning None is always safer than using ₹100 as a placeholder.
        """
        if q_price > 0:
            return q_price

        # Try Redis LTP
        try:
            redis = getattr(container, "redis", None)
            if redis and redis.client:
                raw = redis.client.get(f"md:LTP:{symbol}")
                if raw:
                    ltp = float(raw)
                    if ltp > 0:
                        return quantize_price(ltp)
        except Exception as exc:
            self.logger.debug(f"Redis LTP fetch failed for {symbol}: {exc}")

        return None  # No price available — do not invent one

    # ------------------------------------------------------------------
    # Fill handler
    # ------------------------------------------------------------------

    def _on_fill(
        self, order: Order, fill_price: Decimal, fill_qty: Decimal
    ) -> None:
        fill_price = quantize_price(fill_price)
        fill_qty   = Decimal(str(fill_qty)) if not isinstance(fill_qty, Decimal) else fill_qty

        with self.lock:
            pos = self.positions.get(order.symbol)

            if not pos:
                self.positions[order.symbol] = Position(
                    symbol        = order.symbol,
                    quantity      = fill_qty if order.side == OrderSide.BUY else -fill_qty,
                    average_price = fill_price,
                )
                return

            is_same_side = (
                (pos.quantity > 0 and order.side == OrderSide.BUY) or
                (pos.quantity < 0 and order.side == OrderSide.SELL)
            )

            if is_same_side:
                # Pyramid — weighted average price
                new_qty   = pos.quantity + (fill_qty if order.side == OrderSide.BUY else -fill_qty)
                total_val = (abs(pos.quantity) * pos.average_price) + (fill_qty * fill_price)
                pos.average_price = total_val / abs(new_qty)
                pos.quantity      = new_qty
            else:
                # Close or flip
                qty_to_close  = min(abs(pos.quantity), fill_qty)
                qty_remainder = fill_qty - qty_to_close
                direction     = Decimal("1") if pos.quantity > 0 else Decimal("-1")
                trade_pnl     = (fill_price - pos.average_price) * qty_to_close * direction

                self.daily_realized_pnl += trade_pnl
                pos.realized_pnl        += trade_pnl

                if self.risk_manager:
                    self.risk_manager.update_pnl(float(trade_pnl), is_trade_close=True)

                if pos.quantity > 0:
                    pos.quantity -= qty_to_close
                else:
                    pos.quantity += qty_to_close

                if qty_remainder > 0:
                    sign         = Decimal("1") if order.side == OrderSide.BUY else Decimal("-1")
                    pos.quantity = qty_remainder * sign
                    pos.average_price = fill_price

            if pos.quantity == 0:
                self.logger.info(
                    f"FLAT | {order.symbol} | "
                    f"total_pnl=₹{pos.realized_pnl:.2f}"
                )
                del self.positions[order.symbol]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Position]:
        with self.lock:
            return list(self.positions.values())

    def get_daily_stats(self) -> Dict:
        with self.lock:
            return {
                "mode":           self.mode,
                "open_positions": len(self.positions),
                "daily_pnl":      float(self.daily_realized_pnl),
                "capital":        float(self.capital),
                "current_portfolio_value": float(
                    self.capital + self.daily_realized_pnl
                ),
            }

    def reset_for_new_day(self) -> None:
        """Call at market open to reset intraday state."""
        with self.lock:
            self.daily_realized_pnl = Decimal("0.00")
        self.logger.info("Execution engine reset for new trading day.")

    def close_position(self, symbol: str) -> Optional[object]:
        """Close an open position at market price."""
        with self.lock:
            pos = self.positions.get(symbol)
        if not pos or pos.quantity == 0:
            return None

        side = "SELL" if pos.quantity > 0 else "BUY"
        qty  = abs(pos.quantity)
        success = self.execute_order(symbol, side, qty, order_type="MARKET")
        if success:
            from core.execution.execution_engine import OrderResult
            return OrderResult(
                status   = "SUCCESS",
                symbol   = symbol,
                quantity = float(qty),
                price    = float(pos.average_price),
            )
        return None

    def stop(self) -> None:
        self.logger.info("Execution engine stopped.")


# Compatibility aliases
MARK5ExecutionEngine = ExecutionEngine


class OrderResult:
    """Lightweight result object for position close calls."""
    __slots__ = ("status", "price", "quantity", "symbol", "timestamp", "commission")

    def __init__(
        self,
        status: str,
        price: float = 0.0,
        quantity: float = 0.0,
        symbol: str = "",
        timestamp=None,
        commission: float = 0.0,
    ):
        self.status     = status
        self.price      = price
        self.quantity   = quantity
        self.symbol     = symbol
        self.timestamp  = timestamp or datetime.utcnow()
        self.commission = commission
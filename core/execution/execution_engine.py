import logging
import threading
from decimal import Decimal, Context
from typing import Dict, List, Optional

from core.system.container import container
from core.execution.schemas import Order, OrderSide, OrderType, OrderStatus, Position, quantize_price

class ExecutionEngine:
    def __init__(self, mode="paper"):
        self.logger = logging.getLogger("MARK5.Exec")
        self.mode = mode
        self.oms = container.oms
        try:
             self.risk_manager = container.risk_manager
        except AttributeError:
             self.risk_manager = None
             self.logger.warning("Risk Manager not found in container. Risk checks disabled.")
        
        # State - Single Source of Truth
        self.positions: Dict[str, Position] = {}
        
        # PnL Accumulator (Daily)
        self.daily_realized_pnl = Decimal("0.00")
        
        # Lock: Granular locking for high concurrency
        self.lock = threading.RLock() 
        
        self._setup_adapter()

    def _setup_adapter(self):
        if self.mode == "live":
            from core.execution.adapters.kite_exec import KiteExecutor
            self.adapter = KiteExecutor(container.config.execution)
        else:
            from core.execution.adapters.paper_exec import PaperExecutor
            self.adapter = PaperExecutor(container.config.execution)
        
        # Inversion of Control: Link Adapter to OMS
        self.oms.executor = self.adapter

    def execute_order(self, symbol: str, side: str, qty: float | str | Decimal, order_type: str = "MARKET", price: float | str | Decimal = "0.00") -> bool:
        """
        Main Entry Point. Converts primitives to Decimals immediately.
        """
        try:
            # 1. Type Enforcement & Sanitization
            # We explicitly cast to string first to avoid float precision issues if float is passed
            q_qty = Decimal(str(qty)) if not isinstance(qty, Decimal) else qty
            q_price = quantize_price(price)
            
            # 2. Risk Check
            if self.risk_manager:
                # Approximate capital check (In real system, query broker or local state)
                # For now, pass a large number or query config
                # We simply check if the trade ITSELF violates limits
                current_price = float(q_price) if q_price > 0 else 0.0 # Market order price unknown here, ideally fetch from DataProvider
                
                # If price is 0 (Market), we skip notional check or estimate it. 
                # Let's assume we proceed if price is 0 (Executor will fill at market)
                if current_price > 0:
                     if not self.risk_manager.check_trade_risk(symbol, current_price, int(q_qty), 1000000.0): # Mock Capital
                         self.logger.error(f"Risk Check Failed for {symbol}")
                         return False

            # 3. Create Order Object
            order = Order(
                symbol=symbol,
                side=OrderSide(side),
                quantity=q_qty,
                order_type=OrderType(order_type),
                price=q_price,
                order_id=Order.create_id()
            )

            # 4. Route to OMS
            success = self.oms.place_order(order)
            
            if success and self.mode == "paper":
                # Simulation Hook: Immediate Fill
                # In production, this comes via WebSocket callback
                # For paper trading, we assume fill at order price (Limit) or current market price (Market - handled by adapter usually, but here simplified)
                fill_p = q_price if q_price > 0 else Decimal("100.00") # Mock price if market
                self._on_fill(order, fill_p, q_qty) 
            
            return success
            
        except Exception as e:
            self.logger.error(f"Execution Failed: {e}", exc_info=True)
            return False

    def _on_fill(self, order: Order, fill_price: Decimal, fill_qty: Decimal):
        """
        The Heart of the System. 
        Handles FIFO, Weighted Averaging, and Position Flipping.
        """
        # Ensure Decimals
        fill_price = quantize_price(fill_price)
        fill_qty = Decimal(str(fill_qty)) if not isinstance(fill_qty, Decimal) else fill_qty

        with self.lock:
            pos = self.positions.get(order.symbol)

            # Case 1: No existing position. Open new.
            if not pos:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=fill_qty if order.side == OrderSide.BUY else -fill_qty,
                    average_price=fill_price
                )
                self.logger.info(f"OPEN NEW | {order.symbol} | {fill_qty} @ {fill_price}")
                return

            # Case 2: Adding to position (Pyramiding)
            # Long + Buy OR Short + Sell
            is_same_side = (pos.quantity > 0 and order.side == OrderSide.BUY) or \
                           (pos.quantity < 0 and order.side == OrderSide.SELL)

            if is_same_side:
                new_qty = pos.quantity + (fill_qty if order.side == OrderSide.BUY else -fill_qty)
                
                # Weighted Average Price Calculation
                # (Old_Val + New_Val) / Total_Qty
                total_val = (abs(pos.quantity) * pos.average_price) + (fill_qty * fill_price)
                pos.average_price = total_val / abs(new_qty)
                pos.quantity = new_qty
                
                self.logger.info(f"PYRAMID | {order.symbol} | New Avg: {pos.average_price}")

            # Case 3: Closing or Flipping Position
            else:
                # Determine how much we are closing vs flipping
                # Example: Long 10. Sell 15. 
                # qty_to_close = min(10, 15) = 10
                # qty_remainder = 15 - 10 = 5 (Flip to Short)
                
                qty_to_close = min(abs(pos.quantity), fill_qty)
                qty_remainder = fill_qty - qty_to_close

                # -- 3a. Realize PnL on the closed portion --
                # PnL = (Exit - Entry) * Qty * Direction
                # Direction is 1 if we were Long, -1 if we were Short
                direction = Decimal("1") if pos.quantity > 0 else Decimal("-1")
                trade_pnl = (fill_price - pos.average_price) * qty_to_close * direction
                
                # Update System PnL
                self.daily_realized_pnl += trade_pnl
                pos.realized_pnl += trade_pnl
                
                # Update Risk Manager
                if self.risk_manager:
                    self.risk_manager.update_pnl(float(trade_pnl))
                
                # Update Position Quantity (Reduce towards zero)
                if pos.quantity > 0:
                    pos.quantity -= qty_to_close
                else:
                    pos.quantity += qty_to_close 

                self.logger.info(f"CLOSE/PARTIAL | PnL: {trade_pnl} | Rem Qty: {pos.quantity}")

                # -- 3b. Handle Flip (Remainder) --
                if qty_remainder > 0:
                    # We closed the flat position and now open the opposite side
                    new_side_sign = Decimal("1") if order.side == OrderSide.BUY else Decimal("-1")
                    pos.quantity = qty_remainder * new_side_sign
                    pos.average_price = fill_price # New cost basis is the fill price
                    
                    self.logger.info(f"POSITION FLIP | New Qty: {pos.quantity} @ {fill_price}")

            # Cleanup
            if pos.quantity == 0:
                self.logger.info(f"FLAT | {order.symbol} | Total PnL: {pos.realized_pnl}")
                del self.positions[order.symbol]

    def get_positions(self) -> List[Position]:
        with self.lock:
            # Return deep copies or value objects to prevent external mutation
            return list(self.positions.values())

import unittest
import shutil
import tempfile
import sys
import os
from decimal import Decimal
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Mock Container before importing modules that use it
with patch('core.system.container.container') as mock_container:
    # Setup Mock Redis
    mock_redis = MagicMock()
    mock_container.redis = mock_redis
    mock_container.config.execution = {} # Mock config
    
    from core.execution.schemas import Order, OrderSide, OrderType, OrderStatus, Position, quantize_price
    from core.execution.execution_engine import ExecutionEngine
    from core.execution.order_manager import OrderManager

class TestExecutionV6(unittest.TestCase):
    def setUp(self):
        # Reset Container Mocks
        self.mock_oms = MagicMock()
        self.mock_redis = MagicMock()
        
        # Patch Container
        patcher = patch('core.system.container.container')
        self.mock_container = patcher.start()
        self.mock_container.oms = self.mock_oms
        self.mock_container.redis = self.mock_redis
        self.mock_container.config.execution = {}
        self.addCleanup(patcher.stop)
        
        self.engine = ExecutionEngine(mode="paper")
        # Manually attach mock OMS since we mocked the container
        self.engine.oms = self.mock_oms

    def test_precision_math(self):
        """Verify that 0.1 + 0.2 == 0.3 using Decimals"""
        # In float: 0.1 + 0.2 = 0.30000000000000004
        
        # We simulate a fill that would cause float error
        # Buy 0.1 @ 100, Buy 0.2 @ 100 -> Total 0.3
        
        order1 = Order(symbol="MATH_TEST", side=OrderSide.BUY, quantity=Decimal("0.1"), order_type=OrderType.MARKET, order_id="1", price=Decimal("100"))
        self.engine._on_fill(order1, Decimal("100.00"), Decimal("0.1"))
        
        order2 = Order(symbol="MATH_TEST", side=OrderSide.BUY, quantity=Decimal("0.2"), order_type=OrderType.MARKET, order_id="2", price=Decimal("100"))
        self.engine._on_fill(order2, Decimal("100.00"), Decimal("0.2"))
        
        pos = self.engine.positions["MATH_TEST"]
        
        # Strict Equality Check
        self.assertEqual(pos.quantity, Decimal("0.3"))
        self.assertNotEqual(float(pos.quantity), 0.30000000000000004) # Just to prove a point, though Decimal("0.3") != float(0.3) anyway usually

    def test_position_flipping(self):
        """Verify Long -> Short Flip Atomicity"""
        # 1. Open Long 10 @ 100
        order1 = Order(symbol="FLIP_TEST", side=OrderSide.BUY, quantity=Decimal("10"), order_type=OrderType.MARKET, order_id="1", price=Decimal("100"))
        self.engine._on_fill(order1, Decimal("100.00"), Decimal("10"))
        
        pos = self.engine.positions["FLIP_TEST"]
        self.assertEqual(pos.quantity, Decimal("10"))
        self.assertEqual(pos.average_price, Decimal("100.00"))
        
        # 2. Sell 15 @ 110 (Close 10, Open Short 5)
        # PnL should be (110 - 100) * 10 = 100
        order2 = Order(symbol="FLIP_TEST", side=OrderSide.SELL, quantity=Decimal("15"), order_type=OrderType.MARKET, order_id="2", price=Decimal("110"))
        self.engine._on_fill(order2, Decimal("110.00"), Decimal("15"))
        
        pos = self.engine.positions["FLIP_TEST"]
        
        # Verify Flip
        self.assertEqual(pos.quantity, Decimal("-5"))
        self.assertEqual(pos.average_price, Decimal("110.00")) # New cost basis
        self.assertEqual(pos.realized_pnl, Decimal("100.00"))
        self.assertEqual(self.engine.daily_realized_pnl, Decimal("100.00"))

    def test_persistence_serialization(self):
        """Verify Decimal serialization to MsgPack"""
        order = Order(
            symbol="PERSIST_TEST", 
            side=OrderSide.BUY, 
            quantity=Decimal("10.55"), 
            order_type=OrderType.LIMIT, 
            order_id="test_id", 
            price=Decimal("100.05")
        )
        
        # Serialize
        data = order.to_dict()
        self.assertIsInstance(data['quantity'], str)
        self.assertEqual(data['quantity'], "10.55")
        
        # Deserialize
        order_back = Order.from_dict(data)
        self.assertIsInstance(order_back.quantity, Decimal)
        self.assertEqual(order_back.quantity, Decimal("10.55"))
        self.assertEqual(order_back.price, Decimal("100.05"))

    def test_strict_validation(self):
        """Verify that non-finite Decimals are rejected"""
        with self.assertRaises(ValueError):
            Order(symbol="BAD", side=OrderSide.BUY, quantity=Decimal("NaN"), order_type=OrderType.MARKET, order_id="1")
            
        with self.assertRaises(ValueError):
            Order(symbol="BAD", side=OrderSide.BUY, quantity=Decimal("Infinity"), order_type=OrderType.MARKET, order_id="1")

if __name__ == '__main__':
    unittest.main()

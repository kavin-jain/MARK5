import unittest
import sys
import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.execution.adapters.base_exec import BaseExecutor
from core.execution.adapters.paper_exec import PaperExecutor
from core.execution.adapters.kite_exec import KiteExecutor
from core.execution.schemas import Order, OrderSide, OrderType, OrderStatus

class TestAdaptersV6(unittest.TestCase):
    def setUp(self):
        self.mock_config = {"slippage_bps": 10, "simulated_latency_ms": 0}
        
        # Patch Container for PaperExecutor
        patcher = patch('core.system.container.container')
        self.mock_container = patcher.start()
        self.mock_container.redis = MagicMock()
        self.addCleanup(patcher.stop)

    def test_tick_quantization(self):
        """Verify 0.05 rounding logic"""
        # 100.03 -> 100.05 (Round Half Up)
        self.assertEqual(BaseExecutor.quantize_tick(Decimal("100.03")), Decimal("100.05"))
        # 100.01 -> 100.00
        self.assertEqual(BaseExecutor.quantize_tick(Decimal("100.01")), Decimal("100.00"))
        # 100.025 -> 100.05 (Round Half Up)
        self.assertEqual(BaseExecutor.quantize_tick(Decimal("100.025")), Decimal("100.05"))
        # 100.07 -> 100.05
        self.assertEqual(BaseExecutor.quantize_tick(Decimal("100.07")), Decimal("100.05"))

    def test_paper_slippage(self):
        """Verify Slippage Calculation"""
        adapter = PaperExecutor(self.mock_config)
        
        qty = Decimal("1000")
        price = Decimal("100.00")
        
        # Slippage = 100 * (10 / 10000) * Noise
        # Base = 0.1
        # Noise is 0.5 to 1.5
        # Range: 0.05 to 0.15
        
        slippage = adapter.calculate_slippage(qty, price)
        self.assertTrue(Decimal("0.05") <= slippage <= Decimal("0.15"))

    def test_kite_mapping(self):
        """Verify Kite Response -> Order Schema"""
        adapter = KiteExecutor({"api_key": "test", "access_token": "test"})
        adapter.kite = MagicMock()
        
        # Mock Kite Orders
        adapter.kite.orders.return_value = [{
            "tradingsymbol": "INFY",
            "transaction_type": "BUY",
            "quantity": 10,
            "tag": "test_tag",
            "price": 1500.55,
            "order_id": "kite_123",
            "status": "TRIGGER PENDING"
        }]
        
        orders = adapter.fetch_orders()
        self.assertEqual(len(orders), 1)
        o = orders[0]
        
        self.assertEqual(o.symbol, "INFY")
        self.assertEqual(o.side, OrderSide.BUY)
        self.assertEqual(o.quantity, Decimal("10"))
        self.assertEqual(o.price, Decimal("1500.55"))
        self.assertEqual(o.status, OrderStatus.PENDING) # Mapped from TRIGGER PENDING

if __name__ == '__main__':
    unittest.main()

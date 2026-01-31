"""
MARK5 DECISION ENGINE (REACTIVE)
--------------------------------
Stateless processor.
Input: Market State + AI Prediction
Output: Order Execution Request
"""

import logging
from typing import Dict, Optional

from core.system.container import container
# signal_generator and risk are injected or imported from core

class DecisionEngine:
    def __init__(self):
        self.logger = logging.getLogger("MARK5.Decision")
        self.predictor = container.get("predictor")
        self.risk = container.get("risk")
        self.oms = container.get("oms")

    def process_tick(self, ticker: str, data_snapshot: object):
        """
        Main Reactor Loop. Called whenever new data arrives.
        data_snapshot: DataFrame with latest OHLCV
        """
        # 1. Get Prediction
        # The predictor handles feature engineering internally
        ai_result = self.predictor.predict_single(data_snapshot, ticker)
        
        if not ai_result.get('prediction') and not ai_result.get('confidence'):
             # Handle error or hold signal from predictor
             return

        # 2. Logic / Signal Generation
        # Simple thresholding (Real logic goes in SignalGenerator)
        confidence = ai_result.get('confidence', 0.0)
        prediction_val = ai_result.get('prediction', 0.0)
        
        # Interpret prediction (e.g., if it's a return forecast)
        # This logic should ideally be in a SignalGenerator strategy class
        signal = "HOLD"
        
        # Example Thresholds
        BUY_THRESHOLD = 0.001 # 0.1% predicted return
        SELL_THRESHOLD = -0.001
        CONFIDENCE_THRESHOLD = 0.6
        
        if prediction_val > BUY_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
            signal = "BUY"
        elif prediction_val < SELL_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
            signal = "SELL"

        if signal == "HOLD":
            return

        # 3. Risk Check
        if hasattr(data_snapshot, 'close'):
             current_price = data_snapshot['close'].iloc[-1]
        else:
             # Fallback if data_snapshot is not a DataFrame (should not happen with correct typing)
             self.logger.error("Invalid data snapshot format")
             return

        capital = 100000.0 # Get actual capital from Account Manager or Config
        
        # Calc Stops
        sl_price = current_price * 0.99 if signal == "BUY" else current_price * 1.01
        
        # Calc Size
        qty = self.risk.calculate_position_size(capital, current_price, sl_price)
        
        if not self.risk.check_trade_risk(ticker, current_price, qty, capital):
            return

        # 4. Execute
        self.logger.info(f"⚡ ACTION: {signal} {ticker} x {qty}")
        if self.oms:
            self.oms.place_order(
                symbol=ticker,
                side=signal,
                quantity=qty,
                order_type="MARKET"
            )
        else:
            self.logger.warning("OMS not available, order skipped.")

# Legacy Alias
MARK5DecisionEngine = DecisionEngine
